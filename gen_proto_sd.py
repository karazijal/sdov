# %%
import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

import sd
import mmdata


def kmeans_fn(feats, k, device="cpu", no_downsample=False, verbose=False):
    assert not no_downsample, "No downsample not implemented for sklearn"
    f = feats.cpu().reshape(-1, feats.shape[-1]).numpy()
    start = datetime.now()
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto", verbose=int(verbose)).fit(f)
    time = datetime.now() - start
    print("Clustering took", time.total_seconds(), "seconds")
    return torch.from_numpy(kmeans.cluster_centers_).float()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_samples", type=int, default=-1)

    parser.add_argument("--feat_size", type=str, default="64,64")
    parser.add_argument("--timesteps", type=str, default="200")
    parser.add_argument("--layers", type=str, default="0,6:13,15")
    parser.add_argument("--layers1", type=str, default="")

    parser.add_argument("--mask_prefix", type=str, default="proc_masks_cutler_v3/fg_")
    parser.add_argument("--back_prefix", type=str, default="proc_masks_cutler_v3/bg_pp_")

    parser.add_argument("--overwrite", action="store_true", default=False)

    parser.add_argument("--kmeans", type=int, default=32)

    parser.add_argument("--dry-run", action="store_true", default=False, help="Don't save anything")

    parser.add_argument("--scan-dir", action="store_true", default=False)

    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--total", type=int, default=32)
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--progbar", action="store_true", default=False)
    parser.add_argument("data", type=str, default="voc", choices=["voc"])

    parser.add_argument("img_path", type=Path)
    parser.add_argument("out_name", type=str)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available() and "cuda" in args.device:
            torch.cuda.manual_seed(args.seed)

    classlist, cls2prompt = mmdata.get_data_classes(args.data)

    is_slurm = os.getenv("SLURM_JOB_ID") is not None

    data_ind_len = 5

    device = torch.device(args.device)

    feat_size = tuple([int(s.strip()) for s in args.feat_size.split(",")])

    classlist_nobg = [c for c in classlist if "background" not in c]
    iter_target = classlist_nobg
    if args.scan_dir:
        iter_target = [d.name for d in args.img_path.iterdir() if d.is_dir()]
        iter_target = sorted(iter_target)
        print("Scanning", len(iter_target), "dirs")
        for d in iter_target:
            print(d)
    total = len(iter_target)

    array_id = None
    if is_slurm:
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if array_id is not None:
            curr_id = int(array_id)
            min_id = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
            num_ids = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
            iter_target = iter_target[curr_id - min_id : total : num_ids]
            print(f"Running {curr_id} of {num_ids} jobs, {len(iter_target)} classes")

    layer_specs = []
    layers_spec_str = args.layers.split(";")
    layers1_spec_str = args.layers1.split(";")

    lsi_spec_map = {}
    for lsi, layers in enumerate(layers_spec_str):
        lsi_spec_map[lsi] = layers

        layers = layers.split(",")
        layers = [c.strip() for c in layers if len(c.strip())]
        if layers:
            layers = [
                slice(int(c), int(c) + 1) if c.isdigit() else slice(*[int(x) if x else None for x in c.split(":")])
                for c in layers
            ]
            layers = sorted(list(set(sum((sd.LAYERS[s] for s in layers), []))))

        layers1 = []
        if len(layers1_spec_str) > lsi:
            lsi_spec_map[lsi] += "+" + layers1_spec_str[lsi]
            layers1 = layers1_spec_str[lsi]
            layers1 = layers1.split(",")
            layers1 = [c.strip() for c in layers1 if len(c.strip())]
            if layers1:
                layers1 = [
                    slice(int(c), int(c) + 1) if c.isdigit() else slice(*[int(x) if x else None for x in c.split(":")])
                    for c in layers1
                ]
                layers1 = sorted(list(set(sum((sd.LAYERS1[s] for s in layers1), []))))

        layers += layers1

        for l in layers:
            print(lsi_spec_map[lsi], l)

        layer_specs.append(layers)

    timesteps = [int(t.strip()) for t in args.timesteps.strip().split(",")]

    pipe = sd.build_pipeline(device, model_key=args.model, disable_progbar=True, with_inpaint=False)

    all_proto_maps = {lsi: {t: {} for t in timesteps} for lsi in range(len(layer_specs))}
    for c in tqdm(iter_target, desc="Generating Prototype", disable=is_slurm and not args.progbar):

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        cls_dir = args.img_path / c
        imgps = [p for p in cls_dir.glob("*.png") if p.name.startswith("0")]
        imgps = sorted(imgps, key=lambda x: int(x.stem.split("_")[0]))

        if args.total > 0:
            assert len(imgps) >= args.total, f"Could not find {args.total} images ({len(imgps)}) for {c}"

        if len(imgps) == 0:
            assert False, f"No images found in {cls_dir}"

        if args.offset >= 0:
            imgps = imgps[args.offset :]

        if args.num_samples > 0:
            imgps = imgps[: args.num_samples]
            assert len(imgps) == args.num_samples, f"Could not find {args.num_samples} images ({len(imgps)}) for {c}"

        mask_dir = cls_dir / args.mask_prefix
        mask_pre = mask_dir.name.rstrip("0").rstrip("_")
        mask_dir = mask_dir.parent

        if args.back_prefix:
            back_dir = cls_dir / args.back_prefix
            back_pre = back_dir.name.rstrip("0").rstrip("_")
            back_dir = back_dir.parent

        found = 0
        all_ind_protos = {lsi: {t: {} for t in timesteps} for lsi in range(len(layer_specs))}
        for lsi, layer_spec in enumerate(layer_specs):
            for timestep in timesteps:

                proto_map = all_proto_maps[lsi][timestep]
                ind_proto = all_ind_protos[lsi][timestep]

                ind_proto_path = args.img_path / c / f"{c}_{args.out_name}_{lsi_spec_map[lsi]}_t{timestep}_proto.pt"
                if ind_proto_path.exists() and not args.overwrite:
                    try:
                        ind_proto = torch.load(ind_proto_path)
                        print(f"Loaded {ind_proto_path}")
                        proto = ind_proto["proto"]
                        bg_proto = ind_proto["proto_bg"]

                        ind_fg_protos = ind_proto["ind_proto"]
                        ind_bg_protos = ind_proto["ind_proto_bg"]

                        proto_map[c + "_ind"] = ind_fg_protos
                        proto_map[c + "_bg_ind"] = ind_bg_protos

                        if args.kmeans > 0:
                            proto_map[c + "_kmeans"] = ind_proto["kmeans"]
                            proto_map[c + "_kmeans_bg"] = ind_proto["kmeans_bg"]
                            assert len(ind_proto["kmeans"]) == args.kmeans

                        proto_map[c] = proto
                        proto_map[c + "_bg"] = bg_proto

                        found += 1
                    except KeyError:
                        print("Missing data regenerating")
                        pass

        if found == len(layer_specs) * len(timesteps):
            print("All found")
            continue

        all_cat_feats = {lsi: {t: [] for t in timesteps} for lsi in range(len(layer_specs))}
        all_cat_bg_feats = {lsi: {t: [] for t in timesteps} for lsi in range(len(layer_specs))}

        fsize = feat_size

        for imgp in tqdm(imgps, desc=f"Generating prototype for {c}", disable=is_slurm and not args.progbar):
            data_ind = imgp.stem.split("_")[0].lstrip("0")
            data_ind = int(data_ind) if data_ind else 0

            img = TF.to_tensor(Image.open(imgp))

            mask = TF.to_tensor(
                Image.open(mask_dir / ("{}_{:0>" + str(data_ind_len) + "d}.png").format(mask_pre, data_ind))
            )
            cmap = (
                F.interpolate(mask.view(1, 1, *mask.shape[-2:]), size=[*fsize], mode="bilinear")[0].bool()
            ).flatten()

            if cmap.sum() == 0:
                continue

            if args.back_prefix:
                bg_mask = TF.to_tensor(
                    Image.open(back_dir / ("{}_{:0>" + str(data_ind_len) + "d}.png").format(back_pre, data_ind))
                )
                bgmap = (
                    F.interpolate(bg_mask.view(1, 1, *bg_mask.shape[-2:]), size=[*fsize], mode="bilinear")[0].bool()
                ).flatten()
            else:
                bgmap = ~cmap

            gen = torch.Generator(pipe.device)
            gen.manual_seed(args.seed)
            latents = pipe.encode(img, generator=gen)

            feat_dim = {}
            for lsi, layer_spec in enumerate(layer_specs):
                feat_dim[lsi] = {}
                for timestep in timesteps:
                    gen = torch.Generator(pipe.device)
                    gen.manual_seed(args.seed)
                    feats = pipe.extract_features(latents, timestep, layer_spec, feat_size, generator=gen)
                    feats = feats.permute(0, 2, 3, 1).view(*feat_size, -1).detach().float().cpu()

                    feat_dim[lsi][timestep] = feats.shape[-1]

                    if data_ind <= 1 and timestep <= 0:
                        print(lsi_spec_map[lsi], timestep, feats.shape)

                    fs = feats.view(np.prod(fsize), -1)[cmap, :]
                    all_cat_feats[lsi][timestep].append(fs)

                    bg_fs = feats.view(np.prod(fsize), -1)[bgmap, :]
                    # if args.norm:
                    #     bg_fs /= bg_fs.norm(dim=-1, keepdim=True)
                    all_cat_bg_feats[lsi][timestep].append(bg_fs)

        for lsi, layer_spec in enumerate(layer_specs):
            for timestep in timesteps:

                proto_map = all_proto_maps[lsi][timestep]
                ind_proto = all_ind_protos[lsi][timestep]

                cat_feats = all_cat_feats[lsi][timestep]
                cat_bg_feats = all_cat_bg_feats[lsi][timestep]

                if len(cat_feats) == 0:
                    print(f"No features found for {c}")
                    raise ValueError(f"No features found for {c}")

                proto_map[c] = torch.zeros(0, feat_dim[lsi][timestep])
                proto_map[c + "_bg"] = torch.zeros(0, feat_dim[lsi][timestep])

                proto_map[c + "_ind"] = torch.zeros(0, feat_dim[lsi][timestep])
                proto_map[c + "_bg_ind"] = torch.zeros(0, feat_dim[lsi][timestep])

                proto_map[c + "_kmeans"] = torch.zeros(0, feat_dim[lsi][timestep])
                proto_map[c + "_kmeans_bg"] = torch.zeros(0, feat_dim[lsi][timestep])

                if len(cat_feats) > 0:
                    proto_map[c + "_ind"] = torch.stack([f.mean(0) for f in cat_feats])
                    cat_feats = torch.cat(cat_feats, 0)
                    print(f"total FG feats for {c}:", cat_feats.shape)
                    proto_map[c] = cat_feats.mean(0)

                    if args.kmeans > 0:
                        proto_map[c + "_kmeans"] = kmeans_fn(
                            cat_feats, args.kmeans, args.device, verbose=not is_slurm or args.progbar
                        )
                else:
                    print("No FG feats found for", c)
                    continue

                if len(cat_bg_feats) > 0:
                    proto_map[c + "_bg_ind"] = torch.stack([f.mean(0) for f in cat_bg_feats])
                    cat_bg_feats = torch.cat(cat_bg_feats, 0)
                    print(f"total BG feats for {c}:", cat_bg_feats.shape)
                    proto_map[c + "_bg"] = cat_bg_feats.mean(0)

                    if args.kmeans > 0:
                        proto_map[c + "_kmeans_bg"] = kmeans_fn(
                            cat_bg_feats, args.kmeans, args.device, verbose=not is_slurm or args.progbar
                        )
                else:
                    print("No BG feats found for", c)

                ind_proto["proto"] = proto_map[c]
                ind_proto["proto_bg"] = proto_map[c + "_bg"]

                ind_proto["ind_proto"] = proto_map[c + "_ind"]
                ind_proto["ind_proto_bg"] = proto_map[c + "_bg_ind"]

                ind_proto["kmeans"] = proto_map[c + "_kmeans"]
                ind_proto["kmeans_bg"] = proto_map[c + "_kmeans_bg"]

                proto_map["__layers__"] = layer_spec
                proto_map["__timestep__"] = timestep
                proto_map["__feat_size__"] = feat_size
                ind_proto["cls"] = c

                ind_proto_path = args.img_path / c / f"{c}_{args.out_name}_{lsi_spec_map[lsi]}_t{timestep}_proto.pt"
                if args.dry_run:
                    print("dry run, not saving", ind_proto_path)
                    continue
                torch.save(ind_proto, ind_proto_path)

    if array_id is None:
        for lsi, layer_spec in enumerate(layer_specs):
            for timestep in timesteps:
                proto_map = all_proto_maps[lsi][timestep]
                ind_proto = all_ind_protos[lsi][timestep]

                proto_map["__layers__"] = layer_spec
                proto_map["__timestep__"] = timestep
                proto_map["__feat_size__"] = feat_size

                if not args.dry_run:
                    torch.save(
                        proto_map,
                        args.img_path / f"{args.data}_{args.out_name}_{lsi_spec_map[lsi]}_t{timestep}_proto.pt",
                    )
                else:
                    print("dry run, not saving", args.img_path / f"{args.data}_{args.out_name}_proto.pt")


if __name__ == "__main__":
    main()
# %%
