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
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--feature_path_prefix", type=str, default="dino/dino_vitb8_8_0")
    parser.add_argument("--mask_prefix", type=str, default="proc_masks_cutler_v3/fg_")
    parser.add_argument("--back_prefix", type=str, default="proc_masks_cutler_v3/bg_pp_")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--kmeans", type=int, default=32)

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

    classlist, __ = mmdata.get_data_classes(args.data)

    is_slurm = os.getenv("SLURM_JOB_ID") is not None

    data_ind_len = 5

    device = torch.device(args.device)

    classlist_nobg = [c for c in classlist if "background" not in c]
    iter_target = classlist_nobg
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

    proto_map = {}
    for c in tqdm(iter_target, desc="Generating Prototype", disable=is_slurm and not args.progbar):

        cls_dir = args.img_path / c
        imgps = [p for p in cls_dir.glob("*.png") if p.name.startswith("0")]
        imgps = sorted(imgps, key=lambda x: int(x.stem.split("_")[0]))

        if args.total > 0:
            assert len(imgps) >= args.total, f"Could not find {args.total} images ({len(imgps)})"

        if len(imgps) == 0:
            assert False, f"No images found in {cls_dir}"

        if args.offset >= 0:
            imgps = imgps[args.offset :]

        if args.num_samples > 0:
            imgps = imgps[: args.num_samples]
        for _img in imgps:
            print(_img)

        feat_dir = cls_dir / args.feature_path_prefix
        feat_pre = feat_dir.name.rstrip("0").rstrip("_")
        feat_dir = feat_dir.parent

        mask_dir = cls_dir / args.mask_prefix
        mask_pre = mask_dir.name.rstrip("0").rstrip("_")
        mask_dir = mask_dir.parent

        if args.back_prefix:
            back_dir = cls_dir / args.back_prefix
            back_pre = back_dir.name.rstrip("0").rstrip("_")
            back_dir = back_dir.parent

        ind_proto = {}
        ind_proto_path = args.img_path / c / f"{c}_{args.out_name}_proto.pt"
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

                continue
            except (OSError, IOError) as e:
                print(ind_proto_path, e)
                raise
            except KeyError:
                print("Missing data regenerating")
                pass

        cat_feats = []
        cat_bg_feats = []

        for imgp in tqdm(imgps, desc=f"Generating prototype for {c}", disable=is_slurm and not args.progbar):
            data_ind = imgp.stem.split("_")[0].lstrip("0")
            data_ind = int(data_ind) if data_ind else 0

            fp = feat_dir / f"{feat_pre}_{data_ind:0>5d}.pt"
            try:
                feats = torch.load(fp)
            except (OSError, IOError) as e:
                print(fp, e)
                raise

            if "dino" in feat_pre or "mae" in feat_pre or "clip" in feat_pre:
                feats = feats["feats"][0]
                # print('dino feats', feats.shape)
                feats = feats.permute(1, 2, 0).detach().cpu()
            else:
                raise ValueError(f"Unknown feature type {feat_pre}")

            fsize = feats.shape[:2]
            feat_dim = feats.shape[-1]

            try:
                mask = TF.to_tensor(
                    Image.open(mask_dir / ("{}_{:0>" + str(data_ind_len) + "d}.png").format(mask_pre, data_ind))
                )
            except (OSError, IOError) as e:
                print(mask_dir / ("{}_{:0>" + str(data_ind_len) + "d}.png").format(mask_pre, data_ind), e)
                raise
            cmap = (
                F.interpolate(mask.view(1, 1, *mask.shape[-2:]), size=[*fsize], mode="bilinear")[0].bool()
            ).flatten()
            if cmap.sum() == 0:
                continue

            fs = feats.view(np.prod(fsize), -1)[cmap, :]
            cat_feats.append(fs)

            if args.back_prefix:
                try:
                    bg_mask = TF.to_tensor(
                        Image.open(back_dir / ("{}_{:0>" + str(data_ind_len) + "d}.png").format(back_pre, data_ind))
                    )
                except (OSError, IOError) as e:
                    print(back_dir / ("{}_{:0>" + str(data_ind_len) + "d}.png").format(back_pre, data_ind), e)
                    raise
                bgmap = (
                    F.interpolate(bg_mask.view(1, 1, *bg_mask.shape[-2:]), size=[*fsize], mode="bilinear")[0].bool()
                ).flatten()
            else:
                bgmap = ~cmap

            bg_fs = feats.view(np.prod(fsize), -1)[bgmap, :]
            cat_bg_feats.append(bg_fs)

        if len(cat_feats) == 0:
            print(f"No features found for {c}")
            raise ValueError(f"No features found for {c}")

        proto_map[c] = torch.zeros(0, feat_dim)
        proto_map[c + "_bg"] = torch.zeros(0, feat_dim)

        proto_map[c + "_ind"] = torch.zeros(0, feat_dim)
        proto_map[c + "_bg_ind"] = torch.zeros(0, feat_dim)

        proto_map[c + "_kmeans"] = torch.zeros(0, feat_dim)
        proto_map[c + "_kmeans_bg"] = torch.zeros(0, feat_dim)

        if len(cat_feats) > 0:
            proto_map[c + "_ind"] = torch.stack([f.mean(0) for f in cat_feats])
            cat_feats = torch.cat(cat_feats, 0)
            proto_map[c] = cat_feats.mean(0)

            if args.kmeans > 0:
                proto_map[c + "_kmeans"] = kmeans_fn(cat_feats, args.kmeans, args.device)

        if len(cat_bg_feats) > 0:
            proto_map[c + "_bg_ind"] = torch.stack([f.mean(0) for f in cat_bg_feats])
            cat_bg_feats = torch.cat(cat_bg_feats, 0)
            proto_map[c + "_bg"] = cat_bg_feats.mean(0)

            if args.kmeans > 0:
                proto_map[c + "_kmeans_bg"] = kmeans_fn(cat_bg_feats, args.kmeans, args.device)

        ind_proto["proto"] = proto_map[c]
        ind_proto["proto_bg"] = proto_map[c + "_bg"]

        ind_proto["ind_proto"] = proto_map[c + "_ind"]
        ind_proto["ind_proto_bg"] = proto_map[c + "_bg_ind"]

        ind_proto["kmeans"] = proto_map[c + "_kmeans"]
        ind_proto["kmeans_bg"] = proto_map[c + "_kmeans_bg"]

        ind_proto["cls"] = c

        torch.save(ind_proto, ind_proto_path)

    if array_id is None:
        torch.save(proto_map, args.img_path / f"{args.data}_{args.out_name}_proto.pt")


if __name__ == "__main__":
    main()
# %%
