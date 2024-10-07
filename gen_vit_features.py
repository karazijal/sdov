# %%
import argparse
import gc
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm

import mmdata
from utils.extractor import PatchResize, StridedCLIPExtractor, StridedViTExtractor


def extract(model, imgt, device, cpu=False, layer=-1, facet="key"):
    imgt = imgt.to(device)
    b, c, h, w = imgt.shape

    stride = model.stride[0]
    fW = w // stride - model.p // stride + 1
    fH = h // stride - model.p // stride + 1
    # print(h, w, fW, fH)
    feats = model.extract_descriptors(imgt, layer=layer, facet=facet)  # Bx1xtx(dxh)

    if cpu:
        feats = feats.cpu()
    feats = feats.squeeze(1).transpose(1, 2).reshape(b, -1, fH, fW)  # Bxtxd
    return feats


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="dino_vitb8")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--square", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--progbar", action="store_true", default=False)
    parser.add_argument("--lowmem", action="store_true", default=False)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--facet", type=str, default="key", choices=["key", "query", "value", "token"])
    parser.add_argument("data", type=str, default="voc", choices=["voc"])
    parser.add_argument("img_path", type=Path)

    args = parser.parse_args()
    classlist, _ = mmdata.get_data_classes(args.data)
    is_slurm = os.getenv("SLURM_JOB_ID") is not None
    device = torch.device(args.device)

    classlist_nobg = [c for c in classlist if "background" not in c]
    iter_target = classlist_nobg
    total = len(iter_target)

    if is_slurm:
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if array_id is not None:
            curr_id = int(array_id)
            min_id = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
            num_ids = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
            iter_target = iter_target[curr_id - min_id : total : num_ids]
            print(f"Running {curr_id} of {num_ids} jobs, {len(iter_target)} classes")

    model_key = args.model_key
    stride = args.stride
    dirname = model_key.split("_")[0]

    assert (
        model_key.startswith("dino_vit")
        or model_key.startswith("dinov2_vit")
        or model_key.startswith("mae_")
        or model_key.startswith("clip_")
    )
    if model_key.startswith("clip_"):
        model = StridedCLIPExtractor(model_key, stride=stride, device=device)
    else:
        model = StridedViTExtractor(model_key, stride=stride, device=device)
    stride = model.stride[0]
    ps = stride

    resize = PatchResize(ps, interpolation=TF.InterpolationMode.LANCZOS)

    for c in tqdm(iter_target, desc=f"Generating {dirname.upper()} features", disable=is_slurm and not args.progbar):

        cls_dir = args.img_path / c
        imgps = [p for p in cls_dir.glob("*.png") if p.name.startswith("0")]
        imgps = sorted(imgps, key=lambda x: int(x.stem))

        if len(imgps) == 0:
            assert False, f"No images found in {cls_dir}"

        if args.num_samples > 0:
            imgps = imgps[: args.num_samples]

        out_dir = cls_dir / dirname
        out_dir.mkdir(exist_ok=True)

        for imgp in tqdm(
            imgps, desc=f"Generating {dirname.upper()}  features for {c}", disable=is_slurm and not args.progbar
        ):
            data_ind = imgp.stem.lstrip("0")
            data_ind = int(data_ind) if data_ind else 0

            idents = [str(model_key).lower().replace("/", "_"), str(stride)]

            if args.size is not None:
                if args.square:
                    idents.append(f"{args.size}x{args.size}")
                else:
                    idents.append(str(args.size))

            if args.facet != "key":
                idents.append(args.facet)
            if args.layer != -1:
                idents.append(str(args.layer))

            p = Path(out_dir) / f'{"_".join(idents)}_{data_ind:0>5d}.pt'

            if p.exists():
                print("skip; exists", p)
                continue

            original_img = Image.open(imgp).convert("RGB")
            if args.size is not None:
                size = args.size
                if args.square:
                    size = (size, size)
                original_img = TF.resize(original_img, args.size)

            oW, oH = TF.get_image_size(original_img)
            if not torch.is_tensor(original_img):
                _imgt = TF.to_tensor(original_img)
            else:
                _imgt = original_img.view(3, oH, oW)

            imgt = resize(original_img).to(device)
            imgt = TF.normalize(imgt, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            b, c, h, w = imgt.shape
            with torch.no_grad():
                feats = extract(model, imgt, device, cpu=args.lowmem, layer=args.layer, facet=args.facet)

            feats = feats.detach().float().cpu()

            torch.save(
                {
                    "feats": feats,
                    "original_h": oH,
                    "original_w": oW,
                },
                p,
            )
            if args.lowmem:
                del feats
                # torch.cuda.empty_cache()
                gc.collect()


if __name__ == "__main__":
    main()
# %%
