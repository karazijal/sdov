# %%
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False

import argparse
import gc
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.nn.functional as F
import torchvision
from skimage.filters import threshold_multiotsu
from skimage.morphology import binary_erosion
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

import mmdata
import sd
from utils.cutler import CUTLER_Processor

NEGATIVE_PROMPT = "text, low quality, blurry, cartoon, meme, low resolution, bad, poor, faded"


def write_imgs(masks, out_dir, prefix="mask_"):
    out_dir = Path(out_dir)
    for i, mask in enumerate(masks):
        torchvision.transforms.functional.to_pil_image(mask).save(out_dir / f"{prefix}{i:0>5d}.png")


def write_mask(mask, path):
    mask = F.interpolate(mask[None], size=(512, 512), mode="nearest")[0]
    mask = to_pil_image(mask)
    mask.save(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--bs", type=int, default=8)  # This affects ro
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample", action="store_true", default=False)
    parser.add_argument("--num_inference_steps", type=int, default=30)

    parser.add_argument(
        "--cpu", action="store_true", default=False, help="Offload partial results to CPU to save GPU memory"
    )

    parser.add_argument("--no-overlay", dest="overlay", action="store_false", default=True)

    parser.add_argument("--progbar", action="store_true", default=False)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--prompt", type=str, default=None)
    group.add_argument("--categories", type=str, nargs="*")

    parser.add_argument("data", type=str, default="voc", choices=["voc"])
    parser.add_argument("out_path", type=Path)

    args = parser.parse_args()

    classlist, cls2prompt = mmdata.get_data_classes(args.data, apply_overlay=args.overlay)

    is_slurm = os.getenv("SLURM_JOB_ID") is not None

    device = torch.device(args.device)

    out_path = Path(args.out_path)
    out_path.mkdir(exist_ok=True, parents=True)

    batch_size = args.bs

    classlist_nobg = [c for c in classlist if "background" not in c]
    iter_target = classlist_nobg
    use_template = True
    use_remapping = True

    if args.prompt:
        iter_target = [args.prompt]
        use_template = False
        use_remapping = False
    elif args.categories:
        iter_target = list(args.categories)
        use_template = True
        use_remapping = False

    total = len(iter_target)

    if is_slurm:
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if array_id is not None:
            curr_id = int(array_id)
            min_id = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
            num_ids = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
            iter_target = iter_target[curr_id - min_id : total : num_ids]
            print(f"Running {curr_id} of {num_ids} jobs, {len(iter_target)} classes")

    cutler_old = CUTLER_Processor()
    cutler_old = cutler_old.to(device)

    prmpt_pipe = sd.build_pipeline(device, args.model, disable_progbar=is_slurm and not args.progbar)

    for c in tqdm(iter_target, desc="Sampling images", disable=is_slurm):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        gen = torch.Generator("cpu")
        gen.manual_seed(args.seed)

        cls_dir_name = c
        if args.prompt:
            cls_dir_name = c.replace(" ", "_").replace("/", "+")
        cls_dir = out_path / cls_dir_name
        cls_dir.mkdir(exist_ok=True)
        daam_dir = cls_dir / "daam"
        daam_dir.mkdir(exist_ok=True)
        old_idr = cls_dir / "proc_masks_cutler_v3"
        old_idr.mkdir(exist_ok=True)

        use_cpu = args.cpu

        N = args.n

        if not args.resample:
            exising = [i for i in cls_dir.glob("*.png") if i.name.startswith("0")]
            existing_bg = [i for i in old_idr.glob("bg_pp_*.png") if i.name.startswith("bg_pp_0")]
            if len(exising) >= N and len(existing_bg) == len(exising):
                print(f"Skipping {c} as {len(exising)} images already exist")
                continue

        imgs = []
        daams = []

        gc.collect()
        torch.cuda.empty_cache()

        target = c
        if use_remapping:
            target = cls2prompt[c]
        prompt = target
        if use_template:
            prompt = f"A good picture of a {target}"

        start_time = datetime.now()
        print("sampling", c)
        with tqdm(total=N, desc=f"Generating <{prompt}>", disable=is_slurm and not args.progbar) as pbar:
            with torch.no_grad():
                for i in range(int(args.start), N, batch_size):
                    ebs = min(batch_size, N - i)

                    gs = 8.0
                    images, hms = prmpt_pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        token_prompts=[target],
                        guidance_scale=gs,
                        generator=gen,
                        num_inference_steps=args.num_inference_steps,
                        max_iter_to_alter=0,
                        num_images_per_prompt=ebs,
                    )
                    if use_cpu:
                        images = images.cpu()
                        for img in images:
                            imgs.append(img)
                        hms = hms.cpu()
                        for hm in hms:
                            daams.append(hm)
                    else:
                        imgs.append(images)
                        daams.append(hms)
                    pbar.update(ebs)

        print("generating masks", c)

        if not use_cpu:
            imgs = torch.cat(imgs, 0)
            daams = torch.cat(daams, 0)
        else:
            daams = torch.stack(daams, 0)

        daams -= daams.flatten(2).min(-1, keepdim=True)[0][..., None]
        daams /= daams.flatten(2).max(-1, keepdim=True)[0][..., None].clamp(min=1e-6)

        daams_small = daams

        daams = F.interpolate(daams, size=imgs[0].shape[-2:], mode="bilinear", align_corners=False)
        daams -= daams.flatten(2).min(-1, keepdim=True)[0][..., None]
        daams /= daams.flatten(2).max(-1, keepdim=True)[0][..., None].clamp(min=1e-6)

        daams_np = daams_small.cpu().numpy()
        try:
            bg_thresh, fg_thresh = threshold_multiotsu(daams_np, classes=3)
        except ValueError as exc:
            print(f"Error calculating Otsu thresholds: {exc}")
            bg_thresh, fg_thresh = 0.2, 0.5  # Using conservative defaults that work OKish
            print("Using conservative defaults of bg <{} and fg >{}".format(bg_thresh, fg_thresh))

        fg_thresh_masks = (daams[:, :1] > fg_thresh).float()
        bg_thresh_masks = (daams[:, :1] < bg_thresh).float()

        write_imgs(imgs, cls_dir, prefix="")
        write_imgs(daams, daam_dir, prefix="heatmap_")

        thresh_dir = cls_dir / "proc_masks_thresh"
        thresh_dir.mkdir(exist_ok=True)
        write_imgs(fg_thresh_masks, thresh_dir, prefix="fg_")
        write_imgs(bg_thresh_masks, thresh_dir, prefix="bg_")

        with torch.no_grad():
            fg_masks = []
            bg_masks = []
            for i in tqdm(range(0, len(imgs)), desc="Extracting Masks [OLD]", disable=is_slurm and not args.progbar):

                r = cutler_old(imgs[i] * 255)
                daam = daams[i, 0].view(1, *daams.shape[-2:])

                # print(r.shape, daam.shape)
                if r.shape[0] < 2:
                    fg = (daam > 0.5).float()
                    fg = fg.view(1, *fg.shape[-2:]).float()
                    bg = (daam < 0.2).float()
                    bg = bg.view(1, *bg.shape[-2:]).float()

                    # print(fg.shape, bg.shape)
                    write_mask(fg, old_idr / f"fg_{i:0>5d}.png")
                    write_mask(bg, old_idr / f"bg_{i:0>5d}.png")
                    write_mask(bg, old_idr / f"bg_thresh_{i:0>5d}.png")
                    write_mask(fg, old_idr / f"fg_pp_{i:0>5d}.png")
                    write_mask(bg, old_idr / f"bg_pp_{i:0>5d}.png")
                    continue

                weights = (r * daam / r.sum([-1, -2], keepdim=True)).sum([-1, -2])
                reind = torch.argsort(weights, descending=True)
                weights = weights[reind]
                r = r[reind]
                r_clipped = (r * weights.unsqueeze(-1).unsqueeze(-1)).argmax(0)

                fg = r_clipped == 0
                fg = fg.view(1, *fg.shape[-2:]).float()
                bg = r_clipped == r_clipped.max()
                bg = bg.view(1, *bg.shape[-2:]).float()

                write_mask(fg, old_idr / f"fg_{i:0>5d}.png")
                write_mask(bg, old_idr / f"bg_{i:0>5d}.png")
                bg = bg * (daam < 0.2)[0]

                write_mask(bg, old_idr / f"bg_thresh_{i:0>5d}.png")

                bg = (
                    torch.from_numpy(binary_erosion(bg.detach().round().bool().view(512, 512).cpu().numpy()))
                    .float()
                    .view(1, *daams.shape[-2:])
                )

                write_mask(bg, old_idr / f"bg_pp_{i:0>5d}.png")

        del daams, fg_thresh_masks, bg_thresh_masks, fg_masks, bg_masks


if __name__ == "__main__":
    main()
# %%
