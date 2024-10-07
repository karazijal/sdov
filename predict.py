# %%
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from mmseg.core.evaluation.metrics import intersect_and_union, total_area_to_metrics
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

import sd
import mmdata
import utils.clip_class as clip_class
from protos import build_prototypes
from utils.extractor import StridedCLIPExtractor, StridedViTExtractor
from utils.pamr import PAMR


def np_one_hot(m):
    s = np.unique(m)
    return s.reshape(1, 1, -1) == m[..., None]


class EvalDataset:
    def __init__(self, dataset, size=None):
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_img, original_mask = self.dataset[idx]
        if self.size is not None:
            original_img = TF.resize(original_img, size=self.size, interpolation=TF.InterpolationMode.LANCZOS)
        clip_img_t, _ = mmdata.prep_sample((original_img, original_mask))

        original_img_chw = torch.from_numpy(np.array(original_img)).byte().permute(2, 0, 1)
        original_mask_hwc = torch.from_numpy(np.array(original_mask)).byte()

        return idx, original_img_chw, original_mask_hwc, clip_img_t


class RunningMetric:
    def __init__(
        self,
        num_classes,
        ignore_index=None,
        metrics=["mIoU"],
        confusion=False,
        individual=False,
        reduce_zero_label=False,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metrics = metrics
        self.reduce_zero_label = reduce_zero_label

        self.total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
        self.total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
        self.total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
        self.total_area_label = torch.zeros((num_classes,), dtype=torch.float64)

        self.confusion = None
        if confusion:
            self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)
        self.individual = None
        if individual:
            self.individual = []

    def get_conf(self, pred, gt, num_classes, ignore_index=None):
        n = num_classes
        gt_segm, res_segm = gt.flatten(), pred.flatten()
        if ignore_index is not None:
            to_ignore = gt_segm == ignore_index
            gt_segm, res_segm = gt_segm[~to_ignore], res_segm[~to_ignore]
        return confusion_matrix(gt_segm, res_segm, labels=list(range(n)), normalize=None)

    def update(self, result, gt_seg_map):
        label_map = dict()
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            result, gt_seg_map, self.num_classes, self.ignore_index, label_map, self.reduce_zero_label
        )
        self.total_area_intersect += area_intersect
        self.total_area_union += area_union
        self.total_area_pred_label += area_pred_label
        self.total_area_label += area_label

        if self.individual is not None:
            self.individual.append(
                np.nanmean(
                    total_area_to_metrics(
                        area_intersect, area_union, area_pred_label, area_label, self.metrics, nan_to_num=None, beta=1
                    )["IoU"]
                )
            )

        if self.confusion is not None:
            self.confusion += self.get_conf(result, gt_seg_map, self.num_classes, self.ignore_index)

    def result(self):
        nan_to_num = None
        beta = 1
        return total_area_to_metrics(
            self.total_area_intersect,
            self.total_area_union,
            self.total_area_pred_label,
            self.total_area_label,
            self.metrics,
            nan_to_num=nan_to_num,
            beta=beta,
        )


class SDFeats(nn.Module):
    def __init__(
        self, feat_size, layers, timestep, device="cpu", verbose=False, model=sd.MODEL_PATH
    ):
        super().__init__()
        self.layers = layers
        self.feat_size = feat_size
        pipe = sd.build_pipeline(device, with_inpaint=False, model_key=model).to(device)
        self.pipe = pipe
        self.timestep = timestep

    def forward(self, img_t):
        feats = self.pipe.features(img_t, self.timestep, self.layers, self.feat_size)
        feats = feats.permute(0, 2, 3, 1).view(*self.feat_size, -1).detach().float()
        return feats

    def resize(self, img_t):
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
            return TF.resize(img_t, size=(512, 512))[0]
        return TF.resize(img_t, size=(512, 512))


class ViTFeats(nn.Module):
    def __init__(self, layer, facet="key", device="cpu", verbose=False, model_key="dino_vitb8", stride=None):
        super().__init__()
        self.model_key = model_key
        if model_key.startswith("clip_"):
            self.model = StridedCLIPExtractor(model_key, stride=stride, device=device)
            self.input_size = (448, 448)
        else:
            self.model = StridedViTExtractor(model_key, stride=stride, device=device)
            self.input_size = (448, 448) if "mae" in model_key else 448
        # self.normalise = imgt = TF.normalize(imgt, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.layer = layer
        self.facet = facet
        self.device = device

    def forward(self, img_t):
        imgt = img_t.to(self.device)
        if len(imgt.shape) == 3:
            imgt = imgt.unsqueeze(0)
        imgt = TF.normalize(imgt, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        b, c, h, w = imgt.shape

        stride = self.model.stride[0]
        fW = w // stride - self.model.p // stride + 1
        fH = h // stride - self.model.p // stride + 1
        feats = self.model.extract_descriptors(imgt, layer=self.layer, facet=self.facet)  # Bx1xtx(dxh)

        # if cpu:
        #     feats = feats.cpu()
        feats = feats.squeeze(1).view(b, fH, fW, -1).detach().float()  # Bxtxd

        return feats[0]

    def resize(self, img_t):
        h, w = img_t.shape[-2:]
        if h <= 448 and w <= 448 and "dino" in self.model_key:
            return img_t

        reduce_dim = False
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
            reduce_dim = True
        img_t = TF.resize(img_t, size=self.input_size, max_size=800 if isinstance(self.input_size, int) else None)
        if reduce_dim:
            img_t = img_t[0]
        return img_t


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--size", type=int, default=448)

    # Classifier settings for global filtering
    parser.add_argument("--classmode", type=str, default="clipthresh", choices=["all", "clipthresh"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--no-stuff-include", dest="stuff_include", action="store_false", default=True)
    parser.add_argument("--class_thresh", type=float, default=0.0)

    # Prototype settings and feature processing
    parser.add_argument("--prots", default="custom_outs/pascal_catergories/prototypes.pt", nargs="+", type=str)
    parser.add_argument(
        "--feat_mode",
        type=str,
        default="all",
        choices=["mean", "ind", "kmeans", "all", "mean_ind", "mean_kmeans", "kmeans_ind"],
        help="Which prototype vectors to use",
    )
    parser.add_argument("--no_norm", action="store_true", default=False)

    # Settings for handling backgrounds
    parser.add_argument("--bgmode", type=str, default="all", choices=["zero", "mean", "all"])

    parser.add_argument(
        "--no_class_bg",
        action="store_true",
        default=False,
        help="Do not exclude class-specific background prototypes using a classifier",
    )

    parser.add_argument("--filter_fg_nsigma", type=int, default=0)
    parser.add_argument("--filter_fg_fallback", type=str, default="none", choices=["none", "mean", "top5", "zero"])

    parser.add_argument("--filter_bgfg_nsigma", type=int, default=0)
    parser.add_argument("--filter_bgfg_fallback", type=str, default="none", choices=["none", "mean", "top5", "zero"])

    parser.add_argument("--filter_bgfg_topk", type=int, default=-1)
    parser.add_argument("--filter_bgfg", action="store_true", default=False)

    parser.add_argument("--filter_bg_thres", default="0.85", type=str)

    parser.add_argument("--filter_bg_fallback", default="none", type=str, choices=["none", "mean", "zero", "empty"])

    parser.add_argument("--no-ctx-mode", dest="ctx_mode", action="store_false", default=True)
    parser.add_argument("--no-skip-stuff-bg", action="store_true", default=False)

    
    parser.add_argument("--thresholds", type=str, default="0.0")
    parser.add_argument("--pamr", action="store_true", default=False)

    parser.add_argument("--crops", type=str, default="448,336")
    parser.add_argument("--strides", type=str, default="224,224")

    parser.add_argument("--no-overlay", dest="overlay", action="store_false", default=True)

    # Output settings
    parser.add_argument("--no_overwrite", action="store_true", default=False)
    parser.add_argument("--save_sim", action="store_true", default=False)
    parser.add_argument("--nonlocal", dest='local', action="store_false", default=True)  # Do not save
    parser.add_argument("--brk", default=None)
    parser.add_argument("--progbar", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=-1)

    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("data", type=str, default="voc", choices=["voc"])
    parser.add_argument("out_path", type=Path)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available() and "cuda" in args.device:
            torch.cuda.manual_seed(args.seed)

    dataset, classlist, cls2prompt = mmdata.get_dataset(args.data, apply_overlay=args.overlay)
    if args.data == "voctrain":
        args.data = "voc"
    assert "background" in classlist[0], "Background must be first class; even if not used"
    metric_dataset = dataset.dataset
    classlist_nobg = [c for c in classlist if "background" not in c]
    is_slurm = os.getenv("SLURM_JOB_ID") is not None
    device = torch.device(args.device)

    out_path = Path(args.out_path)
    out_path.mkdir(exist_ok=True, parents=True)

    thresholds = [float(thres.strip()) for thres in args.thresholds.split(",")]

    filter_bg_thres = [
        float(thres.strip()) if thres.strip() != "" else 0.0 for thres in args.filter_bg_thres.split(",")
    ]
    if len(filter_bg_thres) == 1 and len(args.prots) > 1:
        filter_bg_thres = filter_bg_thres * len(args.prots)

    cropsizes = []
    strides = []
    if len(args.crops) > 1:
        cropsizes = [int(cs.strip()) for cs in args.crops.split(",")]
        strides = [int(cs.strip()) for cs in args.strides.split(",")]

    clip_clas_list = [cls2prompt[cname] for cname in classlist[1:]]
    clip_clas_map = {cid: cid + 1 for cid, cname in enumerate(classlist[1:])}
    if args.stuff_include:
        clip_clas_list = [
            (cid, cname)
            for cid, cname in enumerate(classlist)
            if "background" not in cname and (cname not in mmdata.IS_STUFF or not mmdata.IS_STUFF[cname])
        ]
        clip_clas_map = {effid: cid for effid, (cid, cname) in enumerate(clip_clas_list)}
        clip_clas_list = [cls2prompt[cname] for cid, cname in clip_clas_list]

    clip_cls = None
    if args.class_thresh > 0.0:
        clip_cls = clip_class.ScoreThreshold(clip_cls, threshold=args.class_thresh)

    elif args.classmode == "clipthresh":
        clip_cls = clip_class.CLIPClassifierThresh(device=device, class_list=clip_clas_list, topk=args.topk)

    feature_extractors = []
    for index, proto_path in enumerate(args.prots):
        proto_path = proto_path.replace("{dataset}", args.data)
        prototypes = torch.load(proto_path, map_location="cpu")

        layers = prototypes.get("__layers__", None)
        timestep = prototypes.get("__timestep__", 200)
        feat_size = prototypes.get("__feat_size__", (64, 64))

        print(f"Prototypes from {proto_path} t={timestep} {feat_size}")
        if layers is not None:
            for l in layers:
                print(l)

        proto_name = Path(proto_path).name
        if "sd" in proto_name or "n64t200" in proto_name:
            if layers:
                ls = []
                for l in layers:
                    if not l.endswith(".processor"):
                        l += ".processor"
                    ls.append(l)
                layers = ls

                for l in layers:
                    print(l)
            else:
                raise ValueError("No layers specified")

            feature_extractor = SDFeats(feat_size, layers, timestep, device=device, verbose=False)
        elif "clip" in proto_name:
            feature_extractor = ViTFeats(
                -2,
                "key" if "token" not in proto_name else "token",
                device=device,
                verbose=False,
                model_key="clip_ViT-B/16",
                stride=None,
            )
        elif "dino" in proto_name:
            _dino_name = "dino_vitb8" if "dino_vitb8" in proto_name else "dino_vitb16"
            if "dino_vits8" in proto_name:
                _dino_name = "dino_vits8"
            feature_extractor = ViTFeats(
                -1, "key", device=device, verbose=False, model_key=_dino_name, stride=None
            )
        elif "mae" in proto_name:
            feature_extractor = ViTFeats(
                -1, "key", device=device, verbose=False, model_key="mae_longseq_vit_base_imgnet"
            )
        else:
            raise ValueError(f"Unknown feature extractor for {proto_name}")

        print(type(feature_extractor))

        all_prototypes, proto_classes, proto_class_mask, proto_class_mask_aggr = build_prototypes(
            prototypes,
            classlist,
            args.feat_mode,
            0,
            args.device,
            args.bgmode,
            filter_fg_nsigma=args.filter_fg_nsigma,
            filter_fg_fallback=args.filter_fg_fallback,
            filter_bgfg_nsigma=args.filter_bgfg_nsigma,
            filter_bgfg_fallback=args.filter_bgfg_fallback,
            filter_bg_thres=filter_bg_thres[index],
            filter_bg_fallback=args.filter_bg_fallback,
            filter_bgfg_topk=args.filter_bgfg_topk,
            filter_bgfg=args.filter_bgfg,
            ctx_mode=args.ctx_mode,
            no_skip_stuff_bg=args.no_skip_stuff_bg,
        )

        proto_classes = proto_classes.to(device)
        all_prototypes = all_prototypes.to(device)
        proto_class_mask = proto_class_mask.to(device)

        fg_protos = []
        fg_masks = []
        bg_protos = []
        bg_masks = []

        n = 0
        for cid in range(1, len(classlist)):
            _n = (proto_class_mask == cid).long().sum().item()
            if _n > n:
                n = _n

        for cid in range(1, len(classlist)):
            fg = all_prototypes[proto_class_mask == cid]
            fg_pad = torch.zeros(n, *fg.shape[1:], device=fg.device, dtype=fg.dtype)
            fg_pad[: fg.shape[0]] = fg
            fg_mask = torch.ones(n, device=fg.device, dtype=torch.bool)
            fg_mask[fg.shape[0] :] = False

            bg = all_prototypes[torch.logical_and(proto_class_mask == 0, proto_classes == cid)]
            bg_pad = torch.zeros(n, *bg.shape[1:], device=bg.device, dtype=bg.dtype)
            bg_pad[: bg.shape[0]] = bg
            bg_mask = torch.ones(n, device=bg.device, dtype=torch.bool)
            bg_mask[bg.shape[0] :] = False

            # print(cid, classlist[cid], fg.shape, bg.shape)

            fg_protos.append(fg_pad)
            fg_masks.append(fg_mask)
            bg_protos.append(bg_pad)
            bg_masks.append(bg_mask)

        fg_protos = torch.stack(fg_protos, dim=0)
        bg_protos = torch.stack(bg_protos, dim=0)
        fg_masks = torch.stack(fg_masks, dim=0)
        bg_masks = torch.stack(bg_masks, dim=0)

        all_prototypes = torch.stack([bg_protos, fg_protos], dim=0)
        all_proto_mask = torch.stack([bg_masks, fg_masks], dim=0)

        if args.bgmode == "zero":
            all_prototypes = all_prototypes[1:]
            all_proto_mask = all_proto_mask[1:]

        all_prototypes_norm = all_prototypes / all_prototypes.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        all_prototypes_norm = all_prototypes_norm * all_proto_mask[..., None]
        all_prototypes_norm = all_prototypes_norm.to(device)
        all_proto_mask = all_proto_mask.to(device)

        print(proto_path, all_prototypes.shape, all_proto_mask.shape)

        feature_extractor.all_prototypes_norm = all_prototypes_norm
        feature_extractor.all_proto_mask = all_proto_mask

        feature_extractors.append(feature_extractor)

    IGNORE_INDEX = 255
    eval_classlist = classlist
    reduce_zero_label = False
    metrics = RunningMetric(
        len(eval_classlist),
        ignore_index=IGNORE_INDEX,
        metrics=["mIoU"],
        reduce_zero_label=reduce_zero_label,
    )

    pamr = None
    if args.pamr:
        pamr_iter = 10
        pamr_kernel = [1, 2, 4, 8, 12, 24]
        pamr = PAMR(pamr_iter, pamr_kernel)
        pamr = pamr.eval()
        pamr = pamr.to(args.device)

    iter_dataset = EvalDataset(dataset, size=args.size)
    iter_target = list(range(len(dataset)))
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

    if args.local and array_id is not None:
        raise ValueError("Cannot run local mode and sharded eval at the same time")

    iter_dataset = torch.utils.data.Subset(iter_dataset, iter_target)
    num_workers = args.num_workers
    if args.num_workers < 0:
        num_workers = max(min(os.cpu_count(), 12), 4)
        print(f"Using {num_workers} workers")
    iter_loader = torch.utils.data.DataLoader(
        iter_dataset,
        batch_size=1,
        num_workers=max(min(os.cpu_count(), 12), 4),
        pin_memory=args.device != "cpu",
        shuffle=False,
    )

    results = []

    for data_ind, original_img_chw, original_mask_hwc, clip_img_t in tqdm(
        iter_loader, disable=is_slurm and not args.progbar
    ):
        data_ind = data_ind.item()

        res_path = out_path / f"{data_ind:0>5d}.pt"
        if res_path.exists() and args.no_overwrite:
            print(f"Skipping {data_ind} as {res_path} exists")
            continue

        if "clip" in args.classmode:
            clip_img_t = clip_img_t.to(args.device, non_blocking=True)

        original_img_chw = original_img_chw[0].to(args.device, non_blocking=True)
        original_mask_hwc = original_mask_hwc[0]

        if args.classmode == "all":
            class_ids = list(range(len(classlist)))[1:]

        elif "clip" in args.classmode:
            class_ids = clip_cls(clip_img_t)
            clip_cls.update(class_ids)
            class_ids = [clip_clas_map[cid] for cid in class_ids]

        if args.stuff_include:
            extra_class_ids = [
                cid
                for cid, c in enumerate(classlist)
                if c in mmdata.IS_STUFF and mmdata.IS_STUFF[c] and "background" not in c
            ]
            class_ids = list(set(class_ids + extra_class_ids))

        class_ids_wbg = [0] + class_ids
        class_ids_wbg = sorted(class_ids_wbg)

        class_ids_mask = (
            (torch.tensor(class_ids_wbg).unsqueeze(-1) == torch.arange(len(classlist))[None])
            .any(0)[1:]
            .unsqueeze(0)
            .to(device)
        )
        if args.no_class_bg:
            class_ids_mask = class_ids_mask.expand(2, -1)
            class_ids_mask[0] = True

        if args.classmode == "all":
            assert class_ids_mask.all(), "All classes must be used in all mode"

        if args.save_sim or not args.local:
            sample_preds = None
        similarities = None

        if array_id is None:
            gt_mask = original_mask_hwc.cpu().numpy()

        agg_similarity_global = torch.zeros(len(classlist), *original_img_chw.shape[-2:], device=device)

        counts = torch.zeros(1, *original_img_chw.shape[-2:], device=device)
        crops = []
        imgt = original_img_chw.float() / 255
        oH, oW = imgt.shape[-2:]

        for crop, stride in zip(cropsizes, strides):
            h_stride, w_stride = stride, stride
            h_crop, w_crop = crop, crop
            h_grids = max(oH - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(oW - w_crop + w_stride - 1, 0) // w_stride + 1
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, oH)
                    x2 = min(x1 + w_crop, oW)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_imgt = imgt[:, y1:y2, x1:x2]
                    crops.append((crop_imgt, x1, y1, x2, y2))

        # Add full img as well
        y1 = 0
        x1 = 0
        y2 = oH
        x2 = oW
        crop_imgt = imgt
        crops.append((crop_imgt, x1, y1, x2, y2))

        for feature_extractor in feature_extractors:
            all_proto_mask = feature_extractor.all_proto_mask  # [2, num_classes, num_proto]
            simlr_mask = torch.logical_and(all_proto_mask, class_ids_mask[..., None])[..., None, None]  # l c k 1 1
            proto = feature_extractor.all_prototypes_norm

            for crop_imgt, x1, y1, x2, y2 in crops:
                crop_imgt = feature_extractor.resize(crop_imgt[None])[0]
                fs_ = feature_extractor(crop_imgt)

                if not args.no_norm:
                    fs_ = fs_ / fs_.norm(dim=-1, keepdim=True).clamp_min(1e-6)

                agg_similarity = torch.einsum("hwf,lckf->lckhw", fs_, proto)
                if args.save_sim:
                    similarities = agg_similarity.cpu()

                agg_similarity = torch.where(
                    simlr_mask.broadcast_to(agg_similarity.shape),
                    agg_similarity,
                    torch.scalar_tensor(-1, device=agg_similarity.device, dtype=agg_similarity.dtype),
                )
                agg_similarity = agg_similarity.max(dim=2).values  # [l, c, h, w]

                if args.bgmode == "zero":
                    agg_similarity = agg_similarity[0]
                    agg_similarity = torch.cat([torch.zeros_like(agg_similarity[:1]), agg_similarity], dim=0)
                else:
                    agg_similarity = torch.cat(
                        [agg_similarity[0].max(dim=0, keepdim=True).values, agg_similarity[1]], dim=0
                    )

                agg_similarity = F.interpolate(
                    agg_similarity[None], size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False
                )[0].clamp(min=-1, max=1)
                agg_similarity_global[:, y1:y2, x1:x2] += agg_similarity
                counts[:, y1:y2, x1:x2] += 1
                del agg_similarity, fs_, crop_imgt

        agg_similarity = agg_similarity_global / counts.clamp_min(1)

        # If PAMR is used, use it to refine the mask
        # pamr_scores = None
        if args.pamr:
            with torch.no_grad():
                agg_similarity = pamr(original_img_chw[None].float(), agg_similarity[None].float())[0]

        agg_similarity = F.interpolate(
            agg_similarity[None], size=gt_mask.shape[-2:], mode="bilinear", align_corners=False
        )[0]

        for thres in thresholds:
            thresholded_mask = torch.where(
                agg_similarity > thres, agg_similarity, torch.zeros_like(agg_similarity)
            ).float()
            if reduce_zero_label:
                thresholded_mask = thresholded_mask[1:]  # Remove background

            _max = thresholded_mask.max(dim=0)
            hard_mask = _max.indices
            region_mask = hard_mask
            region_scores = thresholded_mask

            if array_id is None:
                metrics.update(region_mask.cpu().numpy(), gt_mask)
                if len(thresholds) == 1:
                    results.append(region_mask.cpu().numpy())

            if args.save_sim or not args.local:
                sample_preds = region_mask.cpu()

        if args.save_sim or not args.local:
            similarities_to_save = {pcn: t for pcn, t in similarities.items() if t is not None}
            torch.save(
                {
                    # 'gt': torch.from_numpy(gt_mask),
                    "preds": sample_preds,
                    "classes": class_ids,
                    "similarities": similarities_to_save if args.save_sim else None,
                },
                res_path,
            )

        if args.brk is not None and data_ind >= int(args.brk):
            break
        if is_slurm and not args.progbar:
            print(f"Progress: {data_ind}/{len(dataset)}", end="\n")

        del imgt, crops, agg_similarity_global, counts, original_img_chw, original_mask_hwc, clip_img_t
        del region_mask, region_scores, hard_mask, thresholded_mask

    if array_id is None:
        print("Evaluating")
        if len(thresholds) == 1:
            metric_dataset.evaluate(results)
        else:
            print("multiple thresholds, not saving results for second metric check")
        print("OLD")

        ret = metrics.result()
        ious = ret["IoU"]
        mIoU = np.nan_to_num(ious).mean()
        print(f"mIoU {mIoU*100:.3f}")

        for i, iou in enumerate(ret["IoU"]):
            print(f"{eval_classlist[i]}: {iou}")
        # better format for copy/pasting
        for i, iou in enumerate(ret["IoU"]):
            print(f"{iou}")


if __name__ == "__main__":
    main()

# %%
