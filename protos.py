# %%
import numpy as np
import torch
from sklearn.cluster import KMeans

import mmdata

def np_one_hot(m):
    s = np.unique(m)
    return s.reshape(1, 1, -1) == m[..., None]


def filter_proto_fg2fg(fg_proto, classname, nsigma, fallback_mode="noop"):
    if fg_proto.shape[0] <= 1:
        return fg_proto
    proto = fg_proto.clone()
    proto /= proto.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    self_sim = torch.einsum("ij,kj->ik", proto, proto)
    # ignore diagonal
    meam_sim = (
        (self_sim / (1 - torch.eye(self_sim.shape[0], device=self_sim.device)))
        .nan_to_num_(posinf=float("nan"))
        .nanmean(-1)
    )
    threshold = meam_sim.mean() - nsigma * meam_sim.std()
    keep_mask = meam_sim >= threshold
    if keep_mask.sum() >= 0:
        print(f"Filtered FG {fg_proto.shape[0] - keep_mask.sum()} for {classname}")
        return fg_proto[keep_mask]

    if fallback_mode == "mean":
        print(f"Fallback to mean for {classname}")
        return fg_proto.mean(0, keepdim=True)
    elif fallback_mode == "top5":
        print(f"Fallback to top5 for {classname}")
        return fg_proto[meam_sim.top(5).indices]

    print(f"Fallback to noop for {classname}")
    return fg_proto


def filter_proto_bg2fg(bg_proto, fg_proto, classname, nsigma, fallback_mode="noop"):
    if bg_proto.shape[0] <= 1:
        return bg_proto
    proto = bg_proto.clone()
    proto /= proto.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    self_sim = torch.einsum("ij,kj->ik", proto, fg_proto / fg_proto.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    # ignore diagonal
    meam_sim = self_sim.nan_to_num_(posinf=float("nan")).nanmean(-1)
    threshold = meam_sim.mean() + nsigma * meam_sim.std()
    keep_mask = meam_sim <= threshold
    if keep_mask.sum() >= 0:

        print(f"Filtered BG {bg_proto.shape[0] - keep_mask.sum()} for {classname}")
        return bg_proto[keep_mask]

    if fallback_mode == "mean":
        print(f"Fallback to mean for {classname}")
        return bg_proto.mean(0, keepdim=True)
    elif fallback_mode == "top5":
        print(f"Fallback to top5 for {classname}")
        return bg_proto[meam_sim.topk(5, largest=False).indices]
    elif fallback_mode == "zero":
        print(f"Fallback to top5 for {classname}")
        return torch.zero_like(bg_proto[:1])

    print(f"Fallback to noop for {classname}")
    return bg_proto


def filter_proto_bg2fg_best(bg_proto, fg_proto, classname, topk):
    if bg_proto.shape[0] <= 1:
        return bg_proto
    proto = bg_proto.clone()
    proto /= proto.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    self_sim = torch.einsum("ij,kj->ik", proto, fg_proto / fg_proto.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    # ignore diagonal
    meam_sim = self_sim.nanmean(-1)
    return bg_proto[meam_sim.topk(topk, largest=True).indices]


def filter_proto_bg2allfg(bg_proto, allfg_proto, classname, threshold, fallback_mode="noop"):
    if bg_proto.shape[0] <= 1:
        return bg_proto
    proto = bg_proto.clone()
    proto /= proto.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    self_sim = torch.einsum("ij,kj->ik", proto, allfg_proto / allfg_proto.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    keep_mask = (self_sim <= threshold).all(-1)
    if keep_mask.sum() >= 0:
        print(
            f"Filtered BG (thresh) {bg_proto.shape[0] - keep_mask.sum()} for {classname} (remaining: {keep_mask.sum()})"
        )
        return bg_proto[keep_mask]

    if fallback_mode == "mean":
        print(f"Fallback to mean for {classname}")
        return bg_proto.mean(0, keepdim=True)
    elif fallback_mode == "zero":
        print(f"Fallback to top5 for {classname}")
        return torch.zero_like(bg_proto[:1])
    elif fallback_mode == "empty":
        print(f"returning empty for {classname}")
        return torch.empty(0, *bg_proto.shape[1:], device=bg_proto.device)

    print(f"Fallback to noop for {classname}")
    return bg_proto


def filter_proto_bg2fg_inter(bg_proto, fg_proto, classname, fg_classname, fallback_mode="noop"):
    if bg_proto.shape[0] <= 1:
        return bg_proto
    proto = bg_proto.clone()
    proto /= proto.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    self_sim = torch.einsum("ij,kj->ik", proto, fg_proto / fg_proto.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    fg_proto = fg_proto.clone()
    fg_proto /= fg_proto.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    self_sim = torch.einsum("ij,kj->ik", fg_proto, fg_proto)
    # set diagonal to -1
    self_sim = self_sim * (
        1
        - torch.eye(self_sim.shape[0], device=self_sim.device)
        - torch.eye(self_sim.shape[0], device=self_sim.device).T
    )
    max_sim = self_sim.max(-1).values
    cros_sim = torch.einsum("ij,kj->ik", fg_proto, proto)

    keep_mask = (cros_sim < max_sim[..., None]).all(0)
    if keep_mask.sum() >= 0:
        if keep_mask.sum() < bg_proto.shape[0]:
            print(
                f"Filtered BG (inter) {bg_proto.shape[0] - keep_mask.sum()} for {classname} vs {fg_classname} (remaining: {keep_mask.sum()})"
            )
        return bg_proto[keep_mask]

    if fallback_mode == "mean":
        print(f"Fallback to mean for {classname}")
        return bg_proto.mean(0, keepdim=True)
    elif fallback_mode == "zero":
        print(f"Fallback to top5 for {classname}")
        return torch.zero_like(bg_proto[:1])
    elif fallback_mode == "empty":
        print(f"returning empty for {classname}")
        return torch.empty(0, *bg_proto.shape[1:], device=bg_proto.device)
    print(f"Fallback to noop for {classname}")
    return bg_proto


def build_prototypes(
    prototypes,
    classlist,
    feat_mode="ind",
    kmeans=0,
    device="cpu",
    bgmode="all",
    filter_fg_nsigma=0,
    filter_fg_fallback="",
    filter_bgfg_nsigma=0,
    filter_bgfg_fallback="",
    filter_bg_thres=0,
    filter_bg_fallback="",
    filter_bgfg_topk=0,
    filter_bgfg=False,
    ctx_mode=False,
    no_skip_stuff_bg=False,
    stuff_no_kmeans=False,
):

    classlist_nobg = [c for c in classlist if "background" not in c]

    fg_suffixs = [""]
    bg_suffixs = ["_bg"]
    if feat_mode == "ind":
        fg_suffixs = ["_ind"]
        bg_suffixs = ["_bg_ind"]
    if feat_mode == "kmeans":
        fg_suffixs = ["_kmeans"]
        bg_suffixs = ["_kmeans_bg"]
    if feat_mode == "mean_ind":
        fg_suffixs = ["", "_ind"]
        bg_suffixs = ["_bg", "_bg_ind"]
    if feat_mode == "ind_kmeans":
        fg_suffixs = ["_ind", "_kmeans"]
        bg_suffixs = ["_bg_ind", "_kmeans_bg"]
    if feat_mode == "mean_kmeans":
        fg_suffixs = ["", "_kmeans"]
        bg_suffixs = ["_bg", "_kmeans_bg"]
    if feat_mode == "all":
        fg_suffixs = ["", "_ind", "_kmeans"]
        bg_suffixs = ["_bg", "_bg_ind", "_kmeans_bg"]

    fg_prototypes = []
    fg_proto_classes = []

    for cid, c in enumerate(classlist_nobg):
        eff_fg_suffixs = [x for x in fg_suffixs]
        if stuff_no_kmeans and c in mmdata.IS_STUFF and mmdata.IS_STUFF[c]:
            eff_fg_suffixs = [x for x in eff_fg_suffixs if x != "_kmeans"]
        fg_prototypes_ = [prototypes[c + fg_suffix] for fg_suffix in eff_fg_suffixs]
        fg_prototypes_ = [p.view(-1, p.shape[-1]) for p in fg_prototypes_]
        fg_prototypes_ = torch.cat(fg_prototypes_, dim=0)

        if kmeans > 0:
            if fg_prototypes_.shape[0] > kmeans:
                kmeans = KMeans(n_clusters=kmeans, random_state=0, n_init="auto").fit(fg_prototypes_)
                fg_prototypes_ = torch.from_numpy(kmeans.cluster_centers_).to(fg_prototypes_)
            else:
                fg_prototypes_ = fg_prototypes_

        if filter_fg_nsigma > 0:
            fg_prototypes_ = filter_proto_fg2fg(fg_prototypes_, c, filter_fg_nsigma, filter_fg_fallback)
        print("fg", cid, c, fg_prototypes_.shape)
        fg_prototypes.append(fg_prototypes_)
        fg_proto_classes.append([cid + 1] * fg_prototypes_.shape[0])

    fg_proto_classes = sum(fg_proto_classes, start=[])
    fg_proto_classes = torch.tensor(fg_proto_classes, dtype=torch.long)
    fg_prototypes = torch.cat(fg_prototypes, dim=0)

    bg_prototypes = []
    bg_proto_classes = []

    for cid, c in enumerate(classlist_nobg):
        if ctx_mode and c in mmdata.IS_STUFF and mmdata.IS_STUFF[c] and not no_skip_stuff_bg:
            print("Skipping context class", c)
            continue
        bg_prototypes_ = [prototypes[c + bg_suffix] for bg_suffix in bg_suffixs]
        bg_prototypes_ = [p.view(-1, p.shape[-1]) for p in bg_prototypes_]
        bg_prototypes_ = [p[p.isfinite().any(-1)] for p in bg_prototypes_]  # only take finite values
        bg_prototypes_ = torch.cat(bg_prototypes_, dim=0)
        if kmeans > 0:
            if bg_prototypes_.shape[0] > kmeans:
                kmeans = KMeans(n_clusters=kmeans, random_state=0, n_init="auto").fit(bg_prototypes_)
                bg_prototypes_ = torch.from_numpy(kmeans.cluster_centers_).to(bg_prototypes_)
            else:
                bg_prototypes_ = bg_prototypes_
        if filter_bgfg_nsigma > 0:
            fg_proto = fg_prototypes[fg_proto_classes == cid + 1]
            bg_prototypes_ = filter_proto_bg2fg(bg_prototypes_, fg_proto, c, filter_bgfg_nsigma, filter_bgfg_fallback)

        if filter_bg_thres > 0:
            if ctx_mode:
                ctx_proto_mask = torch.tensor(
                    [cid + 1 for cid, c in enumerate(classlist_nobg) if c in mmdata.IS_STUFF and mmdata.IS_STUFF[c]]
                )
                ctx_proto_mask = (fg_proto_classes[..., None] == ctx_proto_mask[None]).any(-1)
                fg_proto = fg_prototypes[ctx_proto_mask]
            else:
                fg_proto = fg_prototypes[fg_proto_classes != cid + 1]
            bg_prototypes_ = filter_proto_bg2allfg(bg_prototypes_, fg_proto, c, filter_bg_thres, filter_bg_fallback)

        if filter_bgfg_topk > 0:
            fg_proto = fg_prototypes[fg_proto_classes == cid + 1]
            bg_prototypes_ = filter_proto_bg2fg_best(bg_prototypes_, fg_proto, c, filter_bgfg_topk)

        if filter_bgfg:
            for ocid, oc in enumerate(classlist_nobg):
                if ocid == cid:
                    continue
                fg_proto = fg_prototypes[fg_proto_classes == ocid + 1]
                bg_prototypes_ = filter_proto_bg2fg_inter(bg_prototypes_, fg_proto, c, oc, filter_bgfg_fallback)
        print("bg", cid, c, bg_prototypes_.shape)
        bg_prototypes.append(bg_prototypes_)
        bg_proto_classes.append([cid + 1] * bg_prototypes_.shape[0])

    bg_prototypes = torch.cat(bg_prototypes, dim=0)
    bg_proto_classes = sum(bg_proto_classes, start=[])
    bg_proto_classes = torch.tensor(bg_proto_classes, dtype=torch.long)

    if bgmode == "zero":
        bg_prototypes = torch.zeros_like(bg_prototypes[:1])
        bg_proto_classes = torch.zeros_like(bg_proto_classes[:1])
    elif bgmode == "mean":
        bg_prototypes = torch.mean(bg_prototypes, dim=0, keepdim=True)
        bg_proto_classes = torch.zeros_like(bg_proto_classes[:1])

    all_prototypes = torch.cat([bg_prototypes, fg_prototypes], dim=0).to(device)
    # all_prototypes_norm = all_prototypes / all_prototypes.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    proto_classes = torch.cat([bg_proto_classes, fg_proto_classes], dim=0).to(device)
    proto_class_mask = torch.cat([torch.zeros_like(bg_proto_classes), fg_proto_classes], dim=0)

    proto_class_mask_aggr = (proto_class_mask.unsqueeze(0) == torch.arange(len(classlist)).unsqueeze(-1)).to(device)[
        ..., None, None
    ]  # [num_classes, num_prototypes, 1, 1]

    return all_prototypes, proto_classes, proto_class_mask, proto_class_mask_aggr
