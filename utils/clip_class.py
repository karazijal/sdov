from itertools import chain, combinations

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


class CLIP(nn.Module):
    def __init__(self, device, key="ViT-B/16"):
        super().__init__()

        self.device = device

        self.clip_model, self.clip_preprocess = clip.load(key, device=self.device, jit=False)

        # image augmentation
        self.aug = T.Compose(
            [
                T.Resize((336, 336) if "@336px" in key else (224, 224)),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        # self.gaussian_blur = T.GaussianBlur(15, sigma=(0.1, 10))

    def get_text_embeds(self, prompt, negative_promp, normalize=True):

        # NOTE: negative_prompt is ignored for CLIP.

        text = clip.tokenize(prompt).to(self.device)
        text_z = self.clip_model.encode_text(text)
        if normalize:
            text_z = text_z / text_z.norm(dim=-1, keepdim=True)

        return text_z

    def train_step(self, text_z, pred_rgb):

        pred_rgb = self.aug(pred_rgb)

        image_z = self.clip_model.encode_image(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True)  # normalize features

        loss = -(image_z * text_z).sum(-1).mean()

        return loss


class CLIPClassifier(nn.Module):
    def __init__(self, device, class_list, topk=10, connector=" and ", clip_key="ViT-B/16"):
        super().__init__()
        self.device = device
        self.clip = CLIP(self.device, key=clip_key)
        self.clip.eval()
        self.topk = topk

        self.classes = class_list
        self._cls_text_emb = None
        self.conn = connector
        print(len(self.classes))
        self._fs_metrics = {
            "p": Precision(task="multilabel", num_labels=len(self.classes), average="macro").to(device),
            "r": Recall(task="multilabel", num_labels=len(self.classes), average="macro").to(device),
        }
        self._ss_metrics = {
            "f1": F1Score(task="multilabel", num_labels=len(self.classes), average="macro").to(device),
            "acc": Accuracy(task="multilabel", num_labels=len(self.classes), average="macro").to(device),
        }
        self._per_cls_metrics = [
            {
                "p": Precision(task="binary").to(device),
                "r": Recall(task="binary").to(device),
                "f1": F1Score(task="binary").to(device),
                "acc": Accuracy(task="binary").to(device),
            }
            for _ in range(len(self.classes))
        ]

    @property
    def cls_text_emb(self):
        if self._cls_text_emb is None:
            self._cls_text_emb = self.clip.get_text_embeds([f"a " + c for c in self.classes], "").float()
        return self._cls_text_emb

    def forward(self, img):
        img = img.to(self.device).view(1, 3, *img.shape[-2:])
        img_embs = self.clip.clip_model.encode_image(self.clip.aug(img)).float()
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

        lbl_scores = img_embs @ self.cls_text_emb.T
        pred_class_inds_fs = (torch.topk(lbl_scores, self.topk, dim=-1).indices).cpu().squeeze().tolist()

        first_stage_preds = torch.zeros(len(self.classes), dtype=torch.long, device=self.device)
        # first_stage_preds[pred_class_inds_fs] = 1
        for cid in pred_class_inds_fs:
            first_stage_preds[cid] = 1
        self.first_stage_preds = first_stage_preds[None]

        second_stage_combs = list(powerset(pred_class_inds_fs))
        prompts = [self.conn.join(["a " + self.classes[cid] for cid in comb]) for comb in second_stage_combs]
        text_embs = self.clip.get_text_embeds(prompts, "").float()
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        pred_ind = (img_embs @ text_embs.T).squeeze().argmax(dim=-1).item()
        pred_class_inds = second_stage_combs[pred_ind]

        preds = torch.zeros(len(self.classes), dtype=torch.long, device=self.device)
        # preds[pred_class_inds] = 1
        for cid in pred_class_inds:
            preds[cid] = 1
        self.second_stage_preds = preds[None]

        return pred_class_inds

    def update(self, gt):
        if not torch.is_tensor(gt):
            gt = torch.tensor(gt, device=self.device).long()
        if gt.shape[-1] != len(self.classes):
            _gt = torch.zeros(len(self.classes), dtype=torch.long, device=self.device)
            for i in gt:
                _gt[i] = 1
            gt = _gt
        gt = gt.to(self.device).view(1, len(self.classes))
        self._fs_metrics["p"](self.first_stage_preds, gt)
        self._fs_metrics["r"](self.first_stage_preds, gt)
        self._ss_metrics["f1"](self.second_stage_preds, gt)
        self._ss_metrics["acc"](self.second_stage_preds, gt)
        for i, (p, t) in enumerate(zip(self.second_stage_preds.flatten(), gt.flatten())):
            p = (p > 0).long().view(1, 1)
            t = (t > 0).long().view(1, 1)
            for k, v in self._per_cls_metrics[i].items():
                v(p, t)

    def compute(self):
        return {k: v.compute().squeeze().item() for k, v in {**self._fs_metrics, **self._ss_metrics}.items()}


class CLIPClassifierThresh(CLIPClassifier):

    def forward(self, img):
        img = img.to(self.device).view(1, 3, *img.shape[-2:])
        img_embs = self.clip.clip_model.encode_image(self.clip.aug(img)).float()
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

        lbl_scores = img_embs @ self.cls_text_emb.T
        probs = torch.softmax(lbl_scores, dim=-1).squeeze()
        tp = probs >= (1 / len(self.classes))
        pred_class_inds_fs = torch.where(tp)[0].cpu().tolist()

        if len(pred_class_inds_fs) == 0:
            pred_class_inds_fs = (torch.topk(lbl_scores, self.topk, dim=-1).indices).cpu().squeeze().tolist()

        first_stage_preds = torch.zeros(len(self.classes), dtype=torch.long, device=self.device)
        # first_stage_preds[pred_class_inds_fs] = 1
        for cid in pred_class_inds_fs:
            first_stage_preds[cid] = 1
        self.first_stage_preds = first_stage_preds[None]

        if len(pred_class_inds_fs) > self.topk:
            _probs = torch.where(tp, probs, torch.zeros_like(probs))
            pred_class_inds_fs = (torch.topk(_probs, self.topk, dim=-1).indices).cpu().squeeze().tolist()

        second_stage_combs = list(powerset(pred_class_inds_fs))
        prompts = [self.conn.join(["a " + self.classes[cid] for cid in comb]) for comb in second_stage_combs]
        text_embs = self.clip.get_text_embeds(prompts, "").float()
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        pred_ind = (img_embs @ text_embs.T).squeeze().argmax(dim=-1).item()
        pred_class_inds = second_stage_combs[pred_ind]

        preds = torch.zeros(len(self.classes), dtype=torch.long, device=self.device)
        # preds[pred_class_inds] = 1
        for cid in pred_class_inds:
            preds[cid] = 1
        self.second_stage_preds = preds[None]

        return pred_class_inds


class ScoreThreshold(nn.Module):
    def __init__(self, scorer, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.scorer = scorer
        self.classes = scorer.classes
        self._cls_text_emb = None
        self.conn = " and "
        device = scorer.device
        self.device = device
        print(len(self.classes))
        self._fs_metrics = {
            "p": Precision(task="multilabel", num_labels=len(self.classes), average="macro").to(device),
            "r": Recall(task="multilabel", num_labels=len(self.classes), average="macro").to(device),
        }
        self._ss_metrics = {
            "f1": F1Score(task="multilabel", num_labels=len(self.classes), average="macro").to(device),
            "acc": Accuracy(task="multilabel", num_labels=len(self.classes), average="macro").to(device),
        }
        self._per_cls_metrics = [
            {
                "p": Precision(task="binary").to(device),
                "r": Recall(task="binary").to(device),
                "f1": F1Score(task="binary").to(device),
                "acc": Accuracy(task="binary").to(device),
            }
            for _ in range(len(self.classes))
        ]

    def forward(self, x):
        scores = self.scorer(x)
        preds = torch.where(scores > 1.0 / len(self.scorer.classes) * self.threshold)[0].view(-1).long()

        first_stage_preds = torch.zeros(len(self.classes), dtype=torch.long, device=self.device)

        # first_stage_preds[pred_class_inds_fs] = 1
        for cid in preds:
            first_stage_preds[cid] = 1
        self.first_stage_preds = first_stage_preds[None]

        pred_class_inds = torch.where(first_stage_preds.squeeze())[0].cpu().tolist()

        self.second_stage_preds = self.first_stage_preds

        return pred_class_inds

    def update(self, gt):
        if not torch.is_tensor(gt):
            gt = torch.tensor(gt, device=self.device).long()
        if gt.shape[-1] != len(self.classes):
            _gt = torch.zeros(len(self.classes), dtype=torch.long, device=self.device)
            for i in gt:
                _gt[i] = 1
            gt = _gt
        gt = gt.to(self.device).view(1, len(self.classes))
        self._fs_metrics["p"](self.first_stage_preds, gt)
        self._fs_metrics["r"](self.first_stage_preds, gt)
        self._ss_metrics["f1"](self.second_stage_preds, gt)
        self._ss_metrics["acc"](self.second_stage_preds, gt)
        for i, (p, t) in enumerate(zip(self.second_stage_preds.flatten(), gt.flatten())):
            p = (p > 0).long().view(1, 1)
            t = (t > 0).long().view(1, 1)
            for k, v in self._per_cls_metrics[i].items():
                v(p, t)

    def compute(self):
        return {k: v.compute().squeeze().item() for k, v in {**self._fs_metrics, **self._ss_metrics}.items()}
