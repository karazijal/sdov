from types import SimpleNamespace

import torch
import torchvision
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from CutLER.cutler.config.cutler_config import add_cutler_config
from CutLER.cutler.modeling.meta_arch import build_model


def _setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Disable the use of SyncBN normalization when running on a CPU
    # SyncBN is not supported on CPU and can cause errors, so we switch to BN instead
    if cfg.MODEL.DEVICE == "cpu" and cfg.MODEL.RESNETS.NORM == "SyncBN":
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def _load_model(
    config_path="CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
    model_path="CutLER/cutler_cascade_final.pth",
):

    args = SimpleNamespace(
        config_file=config_path, confidence_threshold=0.35, opts=["MODEL.WEIGHTS", model_path, "MODEL.DEVICE", "cpu"]
    )
    cfg = _setup_cfg(args)

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    return model, cfg


class CUTLER(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.cfg = _load_model()
        self.model.eval()
        self.aug = torchvision.transforms.Resize(self.cfg.INPUT.MIN_SIZE_TEST, max_size=self.cfg.INPUT.MAX_SIZE_TEST)

    def forward(self, img):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        input = []
        for im in img:
            height, width = im.shape[-2:]
            im = self.aug(im)
            input.append({"image": im, "height": height, "width": width})
        return self.model(input)


class CUTLER_Processor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cutler = CUTLER()
        self.cutler.eval()

    def forward(self, img, device=None):
        if not torch.is_tensor(img):
            img = torchvision.transforms.functional.to_tensor(img) * 255
        device = device or img.device
        with torch.no_grad():
            res = self.cutler(img)
            masks = res[0]["instances"].pred_masks.to(device)
            masks = torch.cat(
                [1 - masks.any(dim=0, keepdim=True).float(), masks.float()], dim=0
            )  # Add BG mask in pos 0
        return masks
