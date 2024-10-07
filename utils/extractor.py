import functools
import logging
import math
import types
from pathlib import Path
from typing import List, Tuple, Union

import clip
import timm
import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from clip.model import VisionTransformer as _VisionTransformer
from PIL import Image

LOGGER = logging.getLogger(__name__)


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    print(x.shape)
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class PatchResize(torch.nn.Module):
    def __init__(self, patch_size=16, interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__()
        self.patch_size = patch_size
        self.interpolation = interpolation
        self.to_pil = interpolation == TF.InterpolationMode.LANCZOS

    def forward(self, imgt):
        W, H = TF.get_image_size(imgt)
        nW, nH = int(round(W / self.patch_size) * self.patch_size), int(round(H / self.patch_size) * self.patch_size)
        if torch.is_tensor(imgt) and self.to_pil:
            imgt = TF.to_pil_image(imgt if len(imgt.shape) <= 3 else imgt[0])
        imgt = TF.resize(imgt, (nH, nW), interpolation=self.interpolation)
        if not torch.is_tensor(imgt):
            imgt = TF.to_tensor(imgt)
        return imgt.view(1, 3, nH, nW)


class VitExtractor(torch.nn.Module):
    BLOCK_KEY = "block"
    ATTN_KEY = "attn"
    PATCH_IMD_KEY = "patch_imd"
    QKV_KEY = "qkv"
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, frozen=True):
        super(VitExtractor, self).__init__()
        self.model = torch.hub.load("facebookresearch/dino:main", model_name)
        self.frozen = frozen
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()
        self.preprocess = T.Compose([T.Resize(224), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def forward(self, input_img, layer_num=-1):
        self._register_hooks()
        self.model(input_img)
        cls = self.outputs_dict[VitExtractor.BLOCK_KEY][layer_num][:, 0]  # [B, D]
        qkv_features = self.outputs_dict[VitExtractor.QKV_KEY][layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)[:, :, 1:]  # [B, n_heads, n_patches, D]
        self._clear_hooks()
        self._init_hooks_data()
        return cls, keys

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def _format(self, qkv, input_img_shape):
        b, c, h, w = input_img_shape
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        qkv = qkv.reshape(b, patch_num, 3, head_num, embedding_dim // head_num).permute(2, 0, 3, 1, 4)
        return qkv

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        qkv = self._format(qkv, input_img_shape)
        return qkv[1]

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_queries_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        queries = self.get_queries_from_qkv(qkv_features, input_img.shape)
        return queries

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_values_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        values = self.get_values_from_qkv(qkv_features, input_img.shape)
        return values

    def get_cls_from_input(self, input_img, layer_num):
        feature = self.get_feature_from_input(input_img)[layer_num]
        cls = feature[:, 0]
        return cls

    def get_qkv_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        queries = self.get_queries_from_qkv(qkv_features, input_img.shape)
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        values = self.get_values_from_qkv(qkv_features, input_img.shape)
        concatenated_qkv = torch.cat([queries, keys, values], dim=-1)
        return concatenated_qkv

    def get_queries_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_queries_from_input(input_img, layer_num=layer_num).squeeze(0)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        LOGGER.info_once(f"keys shape: {keys.shape}")
        b, h, t, d = keys.shape
        concatenated_keys = keys.permute(0, 2, 1, 3).reshape(b, t, h * d)
        norm_keys = torch.norm(concatenated_keys, dim=-1, keepdim=True)
        denom = torch.clamp(norm_keys * norm_keys.transpose(-1, -2), 1e-8)
        ssim = torch.einsum("btd,bTd->btT", concatenated_keys, concatenated_keys) / denom
        return ssim

    def get_values_self_sim_from_input(self, input_img, layer_num):
        values = self.get_values_from_input(input_img, layer_num=layer_num).squeeze(0)
        h, t, d = values.shape
        concatenated_values = values.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, ...])
        return ssim_map

    def get_qkv_self_sim_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape).squeeze(0)
        values = self.get_values_from_qkv(qkv_features, input_img.shape).squeeze(0)
        queries = self.get_queries_from_qkv(qkv_features, input_img.shape).squeeze(0)
        # print("qvd self sim", queries.shape, keys.shape, values.shape)
        concatenated_qkv = torch.cat([queries, keys, values], dim=-1)
        h, t, d = concatenated_qkv.shape
        concatenated_qkv = concatenated_qkv.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_qkv[None, None, ...])
        return ssim_map


class MaskedViTExtractor(VitExtractor):
    def generate_ids(self, input_img, mask):
        B = 1
        len_keep = mask.shape[1] if mask.dtype == torch.long else (~mask).float().sum(-1).long().item()
        assert input_img.shape[0] == B, "batch size must be 1"
        L = self.get_patch_num(input_img.shape)
        D = self.get_embedding_dim()
        n = torch.zeros(B, L - 1, device=input_img.device)
        n[mask] = 1
        cls = -torch.ones(B, 1, device=input_img.device)
        n = torch.cat([cls, n], dim=-1)
        ids_shuffle = torch.argsort(n, dim=-1)
        ids_restore = torch.argsort(ids_shuffle, dim=-1)
        ids_shuffle = ids_shuffle[:, : len_keep + 1]
        return ids_shuffle, ids_restore

    def __get_mask_hook(self, ids):
        def hook(module, input, output):
            B, L, D = output.shape
            modified_output = torch.gather(output, 1, ids[..., None].expand(-1, -1, D))
            # print(f"output.shape {output.shape} modified_output.shape: {modified_output.shape}")
            return modified_output

        return hook

    def __restore_feature(self, ids, feature):
        B, L, D = feature.shape
        b, N = ids.shape
        assert B == b, "batch size must be 1"
        f = torch.zeros(B, N - L, D, device=feature.device)
        f = torch.cat([feature, f], dim=1)
        return torch.gather(f, 1, ids[..., None].expand(-1, -1, D))

    def get_feature_from_input(self, input_img, mask):  # List([B, N, D])
        ids_fwd, ids_rest = self.generate_ids(input_img, mask)
        self.hook_handlers.append(self.model.pos_drop.register_forward_hook(self.__get_mask_hook(ids_fwd)))
        feature = super().get_feature_from_input(input_img)
        return feature

    def get_qkv_feature_from_input(self, input_img, mask):
        ids_fwd, ids_rest = self.generate_ids(input_img, mask)
        self.hook_handlers.append(self.model.pos_drop.register_forward_hook(self.__get_mask_hook(ids_fwd)))
        qkv_features = super().get_qkv_feature_from_input(input_img)
        qkv_features = [self.__restore_feature(ids_rest, qkv_feature) for qkv_feature in qkv_features]
        return qkv_features

    def get_queries_from_input(self, input_img, mask, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img, mask)[layer_num]
        queries = self.get_queries_from_qkv(qkv_features, input_img.shape)
        return queries

    def get_keys_from_input(self, input_img, mask, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img, mask)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_values_from_input(self, input_img, mask, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img, mask)[layer_num]
        values = self.get_values_from_qkv(qkv_features, input_img.shape)
        return values

    def get_keys_self_sim_from_input(self, input_img, mask, layer_num):
        keys = self.get_keys_from_input(input_img, mask, layer_num=layer_num).squeeze(0)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map


def to_single_patchsize(ps):
    if isinstance(ps, (int, float)):
        return ps
    elif isinstance(ps, (tuple, list)):
        assert len(ps) == 2, "patch size must be int or list of 2"
        assert ps[0] == ps[1], "patch must be square"
        return ps[0]
    else:
        raise ValueError("patch size must be int or list of 2")


class MAEViT(timm.models.vision_transformer.VisionTransformer):
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        patch_size = to_single_patchsize(self.patch_embed.patch_size)
        w0 = w // patch_size
        h0 = h // patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def _pos_embed(self, x):
        # This follows the official timm implementation of ViT
        # Except adds interpolations for pos_embed
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.interpolate_pos_encoding(x, self.__curr_width, self.__curr_height)
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, self.__curr_width, self.__curr_height)
        return self.pos_drop(x)

    def forward_features(self, x):
        self.__curr_batch, self.__curr_channels, self.__curr_height, self.__curr_width = x.shape
        # Hack around assert in patch_embed for img_size check
        self.patch_embed.img_size = (self.__curr_height, self.__curr_width)

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = timm.models.helpers.checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x


def mae_vit_base_patch16(**kwargs):
    model = MAEViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = MAEViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MAEViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


class StridedViTExtractor:
    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(
        self, model_type: str = "dino_vits8", stride: int = None, model: nn.Module = None, device: str = "cuda"
    ):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device

        if model is not None:
            self.model = model
        else:
            self.model = StridedViTExtractor.create_model(model_type)

        stride = stride if stride is not None else self.model.patch_embed.proj.stride[0]

        self.model = StridedViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = to_single_patchsize(self.model.patch_embed.patch_size)
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        if "dinov2" in model_type:
            model = torch.hub.load("facebookresearch/dinov2:main", model_type)
        elif "dino" in model_type:
            model = torch.hub.load("facebookresearch/dino:main", model_type)
        elif "mae" in model_type:
            model = {
                "mae_pretrain_vit_huge": mae_vit_huge_patch14,
                "mae_pretrain_vit_large": mae_vit_large_patch16,
                "mae_pretrain_vit_base": mae_vit_base_patch16,
                "mae_longseq_vit_base_imgnet": functools.partial(mae_vit_base_patch16, img_size=448),
                "mae_longseq_vit_large_imgnet": functools.partial(mae_vit_large_patch16, img_size=448),
                "mae_longseq_vit_base_coco": functools.partial(mae_vit_base_patch16, img_size=448),
                "mae_longseq_vit_large_coco": functools.partial(mae_vit_large_patch16, img_size=448),
            }[model_type]()
            ckpt_name = {
                "mae_pretrain_vit_huge": "mae_pretrain_vit_huge",
                "mae_pretrain_vit_large": "mae_pretrain_vit_large",
                "mae_pretrain_vit_base": "mae_pretrain_vit_base",
                "mae_longseq_vit_base_imgnet": "vitb_dec384d12h8b_1600ep_img448_crop0.2-1.0_maskds2",
                "mae_longseq_vit_large_imgnet": "vitl_dec512d16h8b_1600ep_img448_crop0.2-1.0_maskds2",
                "mae_longseq_vit_base_coco": "vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2",
                "mae_longseq_vit_large_coco": "vitl_dec512d16h8b_800ep_img448_crop0.2-1.0_maskds2",
            }
            print("loading pretrained model for mae:", f"pretrained_models/{ckpt_name[model_type]}.pth")
            ckpt = torch.load(f"pretrained_models/{ckpt_name[model_type]}.pth", map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                "vit_small_patch16_224": "dino_vits16",
                "vit_small_patch8_224": "dino_vits8",
                "vit_base_patch16_224": "dino_vitb16",
                "vit_base_patch8_224": "dino_vitb8",
            }
            model = torch.hub.load("facebookresearch/dino:main", model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict["head.weight"]
            del temp_state_dict["head.bias"]
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        patch_size = to_single_patchsize(patch_size)
        # print(patch_size)
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all(
            [(patch_size // s_) * s_ == patch_size for s_ in stride]
        ), f"stride {stride} should divide patch_size {patch_size}"

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(StridedViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(
        self, image_path: Union[str, Path], load_size: Union[int, Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        pil_image = Image.open(image_path).convert("RGB")
        if load_size is not None:
            pil_image = T.Resize(load_size, interpolation=T.InterpolationMode.LANCZOS)(pil_image)
        prep = T.Compose([T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        layers = [l if l >= 0 else len(self.model.blocks) + l for l in layers]
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == "attn":
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = -1, facet: str = "key") -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3**k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3**k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim : (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                    :, :, i, j
                                ]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim : (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                    :, :, temp_i, temp_j
                                ]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(
        self, batch: torch.Tensor, layer: int = 11, facet: str = "key", bin: bool = False, include_cls: bool = False
    ) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in [
            "key",
            "query",
            "value",
            "token",
        ], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        del self._feats
        if facet == "token":
            x.unsqueeze_(dim=1)  # Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.model_type == "dino_vits8", f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], "attn")
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0]  # Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1)  # Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
        return cls_attn_maps


def mod_attention(self, x: torch.Tensor):
    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask, average_attn_weights=False)[0]


class ModifiedVisionTransformer(_VisionTransformer):
    def __init__(self, clip_visual):
        input_resolution = clip_visual.input_resolution
        output_dim = clip_visual.output_dim
        width = clip_visual.transformer.width
        patch_size = clip_visual.conv1.kernel_size[0]
        layers = clip_visual.transformer.layers
        heads = clip_visual.transformer.resblocks[0].attn.num_heads
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        self.load_state_dict(clip_visual.state_dict(), strict=True)
        self.width = width
        self.stride = to_single_patchsize(self.conv1.stride[0])
        self.patch_size = to_single_patchsize(self.conv1.kernel_size[0])
        for b in self.transformer.resblocks:
            b.attention = types.MethodType(mod_attention, b)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        # print(x.shape, w, h)
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[0].unsqueeze(0)
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - to_single_patchsize(self.patch_size)) // to_single_patchsize(self.stride)
        h0 = 1 + (h - to_single_patchsize(self.patch_size)) // to_single_patchsize(self.stride)
        assert (
            w0 * h0 == npatch
        ), f"""got wrong grid size for {h}x{w} with patch_size {self.patch_size} and 
                                            stride {self.stride} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=0)

    def forward(self, x: torch.Tensor):
        _, __, h, w = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.positional_embedding.to(x.dtype)
        x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)
        if self.proj is not None:
            # x = x @ self.proj
            x = torch.einsum("nlw,wd->nld", x, self.proj)

        return x


class StridedCLIPExtractor:
    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(
        self, model_type: str = "clip_ViT-B/16", stride: int = None, model: nn.Module = None, device: str = "cuda"
    ):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device

        if model is not None:
            self.model = model
        else:
            self.model = StridedCLIPExtractor.create_model(model_type)

        stride = stride if stride is not None else self.model.stride
        self.model = StridedCLIPExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = to_single_patchsize(self.model.patch_size)
        self.stride = to_single_patchsize(self.model.stride)
        self.stride = (self.stride, self.stride)

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        assert model_type.startswith("clip_"), f"invalid model type {model_type}"
        model_type = model_type.replace("clip_", "")
        clip_model, processor = clip.load(model_type, device="cpu", jit=False)
        model = ModifiedVisionTransformer(clip_model.visual)
        return model

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.conv1.kernel_size
        patch_size = to_single_patchsize(patch_size)
        # print(patch_size)
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all(
            [(patch_size // s_) * s_ == patch_size for s_ in stride]
        ), f"stride {stride} should divide patch_size {patch_size}"

        # fix the stride
        new_conv1 = nn.Conv2d(
            in_channels=3, out_channels=model.width, kernel_size=patch_size, stride=stride, bias=False
        )
        new_conv1.load_state_dict(model.conv1.state_dict())
        model.conv1 = new_conv1
        model.stride = stride

        return model

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["token"]:

            def _hook(model, input, output):
                self._feats.append(output.transpose(0, 1))

            return _hook
        if facet in ["attn"]:

            def _hook(model, input, output):
                self._feats.append(output[1])

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            query, key, value, *_ = input
            qkv = nn.functional._in_projection_packed(query, key, value, module.in_proj_weight, module.in_proj_bias)
            f = qkv[facet_idx]  # L, B, D
            L, B, D = f.shape
            f = f.permute(1, 0, 2).reshape(B, L, module.num_heads, D // module.num_heads).transpose(1, 2)  # B, h t, d
            self._feats.append(f)

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        layers = [l if l >= 0 else len(self.model.transformer.resblocks) + l for l in layers]
        for block_idx, block in enumerate(self.model.transformer.resblocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == "attn":
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = -1, facet: str = "key") -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def extract_descriptors(
        self, batch: torch.Tensor, layer: int = 11, facet: str = "key", bin: bool = False, include_cls: bool = False
    ) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in [
            "key",
            "query",
            "value",
            "token",
        ], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        del self._feats
        # print(x.shape, facet)
        if facet == "token":
            x.unsqueeze_(dim=1)  # Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc
