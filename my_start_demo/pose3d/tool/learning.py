import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '..'))
sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
from functools import partial


def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print('load_weight', len(matched_layers))
    return model


def partial_train_layers(model, partial_list):
    """Train partial layers of a given model."""
    for name, p in model.named_parameters():
        p.requires_grad = False
        for trainable in partial_list:
            if trainable in name:
                p.requires_grad = True
                break
    return model


def load_backbone(cfg):
    if cfg.version == 1:
        from lib.model.mixste_model_plus_2 import MixSTE2 as Model
    elif cfg.version == 2:
        from lib.model.mixste_model_plus_2_plus import MixSTE2 as Model

    from lib.utils.load_parameter import parse_args
    args = parse_args()
    args.hyper_relation_bias = 'uniform'
    args.hyper_cfg = osp.join('..', r'hyper_config/method2/manual_joint_label3.yaml')
    args.spatial_mode = 'hyper_dependent_common'
    args.hyper_head = 1
    args.joint_label = [[0, 1, 1, 1, 2, 2, 2, 0, 0, 3, 3, 4, 4, 4, 5, 5, 5]]
    args.dataset = 'h36m'

    # here please attention, we follow the setting of coordination used by PoseConv3D and MotionBERT, we set (x,y,confidence).
    # therefore, the input channel should be  C=2+1
    model_backbone = Model(num_frame=cfg.maxlen, num_joints=cfg.num_joints,
                           in_chans=2 if cfg.no_conf else 3,
                           embed_dim_ratio=cfg.dim_rep,
                           depth=cfg.depth,
                           num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                           qkv_bias=True, qk_scale=None, drop_path_rate=0.1, hyper_opt=args,
                           graph_mode=args.hyper_relation_bias, hrnet=False)

    print('Warning:== input channel here is [x,y,confidence], which is 3 in total rather than common 3DHPE 2!!')

    return model_backbone
