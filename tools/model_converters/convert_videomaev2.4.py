import os
import sys
path = os.path.join(os.path.dirname(__file__), "..")
path = os.path.join(path, "..")
# print(path)
# return 
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
from mmaction.registry import MODELS
from mmengine.runner import save_checkpoint
from mmaction.utils import register_all_modules
from opentad.models import build_detector

register_all_modules()
# sys.dont_write_bytecode = True

def process_checkpoint(args, out_path):
    

    in_path0=args.in_file0
    in_path1=args.in_file1
    videomae_checkpoint0 = torch.load(in_path0, map_location="cpu")

    model_cfg = dict(
        type="Recognizer3D",
        backbone=dict(
            type="InternVideo2Adapter",
            img_size=224, patch_size=14, embed_dim=1408, 
            depth=40, num_heads=16, mlp_ratio=48/11, 
            attn_pool_num_heads=16, clip_embed_dim=768,
            num_classes=710,num_frames=16,
        ),
        # projection
        cls_head=dict(type="TimeSformerHead", num_classes=710, in_channels=1408, average_clips="prob"),
        data_preprocessor=dict(
            type="ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
    )
    

    # model = build_detector(model_cfg)
    
    model = MODELS.build(model_cfg)
    video_state_dict = videomae_checkpoint0["module"]

    new_state_dict = {}
    for key, value in video_state_dict.items():
        if key in ['pos_embed','cls_token']:continue
        if "backbone." + key in model.state_dict().keys():  # blocks.0.xxx
            new_state_dict["backbone." + key] = value

    print("The following keys exist in model_cfg but not in the new checkpoint:")
    for key, value in model.state_dict().items():
        if key not in new_state_dict.keys():
            print(key)

    model.load_state_dict(new_state_dict, strict=False)
    save_checkpoint(model.state_dict(), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VideoMAEv2 checkpoint")
    parser.add_argument("--in_file0", help="input checkpoint path")
    parser.add_argument("--in_file1", help="input checkpoint path")
    parser.add_argument("--out_file", default='', help="output checkpoint path")
    args = parser.parse_args()

    process_checkpoint(args, args.out_file)

"""example
python tools/model_converters/convert_videomaev2.py \
   vit_g_hybrid_pt_1200e_k710_ft.pth pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth
"""

