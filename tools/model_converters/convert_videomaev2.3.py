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
    # videomae_checkpoint1 = torch.load(in_path1, map_location="cpu")
    # keys0=[]
    # for key in videomae_checkpoint0["module"].keys():
    #     if "fc_cls" in key:
    #         keys0.append(key)
    # print(keys0)
    # print()
    # return

    # keys1=[]
    # for key in videomae_checkpoint1["module"].keys():
    #     if "blocks.25" in key:
    #         keys1.append(key)
    # print(videomae_checkpoint0["module"].keys(),len(videomae_checkpoint0["module"].keys()))
    # print(videomae_checkpoint1["module"].keys(),len(videomae_checkpoint1["module"].keys()))
    # set0=set(videomae_checkpoint0["module"].keys())
    # set1=set(videomae_checkpoint1["module"].keys())

    # set0=set(keys0)
    # set1=set(keys1)
    # print(set0-set1)
    # print(set1-set0)
    # print(videomae_checkpoint0['patch_embed.proj.weight'].shape)
    # print(videomae_checkpoint1["module"]['patch_embed.proj.weight'].shape)
    # return 

    # print(keys,len(keys))
    # print(videomae_checkpoint1["module"].keys(),len(videomae_checkpoint1["module"].keys()))
    # print(len(videomae_checkpoint0.keys()))
    # print(len(videomae_checkpoint1.keys()))
    # return 
    # for key in videomae_checkpoint0.keys():
    #     key1=key.split('backbone.')[-1]
    #     if "head" in key:
    #         key = key.replace("head", "cls_head.fc_cls")
    #     ese:
        
    #     # break
    # print(videomae_checkpoint0["module"].keys())
    # return

    model_cfg = dict(
        type="Recognizer3D",
        backbone=dict(
            type="InternVideo2",
            img_size=224, patch_size=14, embed_dim=1408, 
        depth=40, num_heads=16, mlp_ratio=48/11, 
        attn_pool_num_heads=16, clip_embed_dim=768,num_classes=710
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
    # print(model.state_dict().keys())
    # return 
    # print(" # location 1")
    # return 
    # print(model.backbone.state_dict()['patch_embed.projection.weight'].shape)
    # return 
    # video_state_dict = videomae_checkpoint["module"]
    video_state_dict = videomae_checkpoint0["module"]

    new_state_dict = {}
    for key, value in video_state_dict.items():
        # key = 'backbone.'+key
        # convert keys
        # if "fc1" in key:
        #     key = key.replace("fc1", "layers.0.0")
        # elif "fc2" in key:
        #     key = key.replace("fc2", "layers.1")
        # elif "patch_embed.proj" in key:
        #     key = key.replace("patch_embed.proj", "patch_embed.projection")
        # elif "head" in key:
        #     key = key.replace("head", "cls_head.fc_cls")
        # elif "norm." in key:
        #     key = key.replace("norm", "fc_norm")
        # if key =='head.bias':continue
        # if key =='head.weight':continue
        if "backbone." + key in model.state_dict().keys():  # blocks.0.xxx
            new_state_dict["backbone." + key] = value
        # elif key.startswith("head"):
            
        #     # key1='cls_head.fc_cls.'+key.split('.')[-1]
        #     # print(key,key1)
        #     if key1 in model.state_dict().keys():
        #         new_state_dict[key1] = value

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

