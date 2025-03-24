import argparse
import torch
from mmaction.registry import MODELS
from mmengine.runner import save_checkpoint
from mmaction.utils import register_all_modules


register_all_modules()


def process_checkpoint(args, out_path):
    in_path0=args.in_file0
    in_path1=args.in_file1
    videomae_checkpoint0 = torch.load(in_path0, map_location="cpu")
    # videomae_checkpoint1 = torch.load(in_path1, map_location="cpu")
    # keys=[]
    # for key in videomae_checkpoint0.keys():
    #     if "vision_encoder.encoder" in key:
    #         # key = key.replace("vision_encoder.encoder", "")
    #         keys.append(key)
    # print(videomae_checkpoint0.keys())
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
    # return

    model_cfg = dict(
        type="Recognizer3D",
        backbone=dict(
            type="VisionTransformer",
            img_size=224,
            patch_size=args.patch_size,
            embed_dims=args.embed_dims,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            norm_cfg=dict(type="LN", eps=1e-6),
            tubelet_size=1,
        ),
        cls_head=dict(type="TimeSformerHead", num_classes=700, in_channels=args.embed_dims, average_clips="prob"),
        data_preprocessor=dict(
            type="ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
    )
    

    model = MODELS.build(model_cfg)
    # print(model.backbone.state_dict()['patch_embed.projection.weight'].shape)
    # return 
    # video_state_dict = videomae_checkpoint["module"]
    video_state_dict = videomae_checkpoint0

    new_state_dict = {}
    for key, value in video_state_dict.items():
        key = key.replace("vision_encoder.encoder.", "")
        # convert keys
        if "fc1" in key:
            key = key.replace("fc1", "layers.0.0")
        elif "fc2" in key:
            key = key.replace("fc2", "layers.1")
        elif "patch_embed.proj" in key:
            key = key.replace("patch_embed.proj", "patch_embed.projection")
        elif "head" in key:
            key = key.replace("head", "cls_head.fc_cls")
        elif "norm." in key:
            key = key.replace("norm", "fc_norm")

        if "backbone." + key in model.state_dict().keys():  # blocks.0.xxx
            new_state_dict["backbone." + key] = value
        elif key.startswith("cls_head") and key in model.state_dict().keys():
            new_state_dict[key] = value

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
    parser.add_argument("--embed_dims", default=1408, type=int)
    parser.add_argument("--depth", default=40, type=int)
    parser.add_argument("--num_heads", default=16, type=int)
    parser.add_argument("--patch_size", default=14, type=int)
    parser.add_argument("--out_file", default='', help="output checkpoint path")
    args = parser.parse_args()

    process_checkpoint(args, args.out_file)

"""example
python tools/model_converters/convert_videomaev2.py \
   vit_g_hybrid_pt_1200e_k710_ft.pth pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth
"""
