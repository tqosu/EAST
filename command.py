import argparse
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Description of your program',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add arguments
    parser.add_argument(
        '--split',
        type=str,
        default='1',
    )
    parser.add_argument(
        '--epoch',
        type=str,
        default='247',
    )
    

    return parser.parse_args()


def main():
    """Main entry point.
    
    Args:
        argv: List of command line arguments
        
    Returns:
        int: Return code (0 for success, non-zero for error)
    """
    args = parse_args()
    mystr1 = "CUDA_VISIBLE_DEVICES=6,7 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/gtea/e2e_gtea_ret_ssv2_tpl_l16_25m_768x1_160_adapter.py --cfg-options split=split{} annotation_path=data/annotations/gtea.split{}.json work_dir=exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter_10.13_ai feature_stride=4 model.backbone.backbone.type=VisionTransformerAdapter1 evaluation.save_npz=False model.backbone.backbone.adapter_index1=[".format(args.split, args.split, args.epoch)
    mystr2 = "]  --checkpoint /nfs/hpc/dgx2-6/tmp/2024/10/13/OpenTAD/exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter/split{}/gpu2_id0/checkpoint/epoch_{}.pth ".format(args.split, args.epoch)
    mystr3 = "&& \\"
    for i in range(24):
        if i != 23:
            print(mystr1 + str(i) + mystr2 + mystr3)
        else:
            print(mystr1 + str(i) + mystr2)


if __name__ == '__main__':
    sys.exit(main())
