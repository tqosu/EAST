_base_ = ["e2e_gtea_videomae_s_768x1_160_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            patch_size=16,
            embed_dims=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            adapter_index=list(range(24)),
            tubelet_size=1,
        ),
        custom=dict(pretrain="pretrained/ret_ssv2_tpl_l16_25m_my.pth"),
    ),
    projection=dict(in_channels=1024),
)

work_dir = "exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter"
metr_path = "epoch_metrics_mAP1.csv"

evaluation = dict(
    type="mAP2_4",
    subset="validation",
    tiou_thresholds=[0.1, 0.25, 0.5],
    ground_truth_filename="data/annotations/",
    top_k=[1],
    nms_score_t=0.4,
    bg_class=['background']
)