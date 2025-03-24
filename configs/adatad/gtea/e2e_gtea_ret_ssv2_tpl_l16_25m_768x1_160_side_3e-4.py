_base_ = ["e2e_gtea_videomae_s_768x1_160_side.py"]

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
            # adapter_mlp_ratio=1 / 8,
        ),
        custom=dict(pretrain="pretrained/ret_ssv2_tpl_l16_25m_my.pth"),
    ),
    projection=dict(in_channels=1024),
)

work_dir = "exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_side_3e-4"
metr_path = "epoch_metrics_mAP.csv"

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=3e-4, weight_decay=0.05)]))

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename="data/annotations/",
    save_npz=True,
)