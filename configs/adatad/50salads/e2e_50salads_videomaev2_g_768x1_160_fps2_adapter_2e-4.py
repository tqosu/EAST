_base_ = ["e2e_50salads_videomae_s_768x1_160_fps2_adapter.py"]

annotation_path = "data/annotations/"

model = dict(
    backbone=dict(
        backbone=dict(
            patch_size=14,
            embed_dims=1408,
            depth=40,
            num_heads=16,
            mlp_ratio=48 / 11,
            adapter_index=list(range(40)),
        ),
        custom=dict(pretrain="pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth"),
    ),
    projection=dict(in_channels=1408),
)

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)]))

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=0,
    end_epoch=240,
)

work_dir = "exps/50salads/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps2_adapter_2e-4"

metr_path = "epoch_metrics_mAP.csv"

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename="data/annotations/",
    save_npz=True,
)
