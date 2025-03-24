_base_ = ["e2e_assembly101_videomae_s_768x1_160_adapter.py"]

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

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)]))



workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=1,
    end_epoch=100,
)

metr_path = "epoch_metrics_mAP.csv"
work_dir = "exps/assembly101/adatad/e2e_actionformer_videomaev2_g_768x1_160_adapter"

# evaluation = dict(
#     type="mAP",
#     subset="validation",
#     tiou_thresholds=[0.1, 0.25, 0.5, 0.6, 0.7],
#     ground_truth_filename=annotation_path,
# )
evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename="data/annotations/",
    save_npz=False,
)
