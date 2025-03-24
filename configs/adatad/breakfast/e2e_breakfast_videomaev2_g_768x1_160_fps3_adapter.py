_base_ = ["e2e_breakfast_videomae_s_768x1_160_fp3_adapter.py"]

annotation_path = "data/annotations"

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

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

# optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)]))
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=50,
)

work_dir = "exps/breakfast/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps3_adapter_2e-4"

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