_base_ = ["e2e_breakfast_videomae_s_768x1_160_fp3_adapter.py"]

annotation_path = "data/annotations/"

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1280,
            depth=32,
            num_heads=16,
            adapter_index=list(range(32)),
        ),
        custom=dict(pretrain="pretrained/vit-huge-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"),
    ),
    projection=dict(in_channels=1280),
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

work_dir = "exps/breakfast/adatad/e2e_actionformer_videomae_h_768x1_160_adapter"

metr_path = "epoch_metrics_mAP.csv"

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename="data/annotations/",
    save_npz=False,
)