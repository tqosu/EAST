_base_ = ["e2e_50salads_videomae_s_768x1_160_fps2_adapter.py"]

annotation_path = "data/annotations/"

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            adapter_index=list(range(24)),
        ),
        custom=dict(pretrain="pretrained/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth"),
    ),
    projection=dict(in_channels=1024),
)

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)]))

work_dir = "exps/50salads/adatad/e2e_actionformer_videomae_l_768x1_160_adapter"

workflow = dict(
    logging_interval=50,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=1,
    end_epoch=200,
)

metr_path = "epoch_metrics_mAP2.csv"

evaluation = dict(
    type="mAP2",
    subset="validation",
    tiou_thresholds=[0.1, 0.25, 0.5],
    ground_truth_filename=annotation_path,
    top_k=[1],
    nms_score_t=0.35,
    bg_class=['action_start','action_end']
)
