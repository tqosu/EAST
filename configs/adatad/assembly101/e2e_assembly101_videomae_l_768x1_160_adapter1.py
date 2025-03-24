_base_ = ["e2e_assembly101_videomae_s_768x1_160_adapter.py"]

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

work_dir = "exps/assembly101/adatad/e2e_actionformer_videomae_l_768x1_160_adapter"

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=1,
    end_epoch=100,
)

metr_path = "epoch_metrics_mAP2.csv"

post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=True,
    result_dict='',
)

evaluation = dict(
    type="mAP2",
    subset="validation",
    tiou_thresholds=[0.1, 0.25, 0.5],
    ground_truth_filename=annotation_path,
    top_k=[1],
    nms_score_t=0.3,
)
