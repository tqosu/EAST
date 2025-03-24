_base_ = ["e2e_assembly101_videomae_s_768x1_160_frozen.py"]

annotation_path = "data/annotations/"

model = dict(
    backbone=dict(
        backbone=dict(patch_size=14, embed_dims=1408, depth=40, num_heads=16, mlp_ratio=48 / 11),
        custom=dict(pretrain="pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth"),
    ),
    projection=dict(in_channels=1408),
)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=1,
    end_epoch=100,
)

work_dir = "exps/assembly101/adatad/e2e_actionformer_videomaev2_g_768x1_160_frozen"

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
