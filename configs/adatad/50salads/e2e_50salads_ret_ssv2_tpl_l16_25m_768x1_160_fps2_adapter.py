_base_ = ["e2e_50salads_videomae_s_768x1_160_fps2_adapter.py"]

annotation_path = "data/50salads/annotations/50salads.fps2.split1.json"

model = dict(
    backbone=dict(
        backbone=dict(
            patch_size=16,
            embed_dims=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            adapter_index=list(range(24)),
        ),
        custom=dict(pretrain="pretrained/ret_ssv2_tpl_l16_25m_my.pth"),
    ),
    projection=dict(in_channels=1024),
)

optimizer = dict(lr=1e-3, backbone=dict(custom=[dict(name="adapter", lr=1e-3, weight_decay=0.05)]))

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=0,
    end_epoch=200,
)

work_dir = "exps/50salads/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_fps2_adapter"

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.1, 0.25, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)
