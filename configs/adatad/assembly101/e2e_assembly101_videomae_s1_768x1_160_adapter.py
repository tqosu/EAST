_base_ = ["e2e_assembly101_videomae_s_768x1_160_adapter.py"]

annotation_path = "data/annotations/"

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)]))

work_dir = "exps/assembly101/adatad/e2e_actionformer_videomae_s_768x1_160_adapter"

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=1,
    end_epoch=100,
)

metr_path = "epoch_metrics_mAP.csv"


evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.1, 0.25, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)
