# Assembly101

## Results and Models (Low-Frame-Rate Action Detection Evaluation)

**ViT-G** with the pretrained weight from [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md).

| mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |        Config         |                                                    Download                                                    |
| :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------: | :------------------------------------------------------------------------------------------------------------: |
| 22.5270 | 21.7323 | 20.0149 | 17.8822 | 15.7459 | 19.5805 | [config](e2e_assembly101_videomaev2_g_768x1_160_adapter.py) | [model](https://oregonstate.box.com/s/9g8q0v0n6n8mws8gv5n7e1qtowy903h4) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train EAST on GTEA split 1

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/assembly101/e2e_assembly101_videomaev2_g_768x1_160_adapter.py --cfg-options split=split1 annotation_path=data/annotations/assembly101._fps6.json feature_stride=1 data_path=data/assembly101/Assembly104_4  workflow.logging_interval=10  workflow.val_start_epoch=100 work_dir=exps/assembly101/adatad/e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.5 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5
```


For more details, you can refer to the Training part in the [Usage](../../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test EAST on GTEA split 1

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/assembly101/e2e_assembly101_videomaev2_g_768x1_160_adapter.py --cfg-options split=split1 annotation_path=data/annotations/assembly101._fps6.json feature_stride=1 data_path=data/assembly101/Assembly104_4  workflow.logging_interval=10  workflow.val_start_epoch=100 work_dir=exps/assembly101/adatad/e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.5 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 --checkpoint xxx.pth & \
```

For more details, you can refer to the Test part in the [Usage](../../../docs/en/usage.md).

## Results and Models (High-Frame-Rate Aggregation and Refinement)
We use the EAST detector to create proposals from videos for aggregator input, enabling robust segmentation.
| Data                               | F1@10 | F1@25 | F1@50 | Edit  | Acc  | Weight         | 
| :--------------------------------: | :----: | :----: | :----: | :----: | :----: | :------------: | 
| [train](https://oregonstate.box.com/s/ronbq55v62sn550cfkzl233af7lpp863), [evaluation](https://oregonstate.box.com/s/gv5zoyao42mqq7pad2hruoonkksuxdei) |  42.0 | 39.2 | 32.5 | 39.9 | 47.9 | [model](link3) |


## Train

You can use the following command to train a model.

```shell
python main.py --action train --dataset assembly101 --split 1 --directory_path exps/assembly101/adatad/e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.5_train/split1/gpu2_id0/evaluation/ --directory_path_eva exps/assembly101/adatad/e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.5_test/split1/gpu2_id0/evaluation/  --lambda_val 1 --sampler uniform  --bz 1 
```
lambda_val determines how many high-confidence proposals are excluded.

## Test
```shell
python max_FEA_avg_scores.py
```
The training script logs evaluation performance, while `max_FEA_avg_scores.py` computes the average performance across various splits.