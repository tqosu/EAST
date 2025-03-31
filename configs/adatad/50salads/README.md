# 50salads

## Results and Models (Low-Frame-Rate Action Detection Evaluation)

**ViT-G** with the pretrained weight from [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md).

| mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |        Config         |                                                    Download                                                    |
| :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------: | :------------------------------------------------------------------------------------------------------------: |
|92.6334 | 91.1523 | 89.6745 | 87.5004 | 83.3251 | 88.8571| [config](e2e_50salads_videomaev2_g_768x1_160_fps2_adapter.py) | [model](https://oregonstate.box.com/s/c14851yhp3pibqefkfbcrqukrw0eatis) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train EAST on GTEA split 1

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/50salads/e2e_50salads_videomaev2_g_768x1_160_fps2_adapter.py --cfg-options split=split2 annotation_path=data/annotations/50salads.fps2.split2.json feature_stride=2 workflow.val_start_epoch=1 work_dir=exps/50salads/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps2_adapter3_p0.5 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 
```


For more details, you can refer to the Training part in the [Usage](../../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test EAST on GTEA split 1

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/50salads/e2e_50salads_videomaev2_g_768x1_160_fps2_adapter.py --cfg-options split=split2 annotation_path=data/annotations/50salads.fps2.split2.json feature_stride=2 workflow.val_start_epoch=1 work_dir=exps/50salads/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps2_adapter3_p0.5 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 --checkpoint xxx.pth & \
```

For more details, you can refer to the Test part in the [Usage](../../../docs/en/usage.md).

## Results and Models (High-Frame-Rate Aggregation and Refinement)
We use the EAST detector to create proposals from videos for aggregator input, enabling robust segmentation.
| Data                               | F1@10 | F1@25 | F1@50 | Edit  | Acc  | Weight         | 
| :--------------------------------: | :----: | :----: | :----: | :----: | :----: | :------------: | 
| [train](https://oregonstate.box.com/s/bvhkc5fs1wprdsdmqjiqihgb7hv642j8), [evaluation](https://oregonstate.box.com/s/7n8v0snxyy6eslexlggws87ntsga7t3i) | 93.1 | 91.9 | 88.6 | 88.4 | 91.9 | [model](link3) |


## Train

You can use the following command to train a model.

```shell
python main.py --action train --dataset 50salads --split 1 --directory_path exps/50salads/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps2_adapter3_p0.5_train/split1/gpu2_id0/evaluation/ --directory_path_eva  exps/50salads/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps2_adapter3_p0.5_test/split1/gpu2_id0/evaluation --sampler uniform --lambda_val 5
```
lambda_val determines how many high-confidence proposals are excluded.

## Test
```shell
python max_FEA_avg_scores.py
```
The training script logs evaluation performance, while `max_FEA_avg_scores.py` computes the average performance across various splits.