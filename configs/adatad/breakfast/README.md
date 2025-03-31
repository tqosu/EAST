# Breakfast

## Results and Models (Low-Frame-Rate Action Detection Evaluation)

**ViT-G** with the pretrained weight from [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md).

| mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |        Config         |                                                    Download                                                    |
| :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------: | :------------------------------------------------------------------------------------------------------------: |
| 78.3859 | 76.2402 | 73.2657 | 67.2514 | 57.4398 | 70.5166 | [config](e2e_breakfast_videomaev2_g_768x1_160_fps3_adapter_2e-4.py) | [model](https://oregonstate.box.com/s/sta8m03ug13ofavpdlvbnu25uwerlfn5) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train EAST on GTEA split 1

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/breakfast/e2e_breakfast_videomaev2_g_768x1_160_fps3_adapter_2e-4.py --cfg-options split=split2 annotation_path=data/annotations/breakfast.fps3.split2.json work_dir=exps/breakfast/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps3_adapter3_2e-4_p0.5_f1 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 feature_stride=1 
```


For more details, you can refer to the Training part in the [Usage](../../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test EAST on GTEA split 1

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/breakfast/e2e_breakfast_videomaev2_g_768x1_160_fps3_adapter_2e-4.py --cfg-options split=split2 annotation_path=data/annotations/breakfast.fps3.split2.json work_dir=exps/breakfast/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps3_adapter3_2e-4_p0.5_f1 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 feature_stride=1 --checkpoint xxx.pth & \
```

For more details, you can refer to the Test part in the [Usage](../../../docs/en/usage.md).

## Results and Models (High-Frame-Rate Aggregation and Refinement)
We use the EAST detector to create proposals from videos for aggregator input, enabling robust segmentation.
| Data                               | F1@10 | F1@25 | F1@50 | Edit  | Acc  | Weight         | 
| :--------------------------------: | :----: | :----: | :----: | :----: | :----: | :------------: | 
| [train](https://oregonstate.box.com/s/i9lpnzkq7ds051zruxts1t5ojpvajpr7), [evaluation](https://oregonstate.box.com/s/az64l3a979nhy5wzctejxv4ai173kewz) | 86.2 | 82.2 | 71.8 | 84.5 | 82.8 | [model](link3) |


## Train

You can use the following command to train a model.

```shell
python main.py --action train --dataset breakfast --split 1 --directory_path exps/breakfast/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps3_adapter3_2e-4_p0.5_f1_train/split1/gpu2_id0/evaluation --directory_path_eva exps/breakfast/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps3_adapter3_2e-4_p0.5_f1_test/split1/gpu2_id0/evaluation --sampler uniform --lambda_val 7 & \
```
lambda_val determines how many high-confidence proposals are excluded.

## Test
```shell
python max_FEA_avg_scores.py
```
The training script logs evaluation performance, while `max_FEA_avg_scores.py` computes the average performance across various splits.