# GTEA

## Results and Models (Low-Frame-Rate Action Detection Evaluation)

**ViT-L** with the pretrained weight from [SthSth V2](https://github.com/OpenGVLab/unmasked_teacher/blob/main/single_modality/MODEL_ZOO.md).

| mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |        Config         |                                                    Download                                                    |
| :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------: | :------------------------------------------------------------------------------------------------------------: |
| 95.1838 | 93.7865 | 91.1194 | 84.8779 | 77.8627 | 88.5660 | [config](e2e_gtea_ret_ssv2_tpl_l16_25m_768x1_160_adapter_2e-4.py) | [model](https://oregonstate.box.com/s/b3xf9nehc2tkc15sj913k0u6p6yyv2fv) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train EAST on GTEA split 1

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/gtea/e2e_gtea_ret_ssv2_tpl_l16_25m_768x1_160_adapter_2e-4.py --cfg-options split=split1 annotation_path=data/annotations/gtea.split1.json work_dir=exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter3_2e-4_0.0002_p0.5 feature_stride=4 workflow.end_epoch=400 optimizer.lr=0.0002 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5
```


For more details, you can refer to the Training part in the [Usage](../../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test EAST on GTEA split 1

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/gtea/e2e_gtea_ret_ssv2_tpl_l16_25m_768x1_160_adapter_2e-4.py --cfg-options split=split1 annotation_path=data/annotations/gtea.split1.json work_dir=exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter3_2e-4_0.0002_p0.5 feature_stride=4 workflow.end_epoch=400 optimizer.lr=0.0002 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 evaluation.save_npz=True --checkpoint exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter3_2e-4_0.0002_p0.5/split1/gpu2_id0/checkpoint/epoch_xxx.pth & \
```

For more details, you can refer to the Test part in the [Usage](../../../docs/en/usage.md).

## Results and Models (High-Frame-Rate Aggregation and Refinement)
We use the EAST detector to create proposals from videos for aggregator input, enabling robust segmentation.
This is Stage 2 of the segmentation-by-detection pipeline (Sec. 3.3–3.5 of the [paper](https://arxiv.org/abs/2503.06316)); the code and full instructions live in [`ms-tcn-master2/`](../../../ms-tcn-master2/README.md). It consumes the `.npz` proposals saved by the Stage-1 test command above (`evaluation.save_npz=True`).
| Data                               | F1@10 | F1@25 | F1@50 | Edit  | Acc  | Weight         | 
| :--------------------------------: | :----: | :----: | :----: | :----: | :----: | :------------: | 
| [train](https://oregonstate.box.com/s/efx1bu790n1uljzzd1rmvpzsnpw4rlcs), [evaluation](https://oregonstate.box.com/s/5z4ga87saec9t2mk0egc61xkc17rr44u) | 95.8  | 95.4  | 91.7  | 95.4  | 87.1 | [model](https://oregonstate.box.com/s/bw4qqpi5n20t4q6t22uxk2fdzv1mt8yr) |


## Train

You can use the following command to train a model.

```shell
cd ms-tcn-master2
python main.py --action train --dataset gtea --split 1 --directory_path exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter3_2e-4_0.0002_p0.5_train/split1/gpu2_id0/evaluation --directory_path_eva exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter3_2e-4_0.0002_p0.5_test/split1/gpu2_id0/evaluation --sampler uniform --lambda_val 1  --bg_idx 10 & \
```
lambda_val determines how many high-confidence proposals are excluded, while bg_idx, the background index, is omitted from F1 score calculations.

## Test
```shell
cd ms-tcn-master2
python max_FEA_avg_scores.py
```
The training script logs evaluation performance, while `max_FEA_avg_scores.py` computes the average performance across various splits.