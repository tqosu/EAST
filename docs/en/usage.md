## Usage

By default, we use DistributedDataParallel (DDP) both in single GPU and multiple GPU cases for simplicity.

### Low-Frame-Rate Action Detection Training

`torchrun --nnodes={num_node} --nproc_per_node={num_gpu} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py {config}`

- `num_node` is often set as 1 if all gpus are allocated in a single node. `num_gpu` is the number of used GPU.
- `config` is the path of the config file.

Example:

- GTEA split 1
```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/gtea/e2e_gtea_ret_ssv2_tpl_l16_25m_768x1_160_adapter_2e-4.py --cfg-options split=split1 annotation_path=data/annotations/gtea.split1.json work_dir=exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter3_2e-4_0.0002_p0.5 feature_stride=4 workflow.end_epoch=400 optimizer.lr=0.0002 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5
```


- 50salads split 1
```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/50salads/e2e_50salads_videomaev2_g_768x1_160_fps2_adapter.py --cfg-options split=split1 annotation_path=data/annotations/50salads.fps2.split1.json feature_stride=2 workflow.val_start_epoch=1 work_dir=exps/50salads/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps2_adapter3_p0.5 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5
```

- breakfast split 1
```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/breakfast/e2e_breakfast_videomaev2_g_768x1_160_fps3_adapter_2e-4.py --cfg-options split=split1 annotation_path=data/annotations/breakfast.fps3.split1.json work_dir=exps/breakfast/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps3_adapter3_2e-4_p0.5_f1 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 feature_stride=1
```

- Assembly101
```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/assembly101/e2e_assembly101_videomaev2_g_768x1_160_adapter.py --cfg-options split=split1 annotation_path=data/annotations/assembly101._fps6.json feature_stride=1 data_path=data/assembly101/Assembly104_4  workflow.logging_interval=10  workflow.val_start_epoch=100 work_dir=exps/assembly101/adatad/e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.6 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.6 & \
```

Note:
- By default, evaluation is also conducted in the training, based on the argument in the config file. You can disable this, or increase the evaluation interval to speed up the training. 

### Low-Frame-Rate Action Detection Evaluation


`torchrun --nnodes={num_node} --nproc_per_node={num_gpu} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py {config} --checkpoint {path}`

- if `checkpoint` is not specified, the `best.pth` in the config's result folder will be used.

Example:

- GTEA split 1
```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/gtea/e2e_gtea_ret_ssv2_tpl_l16_25m_768x1_160_adapter_2e-4.py --cfg-options split=split1 annotation_path=data/annotations/gtea.split1.json work_dir=exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter3_2e-4_0.0002_p0.5 feature_stride=4 workflow.end_epoch=400 optimizer.lr=0.0002 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 evaluation.save_npz=True --checkpoint exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter3_2e-4_0.0002_p0.5/split1/gpu2_id0/checkpoint/epoch_xxx.pth & \
```

- 50salads split 1
```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/50salads/e2e_50salads_videomaev2_g_768x1_160_fps2_adapter.py --cfg-options split=split1 annotation_path=data/annotations/50salads.fps2.split1.json feature_stride=2 workflow.val_start_epoch=1 work_dir=exps/50salads/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps2_adapter3_p0.5 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 --checkpoint exps/50salads/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps2_adapter3_p0.5/split1/gpu2_id0/checkpoint/epoch_xxx.pth
```

- breakfast split 1
```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/breakfast/e2e_breakfast_videomaev2_g_768x1_160_fps3_adapter_2e-4.py --cfg-options split=split1 annotation_path=data/annotations/breakfast.fps3.split1.json work_dir=exps/breakfast/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps3_adapter3_2e-4_p0.5_f1 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.5 feature_stride=1 --checkpoint exps/breakfast/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps3_adapter3_2e-4_p0.5_f1/split1/gpu2_id0/checkpoint/epoch_xx.pth
```
- Assembly101
```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/assembly101/e2e_assembly101_videomaev2_g_768x1_160_adapter.py --cfg-options split=split1 annotation_path=data/annotations/assembly101._fps6.json feature_stride=1 data_path=data/assembly101/Assembly104_4  workflow.logging_interval=10  workflow.val_start_epoch=100 work_dir=exps/assembly101/adatad/e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.6 model.backbone.backbone.type=VisionTransformerAdapter3 model.backbone.backbone.adapter_drop_rate=0.6 --checkpoint exps/assembly101/adatad/e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.6/split1/gpu2_id0/checkpoint/epoch_xx.pth & \
```
