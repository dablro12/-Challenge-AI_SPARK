# HUFS_BME_AI_SPARK_CHALLENGE_2403

> Challenge

Name : 제6회 2024 연구개발특구 AI SPARK 챌린지

Link : https://aifactory.space/task/2723/overview


> Model Testing 

| Version | Model            | mIOU  | Feature                 | Test epoch | Train | Pretrained |
|---------|------------------|-------|-------------------------|------------|-------|------------|
| v1      | unet (pt_brain) |       | baseline                | ✅          | ✅     |
| v2 > 0.25| unet (pt_brain)| 0.864 | v1에서 tr/vd : 8:2 → 9:1| ✅          | ✅     |
| v2 of full | unet (pt_brain)|      | v2에서 inference시 전체 확률| ✅          | ✅     |
| v3      | Attention U-Net | 0.843 | Model 아키텍쳐 변경       | 29         | ✅     |            |
| v4      | R2Att U-Net     |       | Model 아키텍쳐 변경       |            | ☑️     |            |
| v5      | UNet++           |       | Model 아키텍쳐 변경       | 29         | ✅     | ✅         |
| v6      | manet            |       | Model 아키텍쳐 변경       |            | ✅     |            |



> Reference

v1~v2 : https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/

v3~v4 : https://github.com/LeeJunHyun/Image_Segmentation/tree/master

v5~v6 : https://github.com/qubvel/segmentation_models.pytorch