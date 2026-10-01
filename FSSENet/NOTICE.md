# 来源与引用

本目录从作者已有的 FSSENet 实验代码整理而来，沿用原目录的 `LICENSE`（Apache License 2.0）。整理过程中保留当前主模型的参数命名与计算逻辑，移除未接入主配置的实验分支、注释中的代码及运行产物，增加可移植入口、数据集注册与检查工具。

基础框架和相关实现来自：

- [BAN](https://github.com/likyoo/BAN)：双时相适配器、分割器与解码器基础实现。Apache-2.0。
- [Open-CD](https://github.com/likyoo/open-cd)：变化检测数据处理、训练与可视化框架。Copyright (c) Open-CD. All rights reserved. Apache-2.0。
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)、[MMEngine](https://github.com/open-mmlab/mmengine)、[MMCV](https://github.com/open-mmlab/mmcv)：骨干网络、训练运行器和基础模块。Copyright (c) OpenMMLab. All rights reserved. Apache-2.0。

CA、PagFM 与 SMSA-RGAs-PCSA 模块保留自原作者提供的项目实现。运行环境中的其他依赖、数据与下载的权重分别遵循其自身许可证。

使用 BAN 框架时请引用：

```bibtex
@article{Li2024BAN,
  author = {Li, Kaiyu and Cao, Xiangyong and Meng, Deyu},
  title = {A New Learning Paradigm for Foundation Model-based Remote Sensing Change Detection},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year = {2024},
  doi = {10.1109/TGRS.2024.3365825}
}
```

FSSENet 自身的正式文献信息尚未写入本目录，因此这里不代拟论文题名、作者或 DOI。
