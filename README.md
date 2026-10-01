FSSENet: A Foundation Model-Based Semantic-Structural Enhanced Network for Remote Sensing Water Body Change Detection

This repository contains the official implementation of the paper: "FSSENet: A Foundation Model-Based Semantic-Structural Enhanced Network for Remote Sensing Water Body Change Detection".

Authors: [Haoran Wang], [Quanqing Ma], [Peng Wang], [Jiaen Chen], [Qingzhan Zhao], [Xuewen Wang], [Yuchen Zheng]

📢 News

[2026-09-03] Our paper was officially published in Pattern Recognition.

[2026-08-25] Our paper was accepted for publication in Pattern Recognition.

[2026-04-24] The repository is created.

📝 Abstract

Water Body Change Detection (WBCD) in remote sensing focuses on automatically delineating surface water dynamic changes via bi-temporal imagery covering the same geographic area. This task faces critical challenges, primarily brought by the complexity arising from the diverse semantic information of water bodies. Meanwhile, existing mainstream WBCD models struggle to fully capture the complicated semantics and continuity of water bodies, especially under diverse environmental backgrounds.

To overcome the above limitations, we present a novel integrated framework, namely Foundation Model-Based Semantic-Structural Enhanced Network (FSSENet). Specifically, FSSENet leverages frozen foundation models to extract high-level general feature representations from bi-temporal images, utilizing their rich semantic priors to facilitate the accurate understanding of complex water body change scenarios. For efficient semantic transfer to WBCD models, we design an Associative Semantic Adaptive Fusion (ASAF) module, which enhances water body-related region feature responses in general features and adaptively fuses them with WBCD-specific features from the customized models. Furthermore, the Spatial Semantics and Continuity Perception (SSCP) self-attention mechanism is incorporated to improve the post-fusion features, which boosts the semantic and structural coherence of changed regions.

Comparative experiments on two WBCD benchmark datasets show that the optimal FSSENet variant reaches advanced performance over other mainstream methods. Extensive ablation studies also verify the efficacy of core components and the robust generalization ability of the proposed FSSENet.

🚀 Framework



📂 Datasets

We comprehensively evaluate the proposed FSSENet through extensive experiments across two public benchmark WBCD datasets:

HSRW-CD: Consists of 2,085 satellite image pairs collected from complex environmental scenarios and diverse regions, covering multiple water body types including urban waterways, natural river systems, lacustrine zones, and man-made reservoirs. Paper

Water-CD: Encompasses China seasonal lakes distributed across the Yangtze River Basin, alongside diverse South Asia seasonal water bodies in the Jumna River region.

📦 Pretrained Weights

The pretrained weights are available on Google Drive.

📖 Citation

If you find this work useful in your research, please cite our paper:

@article{WANG2027114829,
  title = {FSSENet: A Foundation Model-Based Semantic-Structural Enhanced Network for remote sensing water body change detection},
  journal = {Pattern Recognition},
  volume = {182},
  pages = {114829},
  year = {2027},
  doi = {https://doi.org/10.1016/j.patcog.2026.114829},
  author = {Haoran Wang and Quanqing Ma and Peng Wang and Jiaen Chen and Qingzhan Zhao and Xuewen Wang and Yuchen Zheng},
}
