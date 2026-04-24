# FSSENet: A Foundation Model-Based Semantic-Structural Enhanced Network for Remote Sensing Water Body Change Detection

[cite_start]This repository contains the official implementation of the paper: "FSSENet: A Foundation Model-Based Semantic-Structural Enhanced Network for Remote Sensing Water Body Change Detection"[cite: 6, 7].

[cite_start]**Authors:** [Quanqing Ma], [Haoran Wang], [Peng Wang], [Jiaen Chen], [Qingzhan Zhao], [Xuewen Wang], [Yuchen Zheng] [cite: 8]

## 📢 News

* **[2026-04-24]** The repository is created. 
* [cite_start]**[Available Now]** All codes and implementation details are available at https://github.com/QingMal/FSSENet[cite: 21].

## 📝 Abstract

[cite_start]Water Body Change Detection (WBCD) in remote sensing focuses on automatically delineating surface water dynamic changes via bi-temporal imagery covering the same geographic area[cite: 12]. [cite_start]This task faces critical challenges, primarily brought by the complexity arising from the diverse semantic information of water bodies[cite: 13]. [cite_start]Meanwhile, existing mainstream WBCD models struggle to fully capture the complicated semantics and continuity of water bodies, especially under diverse environmental backgrounds[cite: 14].

[cite_start]To overcome the above limitations, we present a novel integrated framework, namely Foundation Model-Based Semantic-Structural Enhanced Network (FSSENet)[cite: 15]. [cite_start]Specifically, FSSENet leverages frozen foundation models to extract high-level general feature representations from bi-temporal images, utilizing their rich semantic priors to facilitate the accurate understanding of complex water body change scenarios[cite: 16]. [cite_start]For efficient semantic transfer to WBCD models, we design an Associative Semantic Adaptive Fusion (ASAF) module, which enhances water body-related region feature responses in general features and adaptively fuses them with WBCD-specific features from the customized models[cite: 17]. [cite_start]Furthermore, the Spatial Semantics and Continuity Perception (SSCP) self-attention mechanism is incorporated to improve the post-fusion features, which boosts the semantic and structural coherence of changed regions[cite: 18].

[cite_start]Comparative experiments on two WBCD benchmark datasets show that the optimal FSSENet variant reaches advanced performance over other mainstream methods[cite: 19]. [cite_start]Extensive ablation studies also verify the efficacy of core components and the robust generalization ability of the proposed FSSENet[cite: 20].

## 🚀 Framework

![FSSENet Framework](framework.png)
*(Suggestion: Place your framework.pdf or framework.png here from Figure 2 of the paper to make the repo attractive.)*

## 📂 Datasets

[cite_start]We comprehensively evaluate the proposed FSSENet through extensive experiments across two public benchmark WBCD datasets[cite: 258]:

* [cite_start]**HSRW-CD:** Consists of 2,085 satellite image pairs collected from complex environmental scenarios and diverse regions, covering multiple water body types including urban waterways, natural river systems, lacustrine zones, and man-made reservoirs[cite: 402].
* [cite_start]**Water-CD:** Encompasses China seasonal lakes distributed across the Yangtze River Basin, alongside diverse South Asia seasonal water bodies in the Jumna River region[cite: 407].

*(Suggestion: You can add download links for these datasets here once the repo is public)*
