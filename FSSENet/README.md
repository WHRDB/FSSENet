# FSSENet

FSSENet 用于遥感双时相图像变化检测：输入配准后的两幅 RGB 图像，输出变化区域的二值掩膜。本项目基于 BAN / Open-CD，包含 ViT-L/14 视觉编码器、MiT-B0 侧分支、CA/PagFM 融合模块及 SMSA-RGAs-PCSA 注意力解码器。

本发布版整理了原项目当前训练入口使用的 **ViT-L/14 + MiT-B0、512×512、WaterCD** 实现，包含训练、评估、单对图像预测和运行检查代码。项目不附带数据、预训练权重或训练后的检查点。未接入该主配置的旧实验模块和注释中的备用代码不在本发布版中。

## 安装

验证环境为 Linux、Python 3.10、PyTorch 2.0.1、CUDA 11.7 和 MMCV 2.0.1。建议创建独立环境，并在本项目目录执行：

```bash
conda create -n fssenet python=3.10 -y
conda activate fssenet
python -m pip install numpy==1.23.5
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install mmcv==2.0.1 --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
python -m pip install -r requirements.txt
```

Open-CD 固定到上游 1.1.0 对应提交；无需修改安装目录中的 MMSegmentation 或 Open-CD。`opencd_custom/datasets/watercd.py` 提供本项目使用的数据集注册。其他 CUDA/PyTorch 组合需要匹配相应的 MMCV 二进制包。

## 数据

```text
WaterCD/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

每组图像及标签使用相同的 `.png` 文件名，A/B 尺寸一致，标签为单通道灰度图。沿用原数据处理规则：标签值小于 128 表示未变化，大于等于 128 表示变化；常用编码为 0/255。**0/1 编码的标签须将三个 dataloader 的 `dataset.format_seg_map` 设为 `None`。**

训练使用 512×512 裁剪，推理采用 512×512 滑窗。保留原配置的数据划分：训练读取 `train`，训练过程的验证和最终测试均读取 `test`。如果需要独立验证集，请修改 `val_dataloader.dataset.data_prefix`。

```bash
python tools/check_dataset.py /path/to/WaterCD
```

## 训练与测试

主配置冻结基础视觉编码器，正式训练需要两份外部初始化权重：`clip_vit-large-patch14-336_3rdparty-0b5df9cb.pth` 和 `mit_b0_20220624-7e0fe6dd.pth`。可以从 [BAN 官方预训练文件列表](https://huggingface.co/likyoo/BAN/tree/main/pretrain) 获取已转换版本。权重可放在项目外，通过参数传入：

```bash
python train.py configs/fssenet_watercd.py \
  --data-root /path/to/WaterCD \
  --pretrained /path/to/weights/clip_vit-large-patch14-336_3rdparty-0b5df9cb.pth \
  --side-pretrained /path/to/weights/mit_b0_20220624-7e0fe6dd.pth \
  --work-dir ../fssenet_runs/watercd
```

默认 batch size 为 8，训练 40,000 次迭代，每 4,000 次验证；若显存不足，可添加 `--cfg-options train_dataloader.batch_size=2`。继续训练时添加 `--resume /path/to/iter_N.pth`，或在相同输出目录下使用 `--resume` 自动读取最后一次保存记录。

评估必须提供完整的 FSSENet 检查点，无需再次提供初始化权重：

```bash
python test.py configs/fssenet_watercd.py /path/to/fssenet_checkpoint.pth \
  --data-root /path/to/WaterCD \
  --work-dir ../fssenet_runs/eval
```

可添加 `--show-dir ../fssenet_runs/visuals` 保存预测可视化。也支持 `--config` / `--checkpoint` 参数形式。`--data-root` 和 `--cfg-options data_root=...` 均会更新训练、验证和测试的数据目录；命令行传入的相对路径以当前工作目录为准，配置中的默认相对路径以项目目录为准。

## 单对图像预测

```bash
python predict.py /path/to/A.png /path/to/B.png /path/to/fssenet_checkpoint.pth \
  --output ../fssenet_runs/change_mask.png
```

输出 PNG 与输入图像尺寸相同，像素为 0 或 255。默认使用 `cuda:0`，可通过 `--device cpu` 切换设备。

## 检查与目录

```bash
python tools/smoke_test.py --device cuda:0
```

该命令使用随机参数检查完整模型的前向、损失、反向梯度和预测，不加载或保存权重，也不衡量精度。主模型默认冻结基础编码器；正式训练不能把缺少预训练权重的随机编码器当作实验复现。

- `configs/fssenet_watercd.py`：完整主配置。
- `opencd_custom/`：模型、数据集及入口共用逻辑。
- `train.py`、`test.py`、`predict.py`：训练、评估、预测。
- `scripts/`：上述入口的 Bash 包装脚本，使用 `bash scripts/train.sh ...` 等命令。
- `tools/`：数据检查与模型运行检查。
- `docs/VALIDATION.md`：发布前检查的范围和结果。

本项目沿用原项目的 Apache-2.0 许可证，见 `LICENSE`；上游来源和 BAN 引用见 [NOTICE.md](NOTICE.md)。FSSENet 论文的正式标题、作者和 DOI 请以作者发布的文献信息为准。
