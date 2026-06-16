# HomeVision 🏠

监控视频有意义画面提取工具。从大量监控视频中自动提取有人、动物或其他活动的画面。

## 工作原理

```
原始视频 → 运动检测（帧差法） → 目标识别（YOLOv8） → 保存截图 + 报告
```

1. **运动检测**：通过帧间差分快速过滤静止画面（>90% 的帧被跳过）
2. **目标识别**：对有运动的帧用 YOLO 检测人/动物等目标，过滤掉光线变化等误报
3. **智能去重**：同一场景连续检测只保留一张代表帧

## 安装

```bash
# Python 3.10+
pip install -r requirements.txt
```

## 快速开始

```bash
# 处理一个目录下所有视频
python homevision.py ~/surveillance/

# 处理单个视频
python homevision.py ~/surveillance/video.mov

# 指定输出目录
python homevision.py ~/surveillance/ -o ~/output/
```

## 常用选项

```bash
# 降低灵敏度（减少误报）
python homevision.py ~/videos/ --threshold 1.0

# 提高灵敏度（捕捉小动物）
python homevision.py ~/videos/ --threshold 0.2

# 只检测人
python homevision.py ~/videos/ --classes person

# 只检测猫和狗
python homevision.py ~/videos/ --classes cat dog

# 用更准的模型（稍慢）
python homevision.py ~/videos/ --model yolov8m

# 用 CPU 推理（无 GPU 时）
python homevision.py ~/videos/ --device cpu

# 不画检测框
python homevision.py ~/videos/ --no-boxes

# 同时保存原始未标注帧
python homevision.py ~/videos/ --save-raw
```

## 配置文件

编辑 `config.yaml` 自定义默认行为，命令行参数会覆盖配置文件。

## 输出结构

```
homevision_output/
├── video1/
│   ├── video1_00-00-05_person.jpg
│   ├── video1_00-01-12_cat.jpg
│   └── ...
├── video2/
│   └── ...
├── report.json     # 结构化报告
└── report.csv      # 表格报告
```

## Apple Silicon 优化

在 M4 Max 上默认使用 MPS (Metal Performance Shaders) 加速 YOLO 推理。
如果遇到 MPS 兼容问题，添加 `--device cpu` 回退到 CPU。

## 支持的视频格式

MOV, MP4, AVI, MKV, M4V, WMV, FLV
