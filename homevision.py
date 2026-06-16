#!/usr/bin/env python3
"""
HomeVision - 监控视频有意义画面提取工具

从监控视频中检测运动并识别人/动物等目标，提取有意义的画面保存为截图。

使用方法:
    python homevision.py /path/to/videos              # 处理目录下所有视频
    python homevision.py /path/to/video.mov            # 处理单个视频
    python homevision.py /path/to/videos -c config.yaml  # 指定配置文件
    python homevision.py /path/to/videos --threshold 1.0  # 调高运动阈值
    python homevision.py /path/to/videos --classes person cat  # 只找人和猫
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("请先安装依赖: pip install -r requirements.txt")
    sys.exit(1)


# COCO 类别中属于动物的 ID
ANIMAL_CLASSES = {
    "person", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".wmv", ".flv"}


@dataclass
class Config:
    """运行配置"""
    # 运动检测
    sample_interval: float = 0.5
    motion_threshold: float = 0.5
    blur_kernel: int = 21
    binary_threshold: int = 25

    # 目标检测
    model_name: str = "yolov8s"
    confidence: float = 0.35
    target_classes: list = field(default_factory=lambda: list(ANIMAL_CLASSES))

    # 输出
    output_dir_name: str = "homevision_output"
    jpeg_quality: int = 95
    draw_boxes: bool = True
    dedup_interval: float = 3.0
    save_raw: bool = False

    # 性能
    workers: int = 1
    device: str = "mps"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        c = cls()
        if m := data.get("motion"):
            c.sample_interval = m.get("sample_interval", c.sample_interval)
            c.motion_threshold = m.get("threshold", c.motion_threshold)
            c.blur_kernel = m.get("blur_kernel", c.blur_kernel)
            c.binary_threshold = m.get("binary_threshold", c.binary_threshold)
        if d := data.get("detection"):
            c.model_name = d.get("model", c.model_name)
            c.confidence = d.get("confidence", c.confidence)
            c.target_classes = d.get("target_classes", c.target_classes)
        if o := data.get("output"):
            c.output_dir_name = o.get("dir_name", c.output_dir_name)
            c.jpeg_quality = o.get("jpeg_quality", c.jpeg_quality)
            c.draw_boxes = o.get("draw_boxes", c.draw_boxes)
            c.dedup_interval = o.get("dedup_interval", c.dedup_interval)
            c.save_raw = o.get("save_raw", c.save_raw)
        if p := data.get("performance"):
            c.workers = p.get("workers", c.workers)
            c.device = p.get("device", c.device)
        return c


@dataclass
class Detection:
    """单次检测结果"""
    video_file: str
    timestamp: float
    frame_number: int
    objects: list  # [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
    image_path: str = ""


class MotionDetector:
    """基于帧差法的运动检测器"""

    def __init__(self, config: Config):
        self.config = config
        self.prev_gray: Optional[np.ndarray] = None

    def reset(self):
        self.prev_gray = None

    def detect(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        检测帧中是否有运动。
        返回 (是否有运动, 运动面积比例%)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.config.blur_kernel, self.config.blur_kernel), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        # 帧差
        delta = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        # 二值化
        _, thresh = cv2.threshold(delta, self.config.binary_threshold, 255, cv2.THRESH_BINARY)

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # 计算运动面积占比
        motion_area = np.count_nonzero(thresh)
        total_area = thresh.shape[0] * thresh.shape[1]
        motion_ratio = (motion_area / total_area) * 100

        has_motion = motion_ratio > self.config.motion_threshold
        return has_motion, motion_ratio


class ObjectDetector:
    """基于 YOLOv8 的目标检测器"""

    def __init__(self, config: Config):
        self.config = config
        print(f"加载模型 {config.model_name}...")
        self.model = YOLO(f"{config.model_name}.pt")
        self.target_set = set(config.target_classes)
        print(f"模型加载完成，目标类别: {', '.join(config.target_classes)}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        检测帧中的目标。
        返回检测结果列表 [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
        """
        results = self.model(frame, conf=self.config.confidence, device=self.config.device, verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                if cls_name in self.target_set:
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    detections.append({
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": [round(x) for x in bbox],
                    })

        return detections


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """在帧上绘制检测框和标签"""
    annotated = frame.copy()

    colors = {
        "person": (0, 255, 0),      # 绿色
        "cat": (255, 165, 0),       # 橙色
        "dog": (255, 0, 0),         # 红色（BGR 所以实际是蓝色）
        "bird": (0, 255, 255),      # 黄色
    }
    default_color = (255, 255, 0)   # 青色

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        conf = det["confidence"]
        color = colors.get(cls, default_color)

        # 画框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # 标签背景
        label = f'{cls} {conf:.0%}'
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return annotated


def format_timestamp(seconds: float) -> str:
    """格式化时间戳为 HH-MM-SS 格式"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}-{minutes:02d}-{secs:02d}"


def process_video(video_path: Path, output_dir: Path, config: Config,
                  motion_detector: MotionDetector, object_detector: ObjectDetector) -> list[Detection]:
    """处理单个视频文件"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠️  无法打开: {video_path.name}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # 采样间隔对应的帧数
    frame_interval = max(1, int(fps * config.sample_interval))

    print(f"  📹 {video_path.name}: {duration:.0f}s, {fps:.0f}fps, 每{frame_interval}帧采样一次")

    motion_detector.reset()
    detections = []
    last_save_time = -config.dedup_interval  # 上次保存的时间戳
    motion_frames = 0
    analyzed_frames = 0

    # 为这个视频创建子目录
    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(exist_ok=True)

    pbar = tqdm(total=total_frames, desc=f"  处理中", unit="帧", leave=False)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        # 只在采样点分析
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        analyzed_frames += 1
        timestamp = frame_idx / fps if fps > 0 else 0

        # 第一级：运动检测
        has_motion, motion_ratio = motion_detector.detect(frame)

        if not has_motion:
            frame_idx += 1
            continue

        motion_frames += 1

        # 去重检查
        if (timestamp - last_save_time) < config.dedup_interval:
            frame_idx += 1
            continue

        # 第二级：目标检测
        objects = object_detector.detect(frame)

        if objects:
            last_save_time = timestamp
            ts_str = format_timestamp(timestamp)
            classes_str = "_".join(sorted(set(o["class"] for o in objects)))

            # 保存图片
            filename = f"{video_path.stem}_{ts_str}_{classes_str}.jpg"
            save_path = video_output_dir / filename

            if config.draw_boxes:
                annotated = draw_detections(frame, objects)
                img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            img.save(str(save_path), "JPEG", quality=config.jpeg_quality)

            # 可选保存原始帧
            if config.save_raw and config.draw_boxes:
                raw_path = video_output_dir / f"raw_{filename}"
                raw_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                raw_img.save(str(raw_path), "JPEG", quality=config.jpeg_quality)

            detection = Detection(
                video_file=video_path.name,
                timestamp=round(timestamp, 2),
                frame_number=frame_idx,
                objects=objects,
                image_path=str(save_path.relative_to(output_dir)),
            )
            detections.append(detection)

            obj_summary = ", ".join(f"{o['class']}({o['confidence']:.0%})" for o in objects)
            tqdm.write(f"    ✅ {ts_str} | {obj_summary} → {filename}")

        frame_idx += 1

    pbar.close()
    cap.release()

    print(f"  📊 分析 {analyzed_frames} 帧, 运动 {motion_frames} 帧, 保存 {len(detections)} 张截图")
    return detections


def save_report(detections: list[Detection], output_dir: Path):
    """保存检测报告"""
    if not detections:
        return

    # JSON 报告
    json_path = output_dir / "report.json"
    report = {
        "total_detections": len(detections),
        "videos_processed": len(set(d.video_file for d in detections)),
        "detections": [
            {
                "video": d.video_file,
                "timestamp": d.timestamp,
                "timestamp_formatted": format_timestamp(d.timestamp),
                "frame": d.frame_number,
                "objects": d.objects,
                "image": d.image_path,
            }
            for d in detections
        ],
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # CSV 报告
    csv_path = output_dir / "report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["视频文件", "时间戳", "帧号", "检测目标", "置信度", "截图路径"])
        for d in detections:
            for obj in d.objects:
                writer.writerow([
                    d.video_file,
                    format_timestamp(d.timestamp),
                    d.frame_number,
                    obj["class"],
                    f"{obj['confidence']:.1%}",
                    d.image_path,
                ])

    print(f"\n📋 报告已保存:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="HomeVision - 监控视频有意义画面提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python homevision.py ~/surveillance/             处理目录下所有视频
  python homevision.py ~/video.mov                 处理单个视频
  python homevision.py ~/videos/ -o ~/output/      指定输出目录
  python homevision.py ~/videos/ --threshold 1.0   降低灵敏度（过滤更多）
  python homevision.py ~/videos/ --classes person   只检测人
  python homevision.py ~/videos/ --device cpu       用 CPU 推理
        """
    )
    parser.add_argument("input", help="视频文件或目录路径")
    parser.add_argument("-c", "--config", help="配置文件路径 (YAML)")
    parser.add_argument("-o", "--output", help="输出目录路径 (默认: 输入目录同级的 homevision_output)")
    parser.add_argument("--threshold", type=float, help="运动检测阈值 (覆盖配置)")
    parser.add_argument("--confidence", type=float, help="目标检测置信度 (覆盖配置)")
    parser.add_argument("--classes", nargs="+", help="目标类别 (覆盖配置)")
    parser.add_argument("--interval", type=float, help="采样间隔秒数 (覆盖配置)")
    parser.add_argument("--device", help="推理设备: mps/cpu (覆盖配置)")
    parser.add_argument("--model", help="YOLO 模型: yolov8n/yolov8s/yolov8m (覆盖配置)")
    parser.add_argument("--no-boxes", action="store_true", help="不在截图上画检测框")
    parser.add_argument("--save-raw", action="store_true", help="同时保存无标注的原始帧")
    parser.add_argument("--dedup", type=float, help="去重间隔秒数 (覆盖配置)")

    args = parser.parse_args()

    # 加载配置
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    if Path(config_path).exists():
        config = Config.from_yaml(str(config_path))
        print(f"📄 已加载配置: {config_path}")
    else:
        config = Config()
        print("📄 使用默认配置")

    # 命令行参数覆盖配置
    if args.threshold is not None:
        config.motion_threshold = args.threshold
    if args.confidence is not None:
        config.confidence = args.confidence
    if args.classes:
        config.target_classes = args.classes
    if args.interval is not None:
        config.sample_interval = args.interval
    if args.device:
        config.device = args.device
    if args.model:
        config.model_name = args.model
    if args.no_boxes:
        config.draw_boxes = False
    if args.save_raw:
        config.save_raw = True
    if args.dedup is not None:
        config.dedup_interval = args.dedup

    # 确定输入文件列表
    input_path = Path(args.input).expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            print(f"❌ 不支持的文件格式: {input_path.suffix}")
            sys.exit(1)
        video_files = [input_path]
        base_dir = input_path.parent
    elif input_path.is_dir():
        video_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in VIDEO_EXTENSIONS and not f.name.startswith(".")
        ])
        base_dir = input_path
    else:
        print(f"❌ 路径不存在: {input_path}")
        sys.exit(1)

    if not video_files:
        print(f"❌ 未找到视频文件: {input_path}")
        sys.exit(1)

    # 确定输出目录
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = base_dir / config.output_dir_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印运行信息
    print(f"\n{'='*60}")
    print(f"🏠 HomeVision - 监控视频有意义画面提取")
    print(f"{'='*60}")
    print(f"输入: {input_path}")
    print(f"输出: {output_dir}")
    print(f"视频数量: {len(video_files)}")
    print(f"模型: {config.model_name} | 设备: {config.device}")
    print(f"采样间隔: {config.sample_interval}s | 运动阈值: {config.motion_threshold}%")
    print(f"检测置信度: {config.confidence} | 去重间隔: {config.dedup_interval}s")
    print(f"目标类别: {', '.join(config.target_classes)}")
    print(f"{'='*60}\n")

    # 初始化检测器
    motion_detector = MotionDetector(config)
    object_detector = ObjectDetector(config)

    # 处理所有视频
    all_detections = []
    start_time = time.time()

    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 处理: {video_file.name}")
        detections = process_video(video_file, output_dir, config, motion_detector, object_detector)
        all_detections.extend(detections)

    elapsed = time.time() - start_time

    # 保存报告
    save_report(all_detections, output_dir)

    # 汇总
    print(f"\n{'='*60}")
    print(f"✅ 处理完成!")
    print(f"{'='*60}")
    print(f"处理视频: {len(video_files)} 个")
    print(f"提取截图: {len(all_detections)} 张")
    print(f"耗时: {elapsed:.1f}s")
    print(f"输出目录: {output_dir}")

    if all_detections:
        # 按类别统计
        class_counts: dict[str, int] = {}
        for d in all_detections:
            for obj in d.objects:
                class_counts[obj["class"]] = class_counts.get(obj["class"], 0) + 1
        print(f"\n检测统计:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {cls}: {count} 次")


if __name__ == "__main__":
    main()
