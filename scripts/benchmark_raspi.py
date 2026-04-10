"""
ラズパイ5上でのFPS計測スクリプト
PyTorch / ONNX / NCNN形式でFPSを比較する

実行環境: Raspberry Pi 5 (8GB)
実行方法:
    cd ~/edge-ai-card-reader
    python benchmark_raspi.py
"""
from ultralytics import YOLO
from picamera2 import Picamera2
import time
import cv2
import numpy as np

MODELS = [
    ("exp_001: YOLO11n", "models/exp_001_best.pt"),
    ("exp_002: YOLO26n", "models/exp_002_best.pt"),
]
NUM_FRAMES = 50
WARMUP_FRAMES = 10


def measure_fps(model, name, imgsz=640):
    """指定モデルでFPSを計測する"""
    print(f"  ウォームアップ中...")
    for _ in range(WARMUP_FRAMES):
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        model(frame_bgr, verbose=False, imgsz=imgsz)

    print(f"  FPS計測中（{NUM_FRAMES}フレーム）...")
    fps_list = []
    for _ in range(NUM_FRAMES):
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        start = time.time()
        model(frame_bgr, verbose=False, imgsz=imgsz)
        end = time.time()
        fps_list.append(1.0 / (end - start))

    avg = np.mean(fps_list)
    std = np.std(fps_list)
    print(f"  -> {name}: {avg:.1f} FPS (std: {std:.2f})\n")
    return {"avg": avg, "min": np.min(fps_list), "max": np.max(fps_list), "std": std}


# カメラ初期化
print("カメラ起動中...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)
print("カメラ起動完了\n")

results = {}
for name, path in MODELS:
    print(f"{'='*50}")
    print(f"{name} (PyTorch)")
    print(f"{'='*50}")
    model = YOLO(path)
    results[f"{name} PyTorch"] = measure_fps(model, "PyTorch")

picam2.stop()

print(f"\n{'='*60}")
print(f"{'モデル':<25} {'平均FPS':>8} {'最小':>6} {'最大':>6} {'標準偏差':>8}")
print(f"{'-'*60}")
for name, m in results.items():
    print(f"{name:<25} {m['avg']:>8.1f} {m['min']:>6.1f} {m['max']:>6.1f} {m['std']:>8.2f}")
print(f"{'='*60}")