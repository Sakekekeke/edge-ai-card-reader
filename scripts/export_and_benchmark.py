"""
モデルのフォーマット変換 + FPS計測
PyTorch → ONNX → NCNN の順に変換し、各形式でFPSを比較する
"""
from ultralytics import YOLO
from picamera2 import Picamera2
import time
import cv2
import numpy as np

NUM_FRAMES = 50
WARMUP_FRAMES = 10

# カメラ初期化
print("カメラ起動中...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)
print("カメラ起動完了\n")

def measure_fps(model, name):
    """指定モデルでFPSを計測する"""
    print(f"  ウォームアップ中...")
    for _ in range(WARMUP_FRAMES):
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        model(frame_bgr, verbose=False)

    print(f"  FPS計測中（{NUM_FRAMES}フレーム）...")
    fps_list = []
    for i in range(NUM_FRAMES):
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        start = time.time()
        model(frame_bgr, verbose=False)
        end = time.time()
        fps_list.append(1.0 / (end - start))

    avg = np.mean(fps_list)
    print(f"  → {name}: {avg:.1f} FPS\n")
    return avg

results = {}

# === YOLO11n ===
print("="*50)
print("YOLO11n フォーマット比較")
print("="*50)

# PyTorch
print("\n[1/3] PyTorch形式")
model_pt = YOLO("models/exp_001_best.pt")
results["YOLO11n PyTorch"] = measure_fps(model_pt, "PyTorch")

# ONNX変換
print("[2/3] ONNX形式にエクスポート中...")
onnx_path = model_pt.export(format="onnx", imgsz=640)
model_onnx = YOLO(onnx_path)
results["YOLO11n ONNX"] = measure_fps(model_onnx, "ONNX")

# NCNN変換
print("[3/3] NCNN形式にエクスポート中...")
ncnn_path = model_pt.export(format="ncnn", imgsz=640)
model_ncnn = YOLO(ncnn_path)
results["YOLO11n NCNN"] = measure_fps(model_ncnn, "NCNN")

# === YOLO26n ===
print("="*50)
print("YOLO26n フォーマット比較")
print("="*50)

# PyTorch
print("\n[1/3] PyTorch形式")
model_pt2 = YOLO("models/exp_002_best.pt")
results["YOLO26n PyTorch"] = measure_fps(model_pt2, "PyTorch")

# ONNX変換
print("[2/3] ONNX形式にエクスポート中...")
onnx_path2 = model_pt2.export(format="onnx", imgsz=640)
model_onnx2 = YOLO(onnx_path2)
results["YOLO26n ONNX"] = measure_fps(model_onnx2, "ONNX")

# NCNN変換
print("[3/3] NCNN形式にエクスポート中...")
ncnn_path2 = model_pt2.export(format="ncnn", imgsz=640)
model_ncnn2 = YOLO(ncnn_path2)
results["YOLO26n NCNN"] = measure_fps(model_ncnn2, "NCNN")

picam2.stop()

# 結果一覧
print(f"\n{'='*50}")
print(f"{'モデル + 形式':<25} {'FPS':>8} {'速度倍率':>10}")
print(f"{'-'*50}")
baseline = results["YOLO11n PyTorch"]
for name, fps in results.items():
    ratio = fps / baseline
    print(f"{name:<25} {fps:>8.1f} {ratio:>9.1f}x")
print(f"{'='*50}")
print(f"\n目標: 30 FPS")
