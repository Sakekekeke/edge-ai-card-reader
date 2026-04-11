"""
imgsz 416 vs 640 比較スクリプト
同じフレームで FPS と検出精度を比較する
"""
from ultralytics import YOLO
from picamera2 import Picamera2
import time
import cv2
import numpy as np

NUM_FRAMES = 50
WARMUP_FRAMES = 10

MODELS = [
    ("NCNN imgsz=416", "models/exp_001_416_ncnn_model", 416),
    ("NCNN imgsz=640", "models/exp_001_640_ncnn_model", 640),
]

# カメラ起動
print("[INFO] カメラ起動中...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)
print("[INFO] カメラ起動完了\n")

# まず1フレームを保存して両モデルの検出結果を比較
print("=" * 60)
print("【検出精度比較】同一フレームでの検出結果")
print("=" * 60)
frame = picam2.capture_array()
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
cv2.imwrite("compare_frame.jpg", frame_bgr)
print("比較用フレームを compare_frame.jpg に保存\n")

for name, path, imgsz in MODELS:
    print(f"--- {name} ---")
    model = YOLO(path, task="detect")
    # ウォームアップ
    for _ in range(3):
        model(frame_bgr, verbose=False, imgsz=imgsz)
    # 検出
    results = model(frame_bgr, verbose=False, imgsz=imgsz)
    for r in results:
        print(f"  検出数: {len(r.boxes)}")
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_name = model.names[int(box.cls[0])]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            print(f"    {cls_name} conf={conf:.4f} bbox={xyxy}")
    print()

# FPS比較
print("=" * 60)
print("【FPS比較】")
print("=" * 60)

results_fps = {}
for name, path, imgsz in MODELS:
    print(f"\n--- {name} ---")
    model = YOLO(path, task="detect")

    # ウォームアップ
    print(f"  ウォームアップ中...")
    for _ in range(WARMUP_FRAMES):
        f = picam2.capture_array()
        f_bgr = cv2.cvtColor(f, cv2.COLOR_RGBA2BGR)
        model(f_bgr, verbose=False, imgsz=imgsz)

    # 計測
    print(f"  FPS計測中（{NUM_FRAMES}フレーム）...")
    fps_list = []
    for _ in range(NUM_FRAMES):
        f = picam2.capture_array()
        f_bgr = cv2.cvtColor(f, cv2.COLOR_RGBA2BGR)
        start = time.time()
        model(f_bgr, verbose=False, imgsz=imgsz)
        end = time.time()
        fps_list.append(1.0 / (end - start))

    avg = np.mean(fps_list)
    std = np.std(fps_list)
    mn = np.min(fps_list)
    mx = np.max(fps_list)
    results_fps[name] = {"avg": avg, "std": std, "min": mn, "max": mx}
    print(f"  -> {avg:.1f} FPS (std: {std:.2f}, min: {mn:.1f}, max: {mx:.1f})")

picam2.stop()

# サマリー
print(f"\n{'=' * 60}")
print(f"{'モデル':<20} {'平均FPS':>8} {'最小':>6} {'最大':>6} {'標準偏差':>8}")
print(f"{'-' * 60}")
for name, m in results_fps.items():
    print(f"{name:<20} {m['avg']:>8.1f} {m['min']:>6.1f} {m['max']:>6.1f} {m['std']:>8.2f}")
print(f"{'=' * 60}")
