"""
リアルタイムカード検出 MJPEG配信サーバー
ラズパイ5上でYOLO11n (NCNN) の検出結果をブラウザに配信する

実行環境: Raspberry Pi 5 (8GB)
前提:
    pip install flask

実行方法:
    cd ~/edge-ai-card-reader
    source venv/bin/activate
    python scripts/live_stream.py

ブラウザでアクセス:
    http://192.168.141.207:8080
    または http://ml-dev-pi.local:8080
"""

import time
import threading
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from ultralytics import YOLO

# ============================================================
# 設定
# ============================================================
MODEL_PATH = "models/exp_001_best_ncnn_model"
IMGSZ = 416                # NCNNモデルのエクスポート時サイズに合わせる（640不可）
CAMERA_SIZE = (640, 480)
JPEG_QUALITY = 80          # JPEG圧縮品質 (1-100)
SERVER_PORT = 8080
FPS_WINDOW = 30            # FPS計算用の直近フレーム数

# BBox描画設定
BOX_COLOR = (0, 255, 0)   # 緑 (BGR)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LABEL_BG_COLOR = (0, 255, 0)
LABEL_TEXT_COLOR = (0, 0, 0)

# ============================================================
# グローバル変数
# ============================================================
output_frame = None
frame_lock = threading.Lock()

# ============================================================
# HTMLテンプレート
# ============================================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge AI Card Reader - Live</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a0a;
            color: #e0e0e0;
            font-family: 'Courier New', monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        h1 {
            font-size: 1.4rem;
            color: #00ff88;
            margin-bottom: 8px;
            letter-spacing: 2px;
        }
        .subtitle {
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 20px;
        }
        .stream-container {
            border: 2px solid #00ff88;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.1);
        }
        img {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .info {
            margin-top: 16px;
            font-size: 0.75rem;
            color: #555;
        }
    </style>
</head>
<body>
    <h1>&#x1f0cf; EDGE AI CARD READER</h1>
    <p class="subtitle">YOLO11n NCNN &middot; Raspberry Pi 5 &middot; Live Detection</p>
    <div class="stream-container">
        <img src="/video_feed" alt="Live Stream">
    </div>
    <p class="info">Stream: {{ camera_size[0] }}x{{ camera_size[1] }} &middot; JPEG Q{{ jpeg_quality }}</p>
</body>
</html>
"""


# ============================================================
# 検出 & 描画ループ
# ============================================================
def detection_loop(model, picam2):
    """カメラからフレームを取得し、推論・描画して output_frame を更新する"""
    global output_frame
    fps_times = deque(maxlen=FPS_WINDOW)

    print("[INFO] 検出ループ開始")
    while True:
        t_start = time.time()

        # フレーム取得
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # 推論
        results = model(frame_bgr, verbose=False, imgsz=IMGSZ)

        # BBox描画
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                # 座標
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                # ラベル文字列
                label = f"{cls_name} {conf:.2f}"

                # BBox
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

                # ラベル背景
                (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), LABEL_BG_COLOR, -1)
                cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4), FONT, FONT_SCALE,
                            LABEL_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # FPS計算 & 表示
        t_end = time.time()
        fps_times.append(t_end - t_start)
        fps = len(fps_times) / sum(fps_times) if fps_times else 0
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame_bgr, fps_text, (10, 30), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # JPEG圧縮してグローバル変数に格納
        _, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with frame_lock:
            output_frame = jpeg.tobytes()


def generate_mjpeg():
    """MJPEGストリームのジェネレータ"""
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            frame_bytes = output_frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # ブラウザ側の負荷軽減（配信FPSの上限を緩やかに制限）
        time.sleep(0.01)


# ============================================================
# Flask アプリ
# ============================================================
app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE, camera_size=CAMERA_SIZE, jpeg_quality=JPEG_QUALITY)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# メイン
# ============================================================
def main():
    # カメラ起動
    print("[INFO] カメラ起動中...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": CAMERA_SIZE})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print(f"[INFO] カメラ起動完了 ({CAMERA_SIZE[0]}x{CAMERA_SIZE[1]})")

    # モデル読み込み
    print(f"[INFO] モデル読み込み中: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[INFO] モデル読み込み完了")

    # ウォームアップ（初回推論は遅いため）
    print("[INFO] ウォームアップ中...")
    for _ in range(10):
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        model(frame_bgr, verbose=False, imgsz=IMGSZ)
    print("[INFO] ウォームアップ完了")

    # 検出ループを別スレッドで起動
    det_thread = threading.Thread(target=detection_loop, args=(model, picam2), daemon=True)
    det_thread.start()

    # Flask サーバー起動
    print(f"\n{'='*50}")
    print(f"  配信開始！ブラウザでアクセス:")
    print(f"  http://192.168.141.207:{SERVER_PORT}")
    print(f"  http://ml-dev-pi.local:{SERVER_PORT}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=SERVER_PORT, threaded=True)


if __name__ == '__main__':
    main()
