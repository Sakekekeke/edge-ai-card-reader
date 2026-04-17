"""
テスト画像撮影スクリプト（ラズパイ用）

フェーズ3: 現状モデルの弱点特定のため、実環境でテスト画像を撮影する。
MJPEG配信でブラウザプレビュー＋ブラウザ上のボタンで撮影。

使い方:
  1. ラズパイ側: python scripts/capture_test_images.py
  2. PC側ブラウザ: http://ml-dev-pi.local:8081
  3. ブラウザ上の「撮影」ボタンで画像保存
  4. 撮影条件メモ（角度・照明等）を入力可能

保存先: test_images/ フォルダ（自動作成）
ファイル名: capture_001.jpg, capture_002.jpg, ...
メタ情報: test_images/capture_log.csv に撮影条件を記録
"""

import csv
import io
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template_string, request
from picamera2 import Picamera2

# ============================================================
# 設定
# ============================================================
CAMERA_SIZE = (640, 480)       # picamera2 取得解像度
JPEG_QUALITY = 90              # 撮影保存時のJPEG品質（プレビューより高品質）
PREVIEW_QUALITY = 80           # MJPEG配信のJPEG圧縮品質
SERVER_PORT = 8081             # live_stream.py(8080)と共存可能にポート変更
SAVE_DIR = "test_images"       # 保存先フォルダ（相対パス、cd ~/edge-ai-card-reader で実行前提）

# ============================================================
# グローバル変数
# ============================================================
app = Flask(__name__)
latest_frame = None            # 最新フレーム（BGR）
frame_lock = threading.Lock()
capture_count = 0              # 撮影枚数カウンター

# ============================================================
# カメラスレッド
# ============================================================
def camera_thread():
    """picamera2でフレームを取得し続けるスレッド"""
    global latest_frame

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CAMERA_SIZE, "format": "XRGB8888"}
    )
    picam2.configure(config)
    picam2.start()
    print(f"カメラ起動完了 ({CAMERA_SIZE[0]}x{CAMERA_SIZE[1]})")

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            with frame_lock:
                latest_frame = frame_bgr
            time.sleep(0.03)  # ~30FPS
    finally:
        picam2.stop()


# ============================================================
# MJPEG配信
# ============================================================
def generate_mjpeg():
    """MJPEGストリーム生成"""
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue

        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
        time.sleep(0.03)


@app.route("/stream")
def stream():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# 撮影API
# ============================================================
@app.route("/capture", methods=["POST"])
def capture():
    """現在のフレームを保存"""
    global capture_count

    with frame_lock:
        frame = latest_frame

    if frame is None:
        return jsonify({"error": "カメラ未起動"}), 500

    # 保存先フォルダ作成
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 連番ファイル名
    capture_count += 1
    filename = f"capture_{capture_count:03d}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    # 高品質で保存
    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # メモ情報を取得
    memo = request.json.get("memo", "") if request.is_json else ""

    # CSVログに記録
    log_path = os.path.join(SAVE_DIR, "capture_log.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filename", "timestamp", "memo"])
        writer.writerow([filename, datetime.now().isoformat(), memo])

    print(f"撮影: {filepath} | メモ: {memo}")
    return jsonify({"filename": filename, "count": capture_count})


# ============================================================
# 撮影枚数取得API
# ============================================================
@app.route("/status")
def status():
    """現在の撮影枚数を返す"""
    return jsonify({"count": capture_count})


# ============================================================
# ブラウザUI
# ============================================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>テスト画像撮影</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #1a1a2e;
    color: #eee;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
  }

  h1 {
    font-size: 1.4rem;
    margin-bottom: 16px;
    color: #e94560;
  }

  .stream-container {
    position: relative;
    border: 2px solid #333;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 16px;
  }

  .stream-container img {
    display: block;
    max-width: 100%;
    height: auto;
  }

  .flash {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: white;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.05s;
  }

  .flash.active {
    opacity: 0.6;
    transition: none;
  }

  .controls {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    width: 100%;
    max-width: 640px;
  }

  .memo-row {
    display: flex;
    gap: 8px;
    width: 100%;
  }

  .memo-row input {
    flex: 1;
    padding: 10px 14px;
    border: 1px solid #333;
    border-radius: 6px;
    background: #16213e;
    color: #eee;
    font-size: 0.95rem;
  }

  .memo-row input::placeholder { color: #666; }

  .btn-capture {
    padding: 14px 48px;
    font-size: 1.2rem;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    background: #e94560;
    color: white;
    transition: background 0.2s, transform 0.1s;
    user-select: none;
  }

  .btn-capture:hover { background: #c81e45; }
  .btn-capture:active { transform: scale(0.96); }

  .counter {
    font-size: 1.1rem;
    color: #aaa;
  }

  .counter span {
    color: #e94560;
    font-weight: bold;
    font-size: 1.3rem;
  }

  .tips {
    margin-top: 20px;
    padding: 16px;
    background: #16213e;
    border-radius: 8px;
    max-width: 640px;
    width: 100%;
    font-size: 0.85rem;
    color: #999;
    line-height: 1.6;
  }

  .tips h3 {
    color: #ccc;
    margin-bottom: 8px;
    font-size: 0.95rem;
  }

  .last-saved {
    font-size: 0.9rem;
    color: #4ecca3;
    min-height: 1.2em;
  }
</style>
</head>
<body>

<h1>テスト画像撮影 - フェーズ3</h1>

<div class="stream-container">
  <img id="stream" src="/stream" alt="カメラプレビュー">
  <div id="flash" class="flash"></div>
</div>

<div class="controls">
  <div class="memo-row">
    <input type="text" id="memo" placeholder="撮影メモ（例: 斜め30度、暗い照明、カード5枚）">
  </div>

  <button class="btn-capture" id="captureBtn" onclick="capture()">
    📸 撮影
  </button>

  <div class="counter">撮影枚数: <span id="count">0</span></div>
  <div class="last-saved" id="lastSaved"></div>
</div>

<div class="tips">
  <h3>撮影ガイドライン</h3>
  ・正面 / 斜め(30度) / 斜め(60度) を各条件で撮影<br>
  ・照明: 通常 / 暗め / 逆光 など変えて撮る<br>
  ・カード枚数: 1枚 / 3枚 / 5-7枚 を変える<br>
  ・背景: テーブル / 布 / 手持ち なども変える<br>
  ・重なりあるケース、端が切れるケースも撮ると良い<br>
  ・メモ欄に条件を書いておくと後で分析しやすい
</div>

<script>
async function capture() {
  const btn = document.getElementById("captureBtn");
  const memo = document.getElementById("memo").value;
  const flash = document.getElementById("flash");

  // フラッシュ演出
  flash.classList.add("active");
  setTimeout(() => flash.classList.remove("active"), 150);

  // ボタン一時無効化（連打防止）
  btn.disabled = true;
  btn.textContent = "保存中...";

  try {
    const res = await fetch("/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ memo: memo })
    });
    const data = await res.json();

    if (data.filename) {
      document.getElementById("count").textContent = data.count;
      document.getElementById("lastSaved").textContent =
        "✓ 保存: " + data.filename;
    }
  } catch (e) {
    document.getElementById("lastSaved").textContent = "⚠ エラー: " + e.message;
  }

  btn.disabled = false;
  btn.textContent = "📸 撮影";
}

// キーボードショートカット: Spaceキーでも撮影
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target.tagName !== "INPUT") {
    e.preventDefault();
    capture();
  }
});
</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    # 既存画像の枚数を確認して連番を継続
    save_path = Path(SAVE_DIR)
    if save_path.exists():
        existing = list(save_path.glob("capture_*.jpg"))
        if existing:
            nums = []
            for f in existing:
                try:
                    nums.append(int(f.stem.split("_")[1]))
                except (IndexError, ValueError):
                    pass
            if nums:
                capture_count = max(nums)
                print(f"既存画像 {len(existing)} 枚検出。連番を {capture_count + 1} から継続")

    # カメラスレッド開始
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()
    print(f"サーバー起動: http://0.0.0.0:{SERVER_PORT}")
    print(f"保存先: {SAVE_DIR}/")
    print("Ctrl+C で終了")

    app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True)
