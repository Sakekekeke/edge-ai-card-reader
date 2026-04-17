"""
テスト画像分析スクリプト（ラズパイ用）

フェーズ3: 撮影したテスト画像に対してNCNNモデルで推論し、
検出精度の弱点を特定する。結果は実験ごとに分けて保存。

出力:
  - test_images/results_{実験名}/ : バウンディングボックス付き画像
  - test_images/report_{実験名}.csv : 画像ごとの検出サマリー
  - ターミナルに全体サマリー表示

使い方:
  cd ~/edge-ai-card-reader
  source venv/bin/activate
  python scripts/analyze_test_images.py exp_001              # exp_001モデルで分析
  python scripts/analyze_test_images.py exp_003              # exp_003モデルで分析
  python scripts/analyze_test_images.py                      # 引数なしで使い方を表示

モデルパスの規則: models/{実験名}_best_ncnn_model
"""

import csv
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# ============================================================
# 設定
# ============================================================
IMGSZ = 416
CONF_THRESHOLD = 0.25          # この値以上を「検出」とカウント
HIGH_CONF_THRESHOLD = 0.7      # この値以上を「高確信検出」とカウント
TEST_DIR = "test_images"

# ============================================================
# コマンドライン引数
# ============================================================
if len(sys.argv) < 2:
    print("使い方: python scripts/analyze_test_images.py <実験名>")
    print()
    print("例:")
    print("  python scripts/analyze_test_images.py exp_001")
    print("  python scripts/analyze_test_images.py exp_003")
    print()
    print("モデルパス: models/<実験名>_best_ncnn_model")
    sys.exit(1)

EXP_NAME = sys.argv[1]
MODEL_PATH = f"models/{EXP_NAME}_best_ncnn_model"
RESULT_DIR = f"test_images/results_{EXP_NAME}"
REPORT_PATH = f"test_images/report_{EXP_NAME}.csv"

# ============================================================
# メイン処理
# ============================================================
def main():
    # モデル存在確認
    if not Path(MODEL_PATH).exists():
        print(f"エラー: モデルが見つかりません: {MODEL_PATH}")
        print(f"  ラズパイの ~/edge-ai-card-reader/{MODEL_PATH} を確認してください")
        sys.exit(1)

    # モデル読み込み
    print(f"実験: {EXP_NAME}")
    print(f"モデル: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")

    # テスト画像一覧
    test_path = Path(TEST_DIR)
    images = sorted(test_path.glob("capture_*.jpg"))

    if not images:
        print(f"エラー: {TEST_DIR}/ に capture_*.jpg が見つかりません")
        return

    print(f"テスト画像: {len(images)} 枚")
    print(f"conf閾値: {CONF_THRESHOLD} / 高確信閾値: {HIGH_CONF_THRESHOLD}")
    print(f"結果保存先: {RESULT_DIR}/")
    print("-" * 60)

    # 結果フォルダ作成
    os.makedirs(RESULT_DIR, exist_ok=True)

    # CSVレポート準備
    report_rows = []

    total_detections = 0
    total_high_conf = 0
    total_low_conf = 0

    for img_path in images:
        # 推論
        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        # 検出情報を抽出
        num_detections = len(boxes)
        confs = boxes.conf.cpu().tolist() if num_detections > 0 else []
        classes = [result.names[int(c)] for c in boxes.cls.cpu().tolist()] if num_detections > 0 else []

        high_conf = [c for c in confs if c >= HIGH_CONF_THRESHOLD]
        low_conf = [c for c in confs if c < HIGH_CONF_THRESHOLD]

        avg_conf = sum(confs) / len(confs) if confs else 0
        min_conf = min(confs) if confs else 0

        total_detections += num_detections
        total_high_conf += len(high_conf)
        total_low_conf += len(low_conf)

        # 検出カード一覧
        card_list = ", ".join(
            f"{cls}({conf:.2f})"
            for cls, conf in sorted(zip(classes, confs), key=lambda x: -x[1])
        )

        # ターミナル出力
        status = "✓" if num_detections > 0 else "✗"
        print(f"{status} {img_path.name}: {num_detections}枚検出 | 平均conf={avg_conf:.2f} | {card_list}")

        # バウンディングボックス付き画像を保存
        annotated = result.plot()
        result_filename = f"result_{img_path.name}"
        cv2.imwrite(os.path.join(RESULT_DIR, result_filename), annotated)

        # レポート行追加
        report_rows.append({
            "filename": img_path.name,
            "num_detections": num_detections,
            "high_conf_count": len(high_conf),
            "low_conf_count": len(low_conf),
            "avg_conf": round(avg_conf, 3),
            "min_conf": round(min_conf, 3),
            "detected_cards": card_list,
        })

    # CSVレポート出力
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    # 全体サマリー
    print("=" * 60)
    print(f"全体サマリー [{EXP_NAME}]")
    print("=" * 60)
    print(f"テスト画像数: {len(images)}")
    print(f"総検出数: {total_detections}")
    print(f"  高確信 (conf>={HIGH_CONF_THRESHOLD}): {total_high_conf}")
    print(f"  低確信 (conf<{HIGH_CONF_THRESHOLD}): {total_low_conf}")
    print(f"検出0枚の画像: {sum(1 for r in report_rows if r['num_detections'] == 0)}")
    print(f"\nレポート: {REPORT_PATH}")
    print(f"結果画像: {RESULT_DIR}/")
    print(f"\nPCに転送:")
    print(f"  scp -r fujiwara@ml-dev-pi.local:~/edge-ai-card-reader/{RESULT_DIR} .")


if __name__ == "__main__":
    main()
