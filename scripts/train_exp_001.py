"""
実験A: YOLO11n トランプ検出モデルの学習 (主実験)
================================================
モデル: YOLO11n (nano) - エッジデバイス向け最軽量モデル
入力サイズ: 640x640
目的: 正面カード検出のベースラインを構築する

実行環境: CUDA対応GPU
実行方法:
    cd D:/edge-ai-card-reader
    python scripts/train_exp_001.py
    （VSCodeからの直接実行も可）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

from train_config import AUGMENTATION, DATA_YAML, TRAIN_PARAMS, download_dataset


def main():
    download_dataset()

    print("\n実験A: YOLO11n 学習開始")
    model = YOLO("yolo11n.pt")

    model.train(
        data=DATA_YAML,
        project="runs/detect",
        name="exp_001_yolo11n",
        **TRAIN_PARAMS,
        **AUGMENTATION,
    )

    print("\nテストセットで評価中...")
    best_model = YOLO("runs/detect/exp_001_yolo11n/weights/best.pt")
    metrics = best_model.val(data=DATA_YAML, split="test")

    print(f"\n{'='*40}")
    print(f"実験A: YOLO11n 結果")
    print(f"{'='*40}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
