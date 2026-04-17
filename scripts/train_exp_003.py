"""
実験003: shear/perspective Augmentation有効化
=============================================
モデル: YOLO11n (nano) - exp_001と同一
入力サイズ: 640x640
目的: shear/perspectiveを有効化し、斜め角度への耐性改善を検証する

変更点（exp_001比）:
  - shear: 0.0 → 15.0（±15度のせん断変形）
  - perspective: 0.0 → 0.005（台形変形）
  - epochs: 50 → 100（Augmentation強化に伴い収束に時間がかかるため）
  - その他のパラメータはexp_001と同一（patience=15で早期終了あり）

実行環境: CUDA対応GPU
実行方法:
    cd D:/edge-ai-card-reader
    python scripts/train_exp_003.py
    （VSCodeからの直接実行も可）

所要時間目安: 2〜4時間（RTX 4060 Ti、最大100エポック。patience=15で早期終了の可能性あり）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

from train_config import AUGMENTATION, DATA_YAML, TRAIN_PARAMS, download_dataset


def main():
    download_dataset()

    # exp_001との差分: shear/perspectiveを有効化
    aug = AUGMENTATION.copy()
    aug["shear"] = 15.0        # ±15度のせん断変形（斜め角度への耐性）
    aug["perspective"] = 0.005  # 台形変形（透視変換への耐性）

    # Augmentation強化に伴いエポック数を増加（patience=15で早期終了あり）
    params = TRAIN_PARAMS.copy()
    params["epochs"] = 100

    print("\n実験003: YOLO11n + shear/perspective 学習開始")
    print(f"  変更点: shear={aug['shear']}, perspective={aug['perspective']}, epochs={params['epochs']}")
    model = YOLO("yolo11n.pt")

    model.train(
        data=DATA_YAML,
        project="runs/detect",
        name="exp_003_shear_perspective",
        **params,
        **aug,
    )

    print("\nテストセットで評価中...")
    best_model = YOLO("runs/detect/exp_003_shear_perspective/weights/best.pt")
    metrics = best_model.val(data=DATA_YAML, split="test")

    print(f"\n{'='*40}")
    print(f"実験003: YOLO11n + shear/perspective 結果")
    print(f"{'='*40}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
