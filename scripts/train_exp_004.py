"""
実験004: 自前データ追加による実環境精度改善
==========================================
モデル: YOLO11n (nano) - exp_001と同一アーキテクチャ
入力サイズ: 640x640
目的: 既存10,100枚 + 自前撮影68枚で再学習し、斜め角度での検出精度を改善する

変更点（exp_001比）:
  - データセット: 既存10,100枚 + 自前撮影68枚（手持ち・斜め角度）
  - epochs: 50 → 100（データ多様化に伴う収束時間の確保）
  - Augmentation: exp_001と同一（shear/perspectiveは無効のまま）
  - その他のパラメータはexp_001と同一

データセット構成:
  - dataset/data_exp_004.yaml を使用
  - train: 既存7,070枚 + 自前68枚 = 7,138枚
  - val: 既存2,020枚（変更なし）
  - test: 既存1,010枚（正面画像での評価用）
  - 実環境評価: dataset/custom_data/test_22/ で別途実施

実行環境: CUDA対応GPU
実行方法:
    cd D:/edge-ai-card-reader
    python scripts/train_exp_004.py
    （VSCodeからの直接実行も可）

所要時間目安: 2〜4時間（RTX 4060 Ti、最大100エポック。patience=15で早期終了の可能性あり）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

from train_config import AUGMENTATION, TRAIN_PARAMS, download_dataset

# exp_004用のdata.yaml（既存 + 自前データ統合）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_YAML_EXP004 = str(PROJECT_ROOT / "dataset" / "data_exp_004.yaml")


def main():
    download_dataset()

    # data.yaml存在確認
    if not Path(DATA_YAML_EXP004).exists():
        print(f"エラー: {DATA_YAML_EXP004} が見つかりません")
        print("dataset/data_exp_004.yaml を配置してください")
        sys.exit(1)

    # 自前データ存在確認
    custom_train = PROJECT_ROOT / "dataset" / "custom_data" / "train_68" / "images"
    if not custom_train.exists():
        print(f"エラー: {custom_train} が見つかりません")
        print("自前学習データを配置してください")
        sys.exit(1)
    custom_count = len(list(custom_train.glob("*")))
    print(f"自前学習データ: {custom_count} 枚")

    # エポック数を増加（データ多様化に伴う収束時間の確保）
    params = TRAIN_PARAMS.copy()
    params["epochs"] = 100

    print(f"\n実験004: YOLO11n + 自前データ追加 学習開始")
    print(f"  データ: 既存 + 自前{custom_count}枚")
    print(f"  data.yaml: {DATA_YAML_EXP004}")
    print(f"  epochs: {params['epochs']}")
    model = YOLO("yolo11n.pt")

    model.train(
        data=DATA_YAML_EXP004,
        project="runs/detect",
        name="exp_004_custom_data",
        **params,
        **AUGMENTATION,
    )

    print("\nテストセットで評価中（既存テスト画像1,010枚）...")
    best_model = YOLO("runs/detect/exp_004_custom_data/weights/best.pt")
    metrics = best_model.val(data=DATA_YAML_EXP004, split="test")

    print(f"\n{'='*50}")
    print(f"実験004: YOLO11n + 自前データ追加 結果")
    print(f"{'='*50}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"{'='*50}")
    print(f"\n次のステップ:")
    print(f"  1. NCNNエクスポート: model.export(format='ncnn', imgsz=416)")
    print(f"  2. ラズパイに転送して実環境テスト画像22枚で評価")


if __name__ == "__main__":
    main()
