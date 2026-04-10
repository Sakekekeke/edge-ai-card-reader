"""
学習の共通設定
=============
全実験で共有するパラメータを一元管理する。
各実験スクリプトではこの設定を読み込み、変更点のみ上書きする。
"""

import os
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    Roboflow = None  # ローカル実行時はデータセット手動配置のため不要

# ============================================================
# データセット設定
# ============================================================
# Roboflow play_cards_standard v2 (raw版, 10,100枚)
# ダウンロード元: https://universe.roboflow.com/augmented-startups-private/play_cards_standard/dataset/2
ROBOFLOW_WORKSPACE = "augmented-startups-private"
ROBOFLOW_PROJECT = "play_cards_standard"
ROBOFLOW_VERSION = 2
DOWNLOAD_FORMAT = "yolov11"

# APIキーは環境変数から取得
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

# ============================================================
# パス設定
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_YAML = str(DATASET_DIR / "data.yaml")

# ============================================================
# 学習 基本設定
# ============================================================
TRAIN_PARAMS = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 32,
    "device": 0,               # GPU使用
    "workers": 4,
    "seed": 42,                # 再現性のため固定
    "patience": 15,            # 15エポック改善なしで早期終了
    "optimizer": "SGD",        # fine-tuningではAdamWより汎化性能が高い傾向
    "lr0": 0.01,               # YOLO11推奨デフォルト値
    "lrf": 0.1,                # cosineスケジューラで最終学習率 = lr0 * lrf
    "warmup_epochs": 3,        # 学習初期の不安定さを防ぐ
    "weight_decay": 0.0005,
    "verbose": True,
}

# ============================================================
# Augmentation設定
# ============================================================
# 実験Aの方針: 正面カード検出に特化
# 斜め・手持ちカードへの対応はフェーズ3で自前データ追加により改善する
AUGMENTATION = {
    # --- 有効 ---
    "hsv_h": 0.015,    # 照明の色温度変化に対応。微小だがコスト低
    "hsv_s": 0.7,      # トランプデザインの彩度差・色褪せに対応
    "hsv_v": 0.4,      # 暗い部屋・影の影響に対応
    "degrees": 10.0,   # テーブル上の軽い傾きのみ。手持ちの大角度はフェーズ3で対応
    "translate": 0.1,  # 画面端のカード検出。過学習防止
    "scale": 0.5,      # カメラ距離の変動に対応 (50%~150%)
    "mosaic": 1.0,     # 複数カード同時検出の強化。BJ場面に有効

    # --- 無効 (根拠付き) ---
    "fliplr": 0.0,       # 数字・スートが左右非対称のため
    "flipud": 0.0,       # カードの上下に意味があるため (6と9の区別等)
    "shear": 0.0,        # 斜めカードはフェーズ3で実データ対応。正面特化
    "perspective": 0.0,  # 透視歪みはフェーズ3で実データ対応。正面特化
    "mixup": 0.0,        # カードの白面がぼやけ境界が不明瞭になるため
}


# ============================================================
# ユーティリティ
# ============================================================
def download_dataset():
    """データセットをダウンロードする。既に存在すればスキップ。"""
    if DATASET_DIR.exists():
        print(f"データセット既存: {DATASET_DIR}")
        return str(DATASET_DIR)

    if not ROBOFLOW_API_KEY:
        raise ValueError(
            "環境変数 ROBOFLOW_API_KEY が設定されていません。\n"
            "  export ROBOFLOW_API_KEY='your_key_here' を実行してください。"
        )

    print("データセットをダウンロード中...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(DOWNLOAD_FORMAT)
    print(f"保存先: {dataset.location}")
    return dataset.location