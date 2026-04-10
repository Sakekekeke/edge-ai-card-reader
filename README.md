# Edge AI Card Reader

ラズパイ5 + カメラでトランプをリアルタイム検出し、カードゲームの最適手を提示するエッジAIシステム。

## プロジェクト概要

既存のトランプ検出モデルを検証し、エッジデバイス（Raspberry Pi 5）上でのリアルタイム動作に向けた最適化を行うプロジェクトです。モデル選定、学習、フォーマット変換による高速化までの一連のMLOpsパイプラインを構築しました。

## 実験結果

### モデル精度（テストセット）

| 実験 | モデル | imgsz | mAP50 | mAP50-95 | Precision | Recall |
|------|--------|-------|-------|----------|-----------|--------|
| exp_001 | YOLO11n | 640 | 0.9950 | 0.9766 | 0.9985 | 0.9995 |
| exp_002 | YOLO26n | 640 | 0.9945 | 0.9675 | 0.9861 | 0.9905 |

### ラズパイ5上の推論速度

| モデル + 形式 | FPS | ベースライン比 |
|---------------|-----|---------------|
| YOLO11n PyTorch | 3.5 | 1.0x |
| YOLO11n ONNX | 6.5 | 1.9x |
| YOLO11n NCNN | 31.1 | 8.9x |
| YOLO26n PyTorch | 3.7 | 1.1x |
| YOLO26n ONNX | 7.4 | 2.1x |

PyTorch形式の3.5FPSからNCNN変換で31.1FPSへ、約9倍の高速化を達成しました。

### モデル選定の結論

YOLO11nを採用しました。理由は以下の通りです。

- YOLO26nより精度が高い（mAP50-95: 0.977 vs 0.968）
- NCNN変換に対応（YOLO26nは非対応）
- NCNN形式で30FPS以上を達成

## 技術スタック

- **ハードウェア**: Raspberry Pi 5 (8GB) + Camera Module 3 (IMX708)
- **モデル**: YOLO11n (Ultralytics)
- **データセット**: Roboflow play_cards_standard v2 (10,100枚, 52クラス)
- **学習環境**: RTX 4060 Ti (ローカル) / Kaggle GPU T4
- **推論最適化**: PyTorch → ONNX → NCNN

## プロジェクト構成

```
edge-ai-card-reader/
├── scripts/
│   ├── train_config.py          # 学習の共通設定（パラメータ・Augmentation）
│   ├── train_exp_001.py         # 実験001: YOLO11n (imgsz=640)
│   ├── train_exp_002.py         # 実験002: YOLO26n (imgsz=640)
│   └── benchmark_raspi.py       # ラズパイFPS計測スクリプト
├── docs/
│   └── YOLO_OUTPUT_GUIDE.md     # YOLO出力ファイル解説マニュアル
├── check_augmentation.py        # Augmentation可視化スクリプト
├── .gitignore
├── LICENSE
└── README.md
```

## データセット

Roboflow Universe からダウンロードできます。

- [play_cards_standard v2 (raw)](https://universe.roboflow.com/augmented-startups-private/play_cards_standard/dataset/2)
- 10,100枚、52クラス（スート4種 × 数字13種）
- 前処理・Augmentationなしのオリジナル画像

`dataset/` フォルダに以下の構成で配置してください。

```
dataset/
├── train/images/
├── train/labels/
├── valid/images/
├── valid/labels/
├── test/images/
├── test/labels/
└── data.yaml
```

## Augmentation設計

正面カード検出に特化した設計としました。斜め・手持ちカードへの対応はフェーズ3（自前データ追加）で改善予定です。

**有効化**
- `hsv_h=0.015` : 照明の色温度変化に対応
- `hsv_s=0.7` : トランプデザインの彩度差・色褪せに対応
- `hsv_v=0.4` : 暗い部屋・影の影響に対応
- `degrees=10.0` : テーブル上の軽い傾きに対応
- `translate=0.1` : 画面端のカード検出・過学習防止
- `scale=0.5` : カメラ距離の変動に対応
- `mosaic=1.0` : 複数カード同時検出の強化

**無効化（根拠付き）**
- `fliplr=0.0` : 数字・スートが左右非対称のため
- `flipud=0.0` : カードの上下に意味があるため（6と9の区別等）
- `shear=0.0` : 斜めカードはフェーズ3で実データ対応
- `perspective=0.0` : 透視歪みはフェーズ3で実データ対応
- `mixup=0.0` : カードの白面がぼやけ境界が不明瞭になるため

## 学習設定

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| epochs | 50 | エポック40前後で収束を確認 |
| batch | 32 | RTX 4060 Ti (8GB) に最適化 |
| optimizer | SGD | fine-tuningではAdamWより汎化性能が高い傾向 |
| lr0 | 0.01 | YOLO11推奨デフォルト値 |
| lrf | 0.1 | cosineスケジューラで滑らかに減衰 |
| patience | 15 | 早期終了で過学習を防止 |
| seed | 42 | 再現性のため固定 |

## セットアップ

### 学習環境（PC）

```bash
git clone https://github.com/Sakekekeke/-edge-ai-card-reader.git
cd edge-ai-card-reader
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install ultralytics
```

### 推論環境（Raspberry Pi 5）

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-opencv libopencv-dev
cd ~/edge-ai-card-reader
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install ultralytics onnxruntime
```

## 今後の予定

- [ ] 自前データ撮影（手持ち・斜めカード）による精度改善
- [ ] ブラックジャック戦略テーブルの実装
- [ ] リアルタイムUI（HIT/STAND表示）の構築
- [ ] デモ動画の撮影

## ライセンス

MIT License
