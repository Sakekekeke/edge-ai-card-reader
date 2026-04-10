# YOLO学習 出力ファイル解説マニュアル

本ドキュメントは、YOLOの学習（`model.train()`）および評価（`model.val()`）によって自動生成される出力ファイルの一覧と、各ファイルの見方を解説するものです。

## 出力フォルダの構成

```
runs/detect/runs/exp_001_yolo11n/
├── weights/
│   ├── best.pt                        # 学習中に最も精度が良かったモデル
│   └── last.pt                        # 最終エポックのモデル
├── args.yaml                          # 実験の全ハイパーパラメータ記録
├── results.csv                        # 全エポックの数値データ
├── results.png                        # ロス・精度の推移グラフ
├── labels.jpg                         # データセットの統計情報
├── BoxF1_curve.png                    # F1スコア曲線
├── BoxPR_curve.png                    # Precision-Recall曲線
├── BoxP_curve.png                     # Precision曲線
├── BoxR_curve.png                     # Recall曲線
├── confusion_matrix.png               # 混同行列
├── confusion_matrix_normalized.png    # 正規化混同行列
├── train_batch0.jpg                   # 学習初期のサンプル画像
├── train_batch1.jpg
├── train_batch2.jpg
├── train_batch{N}.jpg                 # 学習終盤のサンプル画像
├── val_batch0_labels.jpg              # 検証画像 + 正解ラベル
├── val_batch0_pred.jpg                # 検証画像 + モデル予測
├── val_batch1_labels.jpg
├── val_batch1_pred.jpg
├── val_batch2_labels.jpg
├── val_batch2_pred.jpg
└── test_results/                      # テストセット評価結果（後述）
    ├── BoxF1_curve.png
    ├── BoxPR_curve.png
    ├── BoxP_curve.png
    ├── BoxR_curve.png
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── val_batch0_labels.jpg
    └── val_batch0_pred.jpg
```


## 各ファイルの詳細解説

---

### weights/best.pt, weights/last.pt

**取得方法**: `model.train()` の実行中に自動保存される。

**内容**:
- `best.pt`: 学習全体を通して、検証データに対するmAP50-95が最も高かったエポックのモデル。デプロイや評価にはこちらを使う。
- `last.pt`: 最終エポック終了時点のモデル。学習を途中から再開（resume）する場合に使う。

**使い方**:
```python
from ultralytics import YOLO
model = YOLO("runs/detect/runs/exp_001_yolo11n/weights/best.pt")
```

---

### args.yaml

**取得方法**: `model.train()` の開始時に自動保存される。

**内容**: この実験で使用した全てのハイパーパラメータ（epochs, batch, lr0, augmentation設定など）の記録。

**見方**: テキストエディタで開ける。実験の再現性を保証するための記録であり、「この実験はどういう設定で行ったか」を後から正確に確認できる。

---

### results.csv

**取得方法**: 各エポック終了時に自動追記される。

**内容**: 全エポックの以下の数値が記録されている。
- `train/box_loss`: バウンディングボックスの位置精度に関するロス
- `train/cls_loss`: クラス分類のロス
- `train/dfl_loss`: Distribution Focal Loss（境界の微調整ロス）
- `metrics/precision(B)`: Precision（適合率）
- `metrics/recall(B)`: Recall（再現率）
- `metrics/mAP50(B)`: IoU=50%でのmAP
- `metrics/mAP50-95(B)`: IoU=50〜95%の平均mAP
- `lr/pg0`, `lr/pg1`, `lr/pg2`: 各パラメータグループの学習率

**見方**: Excelやスプレッドシートで開いて、数値の推移を細かく分析できる。results.pngのグラフの元データ。

---

### results.png

**取得方法**: 学習完了時に自動生成される。

**内容**: results.csvのデータをグラフ化したもの。以下のサブグラフが含まれる。
- 左側6つ: 各ロス（box_loss, cls_loss, dfl_loss）の推移。右下がりなら正常に学習が進んでいる。
- 右側4つ: 精度指標（Precision, Recall, mAP50, mAP50-95）の推移。右上がりなら改善している。

**見方**:
- ロスが途中から下がらなくなったら「収束した」と判断できる。
- ロスが途中から上がり始めたら「過学習」の兆候。
- mAP50-95が最も重要な指標。この値が最大になったエポックのモデルがbest.ptとして保存される。

**ポートフォリオでの活用**: README.mdに掲載して「学習が安定して収束した」ことを示すのに最適。

---

### labels.jpg

**取得方法**: 学習開始時にデータセットを分析して自動生成される。

**内容**: データセット全体の統計情報を4つのグラフで表示。
- 左上: クラスごとのインスタンス数（各カードの出現回数）。偏りがないか確認できる。
- 右上: バウンディングボックスのサイズ分布（幅×高さ）。カードの大きさのばらつきがわかる。
- 左下: バウンディングボックスの中心位置分布。画像のどこにカードが配置されているか。
- 右下: バウンディングボックスの幅と高さの分布。

**見方**: 52クラスのインスタンス数がほぼ均等であれば、クラスの偏りによる精度低下のリスクは低い。

---

### BoxPR_curve.png (Precision-Recall曲線)

**取得方法**: 学習完了後の検証時に自動生成される。

**内容**: Precision（適合率）とRecall（再現率）のトレードオフを表す曲線。

**見方**:
- 曲線が右上の角に張り付いているほど良いモデル。
- 曲線の下の面積がmAP（Average Precision）に相当する。
- Precisionが高い = 「トランプだ」と予測したものがほぼ正しい。
- Recallが高い = 実際のトランプをほぼ全て見つけられている。

---

### BoxP_curve.png (Precision曲線)

**取得方法**: 学習完了後の検証時に自動生成される。

**内容**: 信頼度（confidence）しきい値を変化させたときのPrecisionの変化。

**見方**: 信頼度しきい値を上げるほどPrecisionは上がる（確信度が高い予測だけ残すため）。「どの信頼度で切れば誤検出を減らせるか」を判断するのに使う。

---

### BoxR_curve.png (Recall曲線)

**取得方法**: 学習完了後の検証時に自動生成される。

**内容**: 信頼度しきい値を変化させたときのRecallの変化。

**見方**: 信頼度しきい値を下げるほどRecallは上がる（より多くの検出を残すため）。「どの信頼度で切れば見逃しを減らせるか」を判断するのに使う。

---

### BoxF1_curve.png (F1スコア曲線)

**取得方法**: 学習完了後の検証時に自動生成される。

**内容**: F1スコア = 2 × (Precision × Recall) / (Precision + Recall) の曲線。PrecisionとRecallのバランスを表す。

**見方**: F1が最大になる信頼度しきい値が、PrecisionとRecallの最適なバランス点。デプロイ時の信頼度しきい値の目安になる。

---

### confusion_matrix.png

**取得方法**: 学習完了後の検証時に自動生成される。

**内容**: 52クラス × 52クラスの表。行が「正解ラベル」、列が「モデルの予測」を表す。対角線上の数値が高ければ正しく分類できている。

**見方**:
- 対角線が濃い色（数値が大きい）= 正しく識別できている。
- 対角線以外に色がついている箇所 = 誤分類。例えば6Hの行の9Hの列に数値があれば「6Hを9Hと間違えた」ことを意味する。
- どのカード同士を間違えやすいかが一目でわかる。

**ポートフォリオでの活用**: 「6と9の混同が見られた」など、具体的な課題を発見するのに使える。

---

### confusion_matrix_normalized.png

**取得方法**: confusion_matrix.pngと同時に自動生成される。

**内容**: confusion_matrix.pngと同じだが、各行を合計が1.0になるように正規化した版。

**見方**: クラスごとの画像枚数に差がある場合、通常の混同行列では枚数が多いクラスの数値が大きくなる。正規化版では割合で比較できるため、クラス間の公平な比較が可能。

---

### train_batch0/1/2.jpg

**取得方法**: 学習開始直後のバッチが自動保存される。

**内容**: 学習に実際に使われた画像。Augmentation（色変化、回転、mosaic合成など）が適用された状態で表示される。バウンディングボックスと正解ラベルも描画されている。

**見方**: 設定したAugmentationが意図通りに効いているか目視確認できる。例えばmosaic=1.0なら4枚合成された画像が見える、degrees=10.0なら軽く回転した画像がある、など。

---

### train_batch{N}.jpg（終盤のバッチ）

**取得方法**: 学習終盤のバッチが自動保存される。

**内容**: train_batch0/1/2.jpgと同様だが、学習の後半で使われた画像。close_mosaic（デフォルトでは最後の10エポック）以降はmosaicが無効化されるため、通常の1枚画像が表示される。

**見方**: 学習終盤でAugmentationがどう変化しているか確認できる。

---

### val_batch{N}_labels.jpg

**取得方法**: 学習完了後の検証時に自動生成される。

**内容**: 検証用画像に「正解ラベル（Ground Truth）」のバウンディングボックスとクラス名を描画したもの。

**見方**: 「この画像にはこのカードがここにある」という正解が確認できる。pred.jpgと並べて比較するために使う。

---

### val_batch{N}_pred.jpg

**取得方法**: 学習完了後の検証時に自動生成される。

**内容**: 同じ検証用画像に「モデルの予測結果」のバウンディングボックス、クラス名、信頼度を描画したもの。

**見方**: labels.jpgと並べて、予測が正解とどれだけ一致しているか目視確認できる。バウンディングボックスの位置ずれ、クラスの間違い、検出漏れが視覚的にわかる。

**ポートフォリオでの活用**: labels.jpgとpred.jpgを並べてREADMEに掲載し、「モデルが正確に検出できている」ことを示すのに効果的。


---

## テストセット評価結果（test_results/）

### 概要

学習中の検証（validation）は `valid/` フォルダのデータに対して行われるが、テスト評価は学習に一切関与していない `test/` フォルダのデータに対して行う。テスト結果はモデルの「本当の実力」を示す最も信頼できる指標であり、ポートフォリオで報告すべき数値はこちらである。

### 取得方法

学習完了後に `best.pt` を使って以下のコマンドで取得する。

```python
from ultralytics import YOLO

model = YOLO("runs/detect/runs/exp_001_yolo11n/weights/best.pt")
metrics = model.val(
    data="dataset/data.yaml",
    split="test",                                           # テストセットを指定
    project="runs/detect/runs/exp_001_yolo11n",             # 実験フォルダ内に保存
    name="test_results",                                    # サブフォルダ名
    exist_ok=True,                                          # 上書き許可
)

print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
```

### 検証（val）とテスト（test）の違い

| 項目 | 検証 (val) | テスト (test) |
|------|-----------|--------------|
| 使用データ | `valid/` フォルダ | `test/` フォルダ |
| 実行タイミング | 学習中に毎エポック自動実行 | 学習完了後に手動実行 |
| 学習への影響 | early stoppingやbest.pt選択に使用 | 学習に一切関与しない |
| 目的 | 学習の進捗監視・過学習検知 | モデルの最終性能評価 |
| 報告すべきか | 参考値として記載可 | ポートフォリオで報告すべき本番の数値 |

### test_results/ の出力ファイル

テスト評価で生成されるファイルは、学習時の検証で生成されるものと同じ種類だが、テストデータに対する結果である点が異なる。

**BoxF1_curve.png, BoxPR_curve.png, BoxP_curve.png, BoxR_curve.png**:
学習時の同名ファイルと同じ形式だが、テストデータに対する曲線。テストデータでも検証データと同等の曲線が得られていれば、モデルが過学習していないことの証拠になる。

**confusion_matrix.png, confusion_matrix_normalized.png**:
テストデータに対する混同行列。検証データの混同行列と比較して、特定のカードの誤分類パターンが変わっていないか確認する。テストデータ固有の弱点が見つかることもある。

**val_batch{N}_labels.jpg, val_batch{N}_pred.jpg**:
テストデータに対する正解ラベルとモデル予測の比較画像。ファイル名は `val_` で始まるが、中身はテストデータの画像である（YOLOの命名規則による）。

### テスト結果の見方と注意点

- テスト結果が検証結果よりも著しく低い場合、過学習の可能性がある。
- テスト結果と検証結果がほぼ同等であれば、モデルの汎化性能が十分であることを示す。
- テスト結果はモデル選択後に1回だけ実行する。テスト結果を見てハイパーパラメータを調整すると、テストデータに対する過学習（データリーク）になるため避けること。

### ポートフォリオでの活用

テスト結果は実験の最終報告値として使う。以下の形式でREADMEやレポートに記載する。

```
| 実験 | モデル | imgsz | mAP50 | mAP50-95 | Precision | Recall |
|------|--------|-------|-------|----------|-----------|--------|
| 001  | YOLO11n | 640  | 0.9950 | 0.9766  | 0.9985   | 0.9995 |
| 002  | YOLO26n | 640  | x.xxxx | x.xxxx  | x.xxxx   | x.xxxx |
| 003  | YOLO11n | 416  | x.xxxx | x.xxxx  | x.xxxx   | x.xxxx |
```
