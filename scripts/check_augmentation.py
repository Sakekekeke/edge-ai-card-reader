"""
Augmentation可視化スクリプト
============================
学習時のAugmentation効果を画像で確認する。

モード:
  - all     : 全Augmentationを一覧表示（従来の動作）
  - exp_003 : exp_003で追加したshear/perspectiveの効果を詳細比較

実行方法:
    cd D:/edge-ai-card-reader
    python scripts/check_augmentation.py all       # 全Augmentation一覧
    python scripts/check_augmentation.py exp_003   # exp_003の効果確認
    python scripts/check_augmentation.py           # 引数なしで使い方を表示
    （VSCodeからの直接実行も可）
"""

import cv2
import numpy as np
import os
import random
import sys

# --- 設定 ---
DATA_DIR = r"D:\edge-ai-card-reader\dataset\train\images"
OUTPUT_DIR = r"D:\edge-ai-card-reader\aug_preview"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

# コマンドライン引数でモード選択
VALID_MODES = ["all", "exp_003"]

if len(sys.argv) < 2 or sys.argv[1] not in VALID_MODES:
    print("使い方: python scripts/check_augmentation.py <モード>")
    print(f"  利用可能なモード: {', '.join(VALID_MODES)}")
    print()
    print("  all     : 全Augmentationのmin/max比較（従来の動作）")
    print("  exp_003 : shear/perspectiveの段階的効果確認")
    sys.exit(1)

MODE = sys.argv[1]

# サンプル画像を3枚選ぶ
all_images = [f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.png'))]
samples = random.sample(all_images, 3)
print(f"選択した画像: {samples}")
print(f"モード: {MODE}")

# ============================================================
# Augmentation関数
# ============================================================

def apply_hsv(img, h_gain, s_gain, v_gain):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = np.clip(hsv[:,:,0] + h_gain * 180, 0, 180)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + s_gain), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * (1 + v_gain), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_rotation(img, degrees):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(114,114,114))

def apply_translate(img, tx_ratio, ty_ratio):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx_ratio*w], [0, 1, ty_ratio*h]])
    return cv2.warpAffine(img, M, (w, h), borderValue=(114,114,114))

def apply_scale(img, scale):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    return cv2.warpAffine(img, M, (w, h), borderValue=(114,114,114))

def apply_fliplr(img):
    return cv2.flip(img, 1)

def apply_perspective(img, strength):
    h, w = img.shape[:2]
    offset = int(strength * w)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[offset,offset],[w-offset,0],[w,h],[0,h-offset]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(114,114,114))

def apply_shear(img, shear_deg):
    h, w = img.shape[:2]
    shear_rad = np.radians(shear_deg)
    M = np.float32([[1, np.tan(shear_rad), 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w + int(h*abs(np.tan(shear_rad))), h), borderValue=(114,114,114))


# ============================================================
# モード: exp_003（shear/perspective 詳細比較）
# ============================================================
def run_exp_003_preview():
    """
    exp_003で有効化するshear/perspectiveの効果を段階的に比較。
    各値で原画→弱→exp_003採用値→強 の4段階を横並びで表示。
    さらにshear+perspective同時適用の効果も確認。
    """
    print("\n--- exp_003 shear/perspective 効果確認 ---")

    # shearの段階比較: 0, 5, 10(採用値), 15, 20
    shear_values = [0, 5, 10, 15, 20]
    # perspectiveの段階比較: 0, 0.0005, 0.001(採用値), 0.002, 0.003
    perspective_values = [0, 0.0005, 0.001, 0.002, 0.003]

    for idx, img_name in enumerate(samples):
        img = cv2.imread(os.path.join(DATA_DIR, img_name))
        if img is None:
            continue
        img = cv2.resize(img, (640, 640))

        # --- shear段階比較 ---
        panels = []
        for val in shear_values:
            try:
                augmented = apply_shear(img.copy(), val)
            except:
                augmented = img.copy()
            augmented = cv2.resize(augmented, (250, 250))
            label = f"shear={val}"
            if val == 10:
                label += " [exp_003]"
                color = (0, 255, 0)  # 採用値は緑
            else:
                color = (255, 255, 255)
            cv2.putText(augmented, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            panels.append(augmented)

        combined = np.hstack(panels)
        out_path = os.path.join(OUTPUT_DIR, f"exp003_sample{idx}_shear_compare.jpg")
        cv2.imwrite(out_path, combined)
        print(f"  保存: exp003_sample{idx}_shear_compare.jpg")

        # --- perspective段階比較 ---
        panels = []
        for val in perspective_values:
            try:
                augmented = apply_perspective(img.copy(), val)
            except:
                augmented = img.copy()
            augmented = cv2.resize(augmented, (250, 250))
            label = f"persp={val}"
            if val == 0.001:
                label += " [exp_003]"
                color = (0, 255, 0)
            else:
                color = (255, 255, 255)
            cv2.putText(augmented, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            panels.append(augmented)

        combined = np.hstack(panels)
        out_path = os.path.join(OUTPUT_DIR, f"exp003_sample{idx}_perspective_compare.jpg")
        cv2.imwrite(out_path, combined)
        print(f"  保存: exp003_sample{idx}_perspective_compare.jpg")

        # --- shear + perspective 同時適用（exp_003の実際の組み合わせ）---
        combos = [
            ("original",               0,    0),
            ("shear=10",              10,    0),
            ("persp=0.001",            0,    0.001),
            ("shear=10+persp=0.001",  10,    0.001),  # exp_003の組み合わせ
            ("shear=15+persp=0.002",  15,    0.002),  # より強い組み合わせ（参考）
        ]
        panels = []
        for label, s_val, p_val in combos:
            try:
                augmented = img.copy()
                if s_val > 0:
                    augmented = apply_shear(augmented, s_val)
                    augmented = cv2.resize(augmented, (640, 640))
                if p_val > 0:
                    augmented = apply_perspective(augmented, p_val)
            except:
                augmented = img.copy()
            augmented = cv2.resize(augmented, (250, 250))

            if "shear=10+persp" in label:
                color = (0, 255, 0)  # 採用組み合わせは緑
            else:
                color = (255, 255, 255)
            cv2.putText(augmented, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            panels.append(augmented)

        combined = np.hstack(panels)
        out_path = os.path.join(OUTPUT_DIR, f"exp003_sample{idx}_combined.jpg")
        cv2.imwrite(out_path, combined)
        print(f"  保存: exp003_sample{idx}_combined.jpg")

    print(f"\n完了！ exp_003用プレビュー: {len(samples) * 3} 枚")


# ============================================================
# モード: all（従来の全Augmentation一覧）
# ============================================================
def run_all_preview():
    """従来の動作: 全Augmentationのmin/max比較"""
    print("\n--- 全Augmentation一覧 ---")

    augmentations = [
        ("hsv_h",       lambda img, v: apply_hsv(img, v, 0, 0),    -0.015, 0.015),
        ("hsv_s",       lambda img, v: apply_hsv(img, 0, v, 0),    -0.7,   0.7),
        ("hsv_v",       lambda img, v: apply_hsv(img, 0, 0, v),    -0.4,   0.4),
        ("degrees",     lambda img, v: apply_rotation(img, v),      -10.0,  10.0),
        ("translate",   lambda img, v: apply_translate(img, v, v),  -0.1,   0.1),
        ("scale",       lambda img, v: apply_scale(img, v),          0.5,   1.5),
        ("fliplr",      lambda img, v: apply_fliplr(img),            0,     1),
        ("perspective", lambda img, v: apply_perspective(img, v),    0.0,   0.001),
        ("shear",       lambda img, v: apply_shear(img, v),         -15.0,  15.0),
    ]

    for idx, img_name in enumerate(samples):
        img = cv2.imread(os.path.join(DATA_DIR, img_name))
        if img is None:
            continue
        img = cv2.resize(img, (640, 640))

        for aug_name, aug_fn, min_val, max_val in augmentations:
            original = img.copy()
            try:
                img_min = aug_fn(img.copy(), min_val)
            except:
                img_min = img.copy()
            try:
                img_max = aug_fn(img.copy(), max_val)
            except:
                img_max = img.copy()

            original = cv2.resize(original, (300, 300))
            img_min = cv2.resize(img_min, (300, 300))
            img_max = cv2.resize(img_max, (300, 300))

            cv2.putText(original, "original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(img_min, f"min={min_val}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(img_max, f"max={max_val}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            combined = np.hstack([original, img_min, img_max])
            out_path = os.path.join(OUTPUT_DIR, f"sample{idx}_{aug_name}.jpg")
            cv2.imwrite(out_path, combined)
            print(f"  保存: sample{idx}_{aug_name}.jpg")

    print(f"\n完了！ 全Augmentationプレビュー: {len(samples) * len(augmentations)} 枚")


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    if MODE == "exp_003":
        run_exp_003_preview()
    elif MODE == "all":
        run_all_preview()

    print(f"\n保存先: {OUTPUT_DIR}")
    print("エクスプローラーで開いて確認してください")
