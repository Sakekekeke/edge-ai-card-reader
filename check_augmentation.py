import cv2
import numpy as np
import os
import random

# --- 設定 ---
DATA_DIR = r"D:\edge-ai-card-reader\train\images"
OUTPUT_DIR = r"D:\edge-ai-card-reader\aug_preview"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

# サンプル画像を3枚選ぶ
all_images = [f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.png'))]
samples = random.sample(all_images, 3)
print(f"選択した画像: {samples}")

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

# --- Augmentation定義 ---
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

# --- 各サンプルで全augmentationを適用 ---
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

        # リサイズして統一
        original = cv2.resize(original, (300, 300))
        img_min = cv2.resize(img_min, (300, 300))
        img_max = cv2.resize(img_max, (300, 300))

        # ラベル追加
        cv2.putText(original, "original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(img_min, f"min={min_val}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(img_max, f"max={max_val}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        combined = np.hstack([original, img_min, img_max])
        out_path = os.path.join(OUTPUT_DIR, f"sample{idx}_{aug_name}.jpg")
        cv2.imwrite(out_path, combined)
        print(f"  保存: sample{idx}_{aug_name}.jpg")

print(f"\n完了！ {OUTPUT_DIR} に画像を保存しました")
print(f"合計: {len(samples) * len(augmentations)} 枚")
print(f"エクスプローラーで開いて確認してください")