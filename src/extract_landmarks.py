"""
Landmark Extraction Script
Reads images from:
    dataset/ASL/asl_alphabet_train/<label>/<image>.jpg
    dataset/ISL/isl_dataset/<label>/<image>.jpg

Outputs:
    data/asl_landmarks.csv
    data/isl_landmarks.csv
"""

import cv2
import mediapipe as mp
import csv
import os
from pathlib import Path

# ─────────────────────────────────────────────
#  Config — points directly to label folders
# ─────────────────────────────────────────────
LANGUAGES_PATHS = {
    "ASL": ["../dataset/ASL/asl_alphabet_train"],
    "ISL": ["../dataset/ISL/isl_dataset"],
}
OUTPUT_DIR = "data"

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Helper: extract landmarks from one image
# ─────────────────────────────────────────────
def extract_landmarks(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand.landmark]

    # Normalize relative to wrist (landmark 0)
    wx, wy, wz = landmarks[0]
    normalized = [(x - wx, y - wy, z - wz) for x, y, z in landmarks]

    flat = [val for (x, y, z) in normalized for val in (x, y, z)]
    return flat  # 63 values


# ─────────────────────────────────────────────
#  Main extraction loop
# ─────────────────────────────────────────────
for lang, paths in LANGUAGES_PATHS.items():
    output_csv = Path(OUTPUT_DIR) / f"{lang.lower()}_landmarks.csv"
    total = 0
    skipped = 0

    print(f"\n[{lang}] Extracting landmarks → {output_csv}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["label"] + [
            f"{axis}{i}"
            for i in range(21)
            for axis in ("x", "y", "z")
        ]
        writer.writerow(header)

        for base_path in paths:
            base = Path(base_path)
            if not base.exists():
                print(f"  [SKIP] {base_path} not found")
                continue

            label_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
            print(f"  Found {len(label_dirs)} labels: {[d.name for d in label_dirs]}")

            for label_dir in label_dirs:
                label = label_dir.name
                images = (
                    list(label_dir.glob("*.jpg")) +
                    list(label_dir.glob("*.jpeg")) +
                    list(label_dir.glob("*.png"))
                )

                label_count = 0
                for img_path in images:
                    landmarks = extract_landmarks(img_path)
                    if landmarks is None:
                        skipped += 1
                        continue
                    writer.writerow([label] + landmarks)
                    label_count += 1
                    total += 1

                print(f"    {label:>10s} : {label_count} saved  "
                      f"({len(images) - label_count} skipped)")

    print(f"\n  ✓ {lang} done — {total} rows saved, {skipped} skipped")
    print(f"  Saved to: {output_csv}")

hands.close()
print("\n[Done] All landmarks extracted.")
print("Next step: run train.py to train the classifier.")