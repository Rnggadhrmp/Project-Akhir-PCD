from ultralytics import YOLO
import cv2
import glob
import os
import numpy as np

# ==========================
# LOAD MODEL
# ==========================
model = YOLO("best.pt")

# ==========================
# DATASET FOLDER
# ==========================
folder_path = "Tomato_Septoria_leaf_spot"

# Cari semua gambar
image_files = glob.glob(
    os.path.join(folder_path, "**", "*.*"),
    recursive=True
)

image_files = [
    f for f in image_files
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("=" * 50)
print(f"Total gambar ditemukan : {len(image_files)}")
print("=" * 50)

if len(image_files) == 0:
    print("Tidak ada gambar ditemukan!")
    exit()

# ==========================
# PARAMETER
# ==========================
CARD_W = 300
IMAGE_H = 250
INFO_H = 40

BATCH_SIZE = 10

# ==========================
# LOOP BATCH
# ==========================
for batch_start in range(0, len(image_files), BATCH_SIZE):

    batch_files = image_files[batch_start:batch_start+BATCH_SIZE]
    cards = []

    print(
        f"\nBatch {batch_start//BATCH_SIZE + 1}"
    )

    for idx, img_path in enumerate(batch_files):

        img = cv2.imread(img_path)

        if img is None:
            continue

        # ======================
        # YOLO INFERENCE
        # ======================
        results = model(
            img,
            conf=0.50,
            verbose=False
        )

        annotated = img.copy()

        # ======================
        # DRAW BOX
        # ======================
        for box in results[0].boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            class_name = model.names[cls]

            label = f"{class_name} {conf:.2f}"

            # Bounding box hijau
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Ukuran teks
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )

            # Pastikan label tidak keluar gambar
            label_y = max(y1, th + 15)

            # Background label
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 10),
                (x1 + tw + 10, label_y),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ======================
        # RESIZE GAMBAR
        # ======================
        annotated = cv2.resize(
            annotated,
            (CARD_W, IMAGE_H)
        )

        # ======================
        # PANEL NAMA FILE
        # ======================
        info_panel = np.full(
            (INFO_H, CARD_W, 3),
            40,
            dtype=np.uint8
        )

        image_number = batch_start + idx + 1

        cv2.putText(
            info_panel,
            f"Image {image_number:03d}",
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Gabungkan
        card = np.vstack([
            annotated,
            info_panel
        ])

        # Border putih
        card = cv2.copyMakeBorder(
            card,
            3, 3, 3, 3,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        cards.append(card)

        print(f"✓ Image {image_number:03d}")

    # ==========================
    # FILL KOSONG
    # ==========================
    while len(cards) < 10:

        blank = np.full(
            (IMAGE_H + INFO_H + 6, CARD_W + 6, 3),
            30,
            dtype=np.uint8
        )

        cv2.putText(
            blank,
            "EMPTY",
            (105, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (180, 180, 180),
            2
        )

        cards.append(blank)

    # ==========================
    # GAP ANTAR GAMBAR
    # ==========================
    gap_h = np.full(
        (IMAGE_H + INFO_H + 6, 10, 3),
        25,
        dtype=np.uint8
    )

    gap_v = np.full(
        (10, (CARD_W + 6) * 5 + 40, 3),
        25,
        dtype=np.uint8
    )

    # ==========================
    # BARIS PERTAMA
    # ==========================
    row1 = np.hstack([
        cards[0], gap_h,
        cards[1], gap_h,
        cards[2], gap_h,
        cards[3], gap_h,
        cards[4]
    ])

    # ==========================
    # BARIS KEDUA
    # ==========================
    row2 = np.hstack([
        cards[5], gap_h,
        cards[6], gap_h,
        cards[7], gap_h,
        cards[8], gap_h,
        cards[9]
    ])

    gallery = np.vstack([
        row1,
        gap_v,
        row2
    ])

    # ==========================
    # HEADER
    # ==========================
    header = np.full(
        (60, gallery.shape[1], 3),
        20,
        dtype=np.uint8
    )

    cv2.putText(
        header,
        f"YOLO Tomato Disease Detection - Batch {batch_start//BATCH_SIZE + 1}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    final_view = np.vstack([
        header,
        gallery
    ])

    cv2.imshow(
        "YOLO Detection Viewer",
        final_view
    )

    print("\n[SPACE] Batch berikutnya")
    print("[Q] Keluar")

    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()