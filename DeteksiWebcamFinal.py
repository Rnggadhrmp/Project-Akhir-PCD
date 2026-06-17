from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Warna tiap kelas (BGR)
colors = {
    "Tomato-Late-blight": (0, 0, 255),      # Merah
    "Tomato_Septoria_leaf_spot": (255, 0, 255) # Ungu
}

# Buka webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Prediksi
    results = model.predict(
        source=frame,
        conf=0.5,
        iou=0.45,
        verbose=False
    )

    annotated_frame = frame.copy()

    best_box = None
    best_conf = 0

    # Cari deteksi terbaik
    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf
                best_box = box

    # Tampilkan hanya deteksi terbaik
    if best_box is not None:

        cls_id = int(best_box.cls[0])
        conf = float(best_box.conf[0])

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        class_name = model.names[cls_id]

        color = colors.get(class_name, (255, 255, 255))

        label = f"{class_name} {conf:.2f}"

        # Bounding box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # Background label
        (w, h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        cv2.rectangle(
            annotated_frame,
            (x1, y1 - h - 12),
            (x1 + w + 10, y1),
            color,
            -1
        )

        # Teks label
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("Tomato Disease Detection", annotated_frame)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()