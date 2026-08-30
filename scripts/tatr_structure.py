"""
Table Transformer (TATR) — استخراج صفوف وأعمدة الجدول
====================================================================
يشتغل على جهازكم (يحتاج انترنت وقت أول تشغيل بس، لتحميل الموديل).

المتطلبات:
    pip install transformers torch pillow
"""

from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import torch

MODEL_NAME = "microsoft/table-transformer-structure-recognition-v1.1-all"

print("جاري تحميل TATR (أول مرة فقط)...")
processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    size={"shortest_edge": 800, "longest_edge": 1333},
)
model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME)
print("تم التحميل.\n")


def detect_structure(image_path, confidence_threshold=0.7):
    """
    يرجع قائمة عناصر، كل عنصر فيه:
        {"label": "table row" | "table column" | ..., "score": float, "box": [x0,y0,x1,y1]}
    """
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # target_sizes بترتيب (height, width) — عكس PIL.size اللي يرجع (width, height)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 3),
            "box": [round(v, 1) for v in box.tolist()],  # [x0, y0, x1, y1]
        })
    return detections


def separate_rows_and_columns(detections):
    """يفرز النتائج لصفوف وأعمدة بمفردهم، مرتبين حسب الموقع"""
    rows = [d for d in detections if d["label"] == "table row"]
    columns = [d for d in detections if d["label"] == "table column"]

    rows.sort(key=lambda d: d["box"][1])   # رتبوهم عمودياً (y0)
    columns.sort(key=lambda d: d["box"][0])  # رتبوهم أفقياً (x0)
    return rows, columns


if __name__ == "__main__":
    # ===== عدّلوا هذا لمسار صورة الجدول عندكم (نفس الصورة اللي رفعتوها لي) =====
    IMAGE_PATH = "1954-P000002_page-0001_denoised_table_part.png"

    detections = detect_structure(IMAGE_PATH, confidence_threshold=0.7)
    rows, columns = separate_rows_and_columns(detections)

    print(f"عدد الصفوف المكتشفة: {len(rows)}")
    print(f"عدد الأعمدة المكتشفة: {len(columns)}")

    # اطبعوا كل صف/عمود مع درجة الثقة (score) — راجعوها، لو أقل من 0.7
    # غالباً غلط أو تكرار
    for r in rows:
        print(f"  صف: y={r['box'][1]}-{r['box'][3]}  ثقة={r['score']}")
    for c in columns:
        print(f"  عمود: x={c['box'][0]}-{c['box'][2]}  ثقة={c['score']}")
