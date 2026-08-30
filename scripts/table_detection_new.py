"""
Table Transformer (TATR) — استخراج صفوف وأعمدة الجدول (نسخة محسّنة)
====================================================================
يشتغل على جهازكم (يحتاج انترنت وقت أول تشغيل بس، لتحميل الموديلات).

الفرق عن النسخة القديمة:
    1) ما عاد نشغّل موديل الـ structure على الصفحة كاملة —
       نكتشف صندوق الجدول نفسه أولاً (Table Detection)، نقصّه،
       ثم نشغّل عليه موديل الـ structure. هذا وحده بيحسّن النتيجة
       بشكل كبير جداً لأن الموديل مُدرَّب أصلاً على صور جدول مقصوص.
    2) لو الجدول المقصوص صغير، نكبّره (upscale) قبل ما نمرره للموديل
       عشان الخطوط الرفيعة بين الأعمدة ما تضيعش.
    3) نضيف تصفية NMS (Non-Max Suppression) لحذف الصناديق المكررة
       المتراكبة على بعضها.
    4) عتبة الثقة صارت قابلة للتعديل بسهولة، ونزّلناها شوي افتراضياً
       (0.5) لأن الخط يدوي وقد يضيّع الموديل بعض الصفوف/الأعمدة
       بثقة عالية.

المتطلبات:
    pip install transformers torch pillow
"""

from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import numpy as np
import cv2
import torch

DETECTION_MODEL_NAME = "microsoft/table-transformer-detection"
STRUCTURE_MODEL_NAME = "microsoft/table-transformer-structure-recognition-v1.1-all"

print("جاري تحميل موديلات TATR (أول مرة فقط)...")
det_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_NAME)
det_model = TableTransformerForObjectDetection.from_pretrained(DETECTION_MODEL_NAME)

struct_processor = AutoImageProcessor.from_pretrained(STRUCTURE_MODEL_NAME)
struct_model = TableTransformerForObjectDetection.from_pretrained(STRUCTURE_MODEL_NAME)

# --- تصحيح توافق إصدارات transformers ---
# بعض نسخ transformers الحديثة تتوقع إن قاموس "size" يحتوي
# shortest_edge + longest_edge (أو height + width) مع بعض،
# لكن preprocessor_config.json لموديلات TATR فيه "longest_edge" بس،
# فيطلع: ValueError: Size must contain 'height' and 'width' keys...
# الحل: نحدد size يدوياً بعد التحميل مباشرة لكل processor.
for proc in (det_processor, struct_processor):
    proc.size = {"shortest_edge": 800, "longest_edge": 1000}
print("تم التحميل.\n")

# الحد الأدنى لعرض الجدول المقصوص بالبكسل قبل ما نبدأ نكبّره (upscale)
MIN_CROP_WIDTH = 1200


def _run_model(image, processor, model, threshold):
    """تشغيل أي موديل TATR (كشف جدول أو بنية) على صورة PIL، ويرجع قائمة detections"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 3),
            "box": [round(v, 1) for v in box.tolist()],  # [x0, y0, x1, y1]
        })
    return detections


def _iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(detections, iou_threshold=0.5):
    """يحذف الصناديق المكررة/المتراكبة لنفس النوع (label)، يبقي الأعلى ثقة فقط"""
    kept = []
    by_label = {}
    for d in detections:
        by_label.setdefault(d["label"], []).append(d)

    for label, items in by_label.items():
        items = sorted(items, key=lambda d: d["score"], reverse=True)
        chosen = []
        for cand in items:
            if all(_iou(cand["box"], c["box"]) < iou_threshold for c in chosen):
                chosen.append(cand)
        kept.extend(chosen)
    return kept


def detect_table(image_path, threshold=0.7, padding=15):
    """
    يكتشف صندوق الجدول نفسه داخل الصفحة، ويرجع:
        (crop_image, offset_x, offset_y, scale)
    - crop_image: صورة الجدول مقصوصة (ومكبّرة لو كانت صغيرة)
    - offset_x, offset_y: إحداثيات ركن القص بالنسبة للصورة الأصلية
      (تحتاجوها لو تبوا ترجعوا الصناديق لإحداثيات الصفحة الأصلية)
    - scale: معامل التكبير اللي طُبّق على القصة (1.0 لو ما كبّرناش)
    """
    image = Image.open(image_path).convert("RGB")
    detections = _run_model(image, det_processor, det_model, threshold)
    tables = [d for d in detections if d["label"] == "table"]

    if not tables:
        print("تحذير: ما تلقاش أي جدول بالكشف التلقائي — رح نستخدم الصفحة كاملة.")
        crop = image
        offset_x, offset_y = 0, 0
    else:
        # ناخذوا أعلى جدول ثقة (لو فيه أكثر من جدول بالصفحة، عدّلوا هذا السطر)
        best = max(tables, key=lambda d: d["score"])
        x0, y0, x1, y1 = best["box"]
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(image.width, x1 + padding)
        y1 = min(image.height, y1 + padding)
        crop = image.crop((x0, y0, x1, y1))
        offset_x, offset_y = x0, y0
        print(f"تم اكتشاف الجدول: ثقة={best['score']}  الحجم={crop.size}")

    scale = 1.0
    if crop.width < MIN_CROP_WIDTH:
        scale = MIN_CROP_WIDTH / crop.width
        new_size = (int(crop.width * scale), int(crop.height * scale))
        crop = crop.resize(new_size, Image.LANCZOS)
        print(f"تم تكبير القصة (upscale) بمعامل {scale:.2f} → {crop.size}")

    return crop, offset_x, offset_y, scale


def detect_structure(image_or_path, confidence_threshold=0.5, nms_iou=0.5):
    """
    يرجع قائمة عناصر بعد التصفية (NMS)، كل عنصر فيه:
        {"label": "table row" | "table column" | ..., "score": float, "box": [x0,y0,x1,y1]}
    image_or_path: يقبل مسار صورة أو صورة PIL جاهزة (مثلاً القصة الناتجة من detect_table)
    """
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert("RGB")
    else:
        image = image_or_path

    detections = _run_model(image, struct_processor, struct_model, confidence_threshold)
    detections = _nms(detections, iou_threshold=nms_iou)
    return detections


def separate_rows_and_columns(detections):
    """يفرز النتائج لصفوف وأعمدة بمفردهم (يشمل صفوف/أعمدة العناوين كمان)، مرتبين حسب الموقع"""
    row_labels = {"table row", "table row header"}
    col_labels = {"table column", "table column header"}

    rows = [d for d in detections if d["label"] in row_labels]
    columns = [d for d in detections if d["label"] in col_labels]

    rows.sort(key=lambda d: d["box"][1])     # رتبوهم عمودياً (y0)
    columns.sort(key=lambda d: d["box"][0])  # رتبوهم أفقياً (x0)
    return rows, columns


def visualize(pil_image, rows, columns, output_path="tatr_preview.png"):
    """يرسم الصفوف (أخضر) والأعمدة (أزرق) فوق صورة قصة الجدول، ويحفظها كصورة PNG"""
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    for row in rows:
        x0, y0, x1, y1 = map(int, row["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)  # أخضر = صفوف
    for col in columns:
        x0, y0, x1, y1 = map(int, col["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)  # أزرق = أعمدة
    cv2.imwrite(output_path, img)
    print(f"\nتم حفظ المعاينة البصرية: {output_path}")


if __name__ == "__main__":
    # ===== مسار صورتكم =====
    IMAGE_PATH = r"C:\Users\VICTUS\1954-Census-Digitization-Project-second-version\data\processed\1954-P000002_page-0001_denoised.png"

    # المرحلة 1: اكتشاف الجدول وقصه (وتكبيره لو صغير)
    table_crop, off_x, off_y, scale = detect_table(IMAGE_PATH, threshold=0.7)

    # المرحلة 2: اكتشاف الصفوف/الأعمدة على القصة فقط
    detections = detect_structure(table_crop, confidence_threshold=0.5)
    rows, columns = separate_rows_and_columns(detections)

    print(f"\nعدد الصفوف المكتشفة: {len(rows)}")
    print(f"عدد الأعمدة المكتشفة: {len(columns)}")

    for r in rows:
        print(f"  صف: y={r['box'][1]}-{r['box'][3]}  ثقة={r['score']}")
    for c in columns:
        print(f"  عمود: x={c['box'][0]}-{c['box'][2]}  ثقة={c['score']}")

    # المرحلة 3: حفظ صورة توضح النتيجة (المعاينة البصرية)
    visualize(table_crop, rows, columns, output_path="tatr_preview.png")