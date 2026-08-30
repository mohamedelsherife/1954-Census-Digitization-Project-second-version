"""
تحويل نتيجة TATR (صفوف + أعمدة منفصلين) لصناديق خلايا فعلية (نسخة محسّنة)
====================================================================
TATR يرجّع صندوق لكل صف كامل وصندوق لكل عمود كامل، مو خلية مباشرة —
هذا الملف يحسب تقاطعاتهم عشان نطلع بصندوق (خلية) واحد لكل تقاطع صف×عمود.

الفرق عن النسخة القديمة: نشتغل الآن على "قصة الجدول" الناتجة من
detect_table() مباشرة (numpy/PIL) بدل ما نعاود نقرأ الصورة الأصلية —
هذا يخلي صناديق الخلايا متطابقة تماماً مع نفس الصورة اللي شغّلنا عليها
موديل الـ structure، فتجيكم المعاينة البصرية مطابقة صح.
"""

import numpy as np
import cv2
from tatr_structure import detect_table, detect_structure, separate_rows_and_columns


def build_cells_from_tatr(rows, columns):
    """
    كل خلية = تقاطع مستطيل الصف مع مستطيل العمود.
    نرجع: قائمة (row_idx, col_idx, x0, y0, x1, y1)
    """
    cells = []
    for r_idx, row in enumerate(rows):
        _, ry0, _, ry1 = row["box"]
        for c_idx, col in enumerate(columns):
            cx0, _, cx1, _ = col["box"]
            cells.append((r_idx, c_idx, cx0, ry0, cx1, ry1))
    return cells


def visualize(pil_image, rows, columns, output_path="tatr_preview.png"):
    """يرسم الصفوف (أخضر) والأعمدة (أزرق) فوق صورة قصة الجدول نفسها"""
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    for row in rows:
        x0, y0, x1, y1 = map(int, row["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)  # أخضر = صفوف
    for col in columns:
        x0, y0, x1, y1 = map(int, col["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)  # أزرق = أعمدة
    cv2.imwrite(output_path, img)
    print(f"تم حفظ المعاينة البصرية: {output_path}")
    print("راجعوها بعينكم — قارنوها مع نتيجة OpenCV اللي عندنا من قبل")


def visualize_cells(pil_image, cells, output_path="tatr_cells_preview.png"):
    """اختياري: يرسم كل خلية لحالها، مفيد للتأكد إن التقاطعات صح"""
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    for (r_idx, c_idx, x0, y0, x1, y1) in cells:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)
    cv2.imwrite(output_path, img)
    print(f"تم حفظ معاينة الخلايا: {output_path}")


if __name__ == "__main__":
    # ===== مسار صورتكم (نفس المسار اللي بملف tatr_structure.py) =====
    IMAGE_PATH = r"C:\Users\VICTUS\1954-Census-Digitization-Project-second-version\data\processed\1954-P000002_page-0001_denoised.png"

    # المرحلة 1: اكتشاف الجدول وقصه (وتكبيره لو صغير) — نفس القصة تُستخدم بعدين للرسم
    table_crop, off_x, off_y, scale = detect_table(IMAGE_PATH, threshold=0.7)

    # المرحلة 2: اكتشاف الصفوف/الأعمدة على القصة فقط
    detections = detect_structure(table_crop, confidence_threshold=0.5)
    rows, columns = separate_rows_and_columns(detections)

    visualize(table_crop, rows, columns)

    cells = build_cells_from_tatr(rows, columns)
    print(f"إجمالي الخلايا الناتجة: {len(cells)}")

    visualize_cells(table_crop, cells)