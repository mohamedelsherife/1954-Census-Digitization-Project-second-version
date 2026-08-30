"""
تحويل نتيجة TATR (صفوف + أعمدة منفصلين) لصناديق خلايا فعلية
====================================================================
TATR يرجّع صندوق لكل صف كامل وصندوق لكل عمود كامل، مو خلية مباشرة —
هذا الملف يحسب تقاطعاتهم عشان نطلع بصندوق (خلية) واحد لكل تقاطع صف×عمود.
"""

import cv2
from tatr_structure import detect_structure, separate_rows_and_columns


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


def visualize(image_path, rows, columns, output_path="tatr_preview.png"):
    img = cv2.imread(image_path)
    for row in rows:
        x0, y0, x1, y1 = map(int, row["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)  # أخضر = صفوف
    for col in columns:
        x0, y0, x1, y1 = map(int, col["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 1)  # أزرق = أعمدة
    cv2.imwrite(output_path, img)
    print(f"تم حفظ المعاينة البصرية: {output_path}")
    print("راجعوها بعينكم — قارنوها مع نتيجة OpenCV اللي عندنا من قبل")


if __name__ == "__main__":
    # ===== مسار صورتكم (نفس المسار اللي بملف tatr_structure.py) =====
    IMAGE_PATH = r"""
تحويل نتيجة TATR (صفوف + أعمدة منفصلين) لصناديق خلايا فعلية
====================================================================
TATR يرجّع صندوق لكل صف كامل وصندوق لكل عمود كامل، مو خلية مباشرة —
هذا الملف يحسب تقاطعاتهم عشان نطلع بصندوق (خلية) واحد لكل تقاطع صف×عمود.
"""

import cv2
from tatr_structure import detect_structure, separate_rows_and_columns


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


def visualize(image_path, rows, columns, output_path="tatr_preview.png"):
    img = cv2.imread(image_path)
    for row in rows:
        x0, y0, x1, y1 = map(int, row["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)  # أخضر = صفوف
    for col in columns:
        x0, y0, x1, y1 = map(int, col["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 1)  # أزرق = أعمدة
    cv2.imwrite(output_path, img)
    print(f"تم حفظ المعاينة البصرية: {output_path}")
    print("راجعوها بعينكم — قارنوها مع نتيجة OpenCV اللي عندنا من قبل")


if __name__ == "__main__":
    # ===== مسار صورتكم (نفس المسار اللي بملف tatr_structure.py) =====
    IMAGE_PATH = r"C:\Users\VICTUS\1954-Census-Digitization-Project-second-version\data\processed\1954-P000002_page-0001_denoised.png"

    detections = detect_structure(IMAGE_PATH, confidence_threshold=0.7)
    rows, columns = separate_rows_and_columns(detections)

    visualize(IMAGE_PATH, rows, columns)

    cells = build_cells_from_tatr(rows, columns)
    print(f"إجمالي الخلايا الناتجة: {len(cells)}")

    detections = detect_structure(IMAGE_PATH, confidence_threshold=0.7)
    rows, columns = separate_rows_and_columns(detections)

    visualize(IMAGE_PATH, rows, columns)

    cells = build_cells_from_tatr(rows, columns)
    print(f"إجمالي الخلايا الناتجة: {len(cells)}")
