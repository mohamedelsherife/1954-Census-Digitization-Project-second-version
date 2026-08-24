import cv2
import numpy as np
import os
import glob

def detect_table(image_path, output_dir="data/cells"):
    """
    Detect rows, columns, and cells from a preprocessed/denoised image.
    """

    # ==========================================
    # 1. Read the preprocessed image
    # ==========================================

    # (تم حذف السطر اللي كان يعمل override لـ image_path هنا،
    #  عشان الدالة تشتغل على أي صورة تنبعتلها كباراميتر)

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # ==========================================
    # 2. Convert to binary
    # ==========================================

    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    # ==========================================
    # 3. Detect horizontal lines
    # ==========================================

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (width // 30, 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # ==========================================
    # 4. Detect vertical lines
    # ==========================================

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, height // 30)
    )

    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    # ==========================================
    # 5. Combine horizontal + vertical lines
    # ==========================================

    grid = cv2.add(
        horizontal_lines,
        vertical_lines
    )

    # Save grid image
    os.makedirs(output_dir, exist_ok=True)

    # اسم الصورة بدون الامتداد، عشان نستخدمه في تسمية ملفات الإخراج
    # فكل صورة تطلعلها ملفاتها الخاصة وما تتكتبش فوق بعض
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    grid_path = os.path.join(
        output_dir,
        f"{base_name}_detected_grid.png"
    )

    cv2.imwrite(grid_path, grid)

    # ==========================================
    # 6. Find contours
    # ==========================================

    contours, _ = cv2.findContours(
        grid,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ==========================================
    # 7. Get bounding boxes
    # ==========================================

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore very small objects
        if w < 20 or h < 20:
            continue

        # Ignore very large object
        if w > width * 0.95 and h > height * 0.95:
            continue

        boxes.append((x, y, w, h))

    # ==========================================
    # 8. Sort boxes by Y
    # ==========================================

    boxes.sort(key=lambda box: box[1])

    # ==========================================
    # 9. Detect rows
    # ==========================================

    rows = []

    row_tolerance = 15

    for box in boxes:

        x, y, w, h = box

        added = False

        for row in rows:

            first_y = row[0][1]

            if abs(y - first_y) < row_tolerance:

                row.append(box)
                added = True
                break

        if not added:
            rows.append([box])

    # ==========================================
    # 10. Sort rows
    # ==========================================

    rows.sort(
        key=lambda row: row[0][1]
    )

    # Sort cells inside each row by X
    for row in rows:

        row.sort(
            key=lambda box: box[0]
        )

    # ==========================================
    # 11. Draw bounding boxes
    # ==========================================

    result_image = img.copy()

    for row_index, row in enumerate(rows):

        for column_index, box in enumerate(row):

            x, y, w, h = box

            # Draw bounding box
            cv2.rectangle(
                result_image,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            # Write row-column number
            cv2.putText(
                result_image,
                f"R{row_index} C{column_index}",
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

    # ==========================================
    # 12. Save result
    # ==========================================

    result_path = os.path.join(
        output_dir,
        f"{base_name}_detected_cells.png"
    )

    cv2.imwrite(
        result_path,
        result_image
    )

    # ==========================================
    # 13. Print results
    # ==========================================

    print(f"[{base_name}] Number of rows:", len(rows))

    print(
        f"[{base_name}] Number of cells:",
        sum(len(row) for row in rows)
    )

    print(
        f"[{base_name}] Cells per row:",
        [len(row) for row in rows]
    )

    return rows


# ==================================================
# Run the program على أكثر من صورة
# ==================================================

if __name__ == "__main__":

    # ياخذ تلقائيًا كل الصور الموجودة فعليًا في المجلد
    # (بدل ما نكتب أسماء يدوي ونضطر نتأكد كل مرة إنها موجودة)
    image_paths = glob.glob("data/processed/*_denoised.png")

    if not image_paths:
        print("لم يتم العثور على أي صور في data/processed/")

    all_results = {}

    for image_path in image_paths:

        if not os.path.exists(image_path):
            print(f"تخطي: الملف غير موجود -> {image_path}")
            continue

        rows = detect_table(image_path)

        all_results[image_path] = rows