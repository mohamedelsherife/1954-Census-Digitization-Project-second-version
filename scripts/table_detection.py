import cv2
import numpy as np
import os
import glob


# =========================================================
# Find line positions
# =========================================================

def find_horizontal_lines(horizontal_image):
    """
    Find Y positions of horizontal table lines.
    """

    # Sum white pixels for every row
    row_sum = np.sum(
        horizontal_image > 0,
        axis=1
    )

    # Threshold
    threshold = horizontal_image.shape[1] * 0.05

    positions = np.where(
        row_sum > threshold
    )[0]

    # Group nearby Y positions
    lines = []

    if len(positions) == 0:
        return lines

    start = positions[0]
    previous = positions[0]

    for position in positions[1:]:

        if position - previous <= 5:

            previous = position

        else:

            center = (start + previous) // 2

            lines.append(center)

            start = position
            previous = position

    # Last group
    center = (start + previous) // 2

    lines.append(center)

    return lines


# =========================================================
# Find vertical line positions
# =========================================================

def find_vertical_lines(vertical_image):
    """
    Find X positions of vertical table lines.
    """

    # Sum white pixels for every column
    column_sum = np.sum(
        vertical_image > 0,
        axis=0
    )

    # Threshold
    threshold = vertical_image.shape[0] * 0.05

    positions = np.where(
        column_sum > threshold
    )[0]

    # Group nearby X positions
    lines = []

    if len(positions) == 0:
        return lines

    start = positions[0]
    previous = positions[0]

    for position in positions[1:]:

        if position - previous <= 5:

            previous = position

        else:

            center = (start + previous) // 2

            lines.append(center)

            start = position
            previous = position

    # Last group
    center = (start + previous) // 2

    lines.append(center)

    return lines


# =========================================================
# Remove lines that are too close
# =========================================================

def merge_close_lines(lines, minimum_distance=15):
    """
    Merge line coordinates that are very close.
    """

    if not lines:
        return []

    lines = sorted(lines)

    merged = [
        lines[0]
    ]

    for line in lines[1:]:

        if line - merged[-1] >= minimum_distance:

            merged.append(line)

        else:

            # Average two close lines
            merged[-1] = (
                merged[-1] + line
            ) // 2

    return merged


# =========================================================
# Detect table
# =========================================================

def detect_table(
    image_path,
    output_dir="data/cells"
):

    # =====================================================
    # 1. Read image
    # =====================================================

    img = cv2.imread(
        image_path
    )

    if img is None:

        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    # =====================================================
    # 2. Threshold
    # =====================================================

    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    # =====================================================
    # 3. Horizontal lines
    # =====================================================

    horizontal_length = max(
        20,
        width // 50
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            horizontal_length,
            1
        )
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Repair horizontal gaps
    horizontal_close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (30, 3)
    )

    horizontal = cv2.morphologyEx(
        horizontal,
        cv2.MORPH_CLOSE,
        horizontal_close_kernel
    )

    # =====================================================
    # 4. Vertical lines
    # =====================================================

    vertical_length = max(
        20,
        height // 50
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            vertical_length
        )
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    # Repair vertical gaps
    vertical_close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 30)
    )

    vertical = cv2.morphologyEx(
        vertical,
        cv2.MORPH_CLOSE,
        vertical_close_kernel
    )

    # =====================================================
    # 5. Combine lines
    # =====================================================

    grid = cv2.add(
        horizontal,
        vertical
    )

    # =====================================================
    # 6. Output directory
    # =====================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # =====================================================
    # 7. Save grid
    # =====================================================

    grid_path = os.path.join(
        output_dir,
        f"{base_name}_detected_grid.png"
    )

    cv2.imwrite(
        grid_path,
        grid
    )

    # =====================================================
    # 8. Find horizontal line positions
    # =====================================================

    horizontal_positions = find_horizontal_lines(
        horizontal
    )

    # =====================================================
    # 9. Find vertical line positions
    # =====================================================

    vertical_positions = find_vertical_lines(
        vertical
    )

    # =====================================================
    # 10. Merge close horizontal lines
    # =====================================================

    horizontal_positions = merge_close_lines(
        horizontal_positions,
        minimum_distance=15
    )

    # =====================================================
    # 11. Merge close vertical lines
    # =====================================================

    vertical_positions = merge_close_lines(
        vertical_positions,
        minimum_distance=15
    )

    # =====================================================
    # 12. Print detected lines
    # =====================================================

    print()
    print("=" * 60)

    print(
        "Horizontal lines:",
        horizontal_positions
    )

    print(
        "Vertical lines:",
        vertical_positions
    )

    print(
        "Number of horizontal lines:",
        len(horizontal_positions)
    )

    print(
        "Number of vertical lines:",
        len(vertical_positions)
    )

    # =====================================================
    # 13. Create bounding boxes
    # =====================================================

    boxes = []

    # Each two horizontal lines
    # + two vertical lines
    # = one cell

    for row in range(
        len(horizontal_positions) - 1
    ):

        y1 = horizontal_positions[row]
        y2 = horizontal_positions[row + 1]

        cell_height = y2 - y1

        # Ignore very small rows
        if cell_height < 20:
            continue

        for column in range(
            len(vertical_positions) - 1
        ):

            x1 = vertical_positions[column]
            x2 = vertical_positions[column + 1]

            cell_width = x2 - x1

            # Ignore very small columns
            if cell_width < 20:
                continue

            # Add small padding
            padding = 2

            x = x1 + padding
            y = y1 + padding

            w = (
                x2 -
                x1 -
                padding * 2
            )

            h = (
                y2 -
                y1 -
                padding * 2
            )

            boxes.append(
                (
                    x,
                    y,
                    w,
                    h
                )
            )

    # =====================================================
    # 14. Create rows
    # =====================================================

    rows = []

    index = 0

    number_of_rows = (
        len(horizontal_positions) - 1
    )

    number_of_columns = (
        len(vertical_positions) - 1
    )

    for row_index in range(
        number_of_rows
    ):

        row = []

        for column_index in range(
            number_of_columns
        ):

            if index < len(boxes):

                row.append(
                    boxes[index]
                )

            index += 1

        if row:

            rows.append(
                row
            )

    # =====================================================
    # 15. Draw bounding boxes
    # =====================================================

    result_image = img.copy()

    for row_index, row in enumerate(rows):

        for column_index, box in enumerate(row):

            x, y, w, h = box

            # ---------------------------------------------
            # Bounding box
            # ---------------------------------------------

            cv2.rectangle(
                result_image,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                (0, 0, 255),
                2
            )

            # ---------------------------------------------
            # Label
            # ---------------------------------------------

            cv2.putText(
                result_image,
                f"R{row_index} C{column_index}",
                (
                    x + 5,
                    y + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

    # =====================================================
    # 16. Draw detected line positions
    # =====================================================

    line_image = img.copy()

    # Horizontal lines
    for y in horizontal_positions:

        cv2.line(
            line_image,
            (0, y),
            (width, y),
            (0, 255, 0),
            2
        )

    # Vertical lines
    for x in vertical_positions:

        cv2.line(
            line_image,
            (x, 0),
            (x, height),
            (255, 0, 0),
            2
        )

    # =====================================================
    # 17. Save bounding boxes
    # =====================================================

    result_path = os.path.join(
        output_dir,
        f"{base_name}_detected_cells.png"
    )

    cv2.imwrite(
        result_path,
        result_image
    )

    # =====================================================
    # 18. Save detected lines
    # =====================================================

    lines_path = os.path.join(
        output_dir,
        f"{base_name}_detected_lines.png"
    )

    cv2.imwrite(
        lines_path,
        line_image
    )

    # =====================================================
    # 19. Print results
    # =====================================================

    print(
        "Number of rows:",
        len(rows)
    )

    print(
        "Number of cells:",
        sum(
            len(row)
            for row in rows
        )
    )

    print(
        "Cells per row:",
        [
            len(row)
            for row in rows
        ]
    )

    print(
        "Grid saved:",
        grid_path
    )

    print(
        "Lines saved:",
        lines_path
    )

    print(
        "Bounding boxes saved:",
        result_path
    )

    print(
        "=" * 60
    )

    return rows


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # Find all denoised images
    image_paths = glob.glob(
        "data/processed/*_denoised.png"
    )

    if not image_paths:

        print(
            "لم يتم العثور على أي صور في "
            "data/processed/"
        )

    # Store results
    all_results = {}

    # Process every image
    for image_path in image_paths:

        print()
        print(
            "Processing:",
            image_path
        )

        try:

            rows = detect_table(
                image_path
            )

            all_results[
                image_path
            ] = rows

        except Exception as e:

            print(
                f"Error processing "
                f"{image_path}: {e}"
            )