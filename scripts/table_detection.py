import cv2
import numpy as np
import os
import glob


# =========================================================
# 1. Find horizontal line positions
# =========================================================

def find_horizontal_lines(horizontal):

    row_sum = np.sum(
        horizontal > 0,
        axis=1
    )

    threshold = horizontal.shape[1] * 0.05

    positions = np.where(
        row_sum > threshold
    )[0]

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

    # Last line
    center = (start + previous) // 2
    lines.append(center)

    return lines


# =========================================================
# 2. Find vertical line positions
# =========================================================

def find_vertical_lines(vertical):

    column_sum = np.sum(
        vertical > 0,
        axis=0
    )

    threshold = vertical.shape[0] * 0.05

    positions = np.where(
        column_sum > threshold
    )[0]

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

    # Last line
    center = (start + previous) // 2
    lines.append(center)

    return lines


# =========================================================
# 3. Merge close lines
# =========================================================

def merge_close_lines(
    lines,
    minimum_distance=15
):

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

            merged[-1] = (
                merged[-1] + line
            ) // 2

    return merged


# =========================================================
# 4. Process ONE region
# =========================================================

def detect_region(
    region,
    region_name,
    output_dir
):

    height, width = region.shape[:2]

    # =====================================================
    # Grayscale
    # =====================================================

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY
    )

    # =====================================================
    # Threshold
    # =====================================================

    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    # =====================================================
    # Horizontal lines
    # =====================================================

    horizontal_length = max(
        15,
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

    # Repair gaps
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
    # Vertical lines
    # =====================================================

    vertical_length = max(
        15,
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

    # Repair gaps
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
    # Combine
    # =====================================================

    grid = cv2.add(
        horizontal,
        vertical
    )

    # =====================================================
    # Find line positions
    # =====================================================

    horizontal_positions = find_horizontal_lines(
        horizontal
    )

    vertical_positions = find_vertical_lines(
        vertical
    )

    # Merge close lines
    horizontal_positions = merge_close_lines(
        horizontal_positions,
        15
    )

    vertical_positions = merge_close_lines(
        vertical_positions,
        15
    )

    # =====================================================
    # Print detected lines
    # =====================================================

    print()
    print("-" * 50)

    print(
        f"{region_name} horizontal lines:",
        len(horizontal_positions)
    )

    print(
        f"{region_name} vertical lines:",
        len(vertical_positions)
    )

    print(
        f"{region_name} horizontal positions:",
        horizontal_positions
    )

    print(
        f"{region_name} vertical positions:",
        vertical_positions
    )

    # =====================================================
    # Create bounding boxes
    # =====================================================

    boxes = []

    for row in range(
        len(horizontal_positions) - 1
    ):

        y1 = horizontal_positions[row]
        y2 = horizontal_positions[row + 1]

        cell_height = y2 - y1

        if cell_height < 20:
            continue

        for column in range(
            len(vertical_positions) - 1
        ):

            x1 = vertical_positions[column]
            x2 = vertical_positions[column + 1]

            cell_width = x2 - x1

            if cell_width < 20:
                continue

            padding = 2

            x = x1 + padding
            y = y1 + padding

            w = (
                x2 -
                x1 -
                2 * padding
            )

            h = (
                y2 -
                y1 -
                2 * padding
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
    # Draw bounding boxes
    # =====================================================

    result = region.copy()

    for index, box in enumerate(boxes):

        x, y, w, h = box

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        # Calculate row / column
        columns = max(
            1,
            len(vertical_positions) - 1
        )

        row_index = index // columns
        column_index = index % columns

        cv2.putText(
            result,
            f"R{row_index} C{column_index}",
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    # =====================================================
    # Draw detected lines
    # =====================================================

    line_image = region.copy()

    # Horizontal = Green
    for y in horizontal_positions:

        cv2.line(
            line_image,
            (0, y),
            (width, y),
            (0, 255, 0),
            2
        )

    # Vertical = Blue
    for x in vertical_positions:

        cv2.line(
            line_image,
            (x, 0),
            (x, height),
            (255, 0, 0),
            2
        )

    # =====================================================
    # Save images
    # =====================================================

    grid_path = os.path.join(
        output_dir,
        f"{region_name}_grid.png"
    )

    lines_path = os.path.join(
        output_dir,
        f"{region_name}_lines.png"
    )

    boxes_path = os.path.join(
        output_dir,
        f"{region_name}_bounding_boxes.png"
    )

    cv2.imwrite(
        grid_path,
        grid
    )

    cv2.imwrite(
        lines_path,
        line_image
    )

    cv2.imwrite(
        boxes_path,
        result
    )

    # =====================================================
    # Print result
    # =====================================================

    print(
        f"{region_name} bounding boxes:",
        len(boxes)
    )

    print(
        f"Saved: {boxes_path}"
    )

    return boxes


# =========================================================
# 5. Main table detection
# =========================================================

def detect_table(
    image_path,
    output_dir="data/cells"
):

    # =====================================================
    # Read image
    # =====================================================

    img = cv2.imread(
        image_path
    )

    if img is None:

        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    height, width = img.shape[:2]

    # =====================================================
    # Create output directory
    # =====================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # =====================================================
    # Split image
    # =====================================================

    middle = width // 2

    left = img[
        :,
        :middle
    ]

    right = img[
        :,
        middle:
    ]

    # =====================================================
    # Save left / right images
    # =====================================================

    left_path = os.path.join(
        output_dir,
        f"{base_name}_LEFT.png"
    )

    right_path = os.path.join(
        output_dir,
        f"{base_name}_RIGHT.png"
    )

    cv2.imwrite(
        left_path,
        left
    )

    cv2.imwrite(
        right_path,
        right
    )

    print()
    print("=" * 60)

    print(
        "Original image:",
        image_path
    )

    print(
        "Image width:",
        width
    )

    print(
        "Image height:",
        height
    )

    print(
        "Splitting at X:",
        middle
    )

    # =====================================================
    # Process LEFT
    # =====================================================

    left_boxes = detect_region(
        left,
        f"{base_name}_LEFT",
        output_dir
    )

    # =====================================================
    # Process RIGHT
    # =====================================================

    right_boxes = detect_region(
        right,
        f"{base_name}_RIGHT",
        output_dir
    )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 60)

    print(
        "FINAL RESULT"
    )

    print(
        "LEFT boxes:",
        len(left_boxes)
    )

    print(
        "RIGHT boxes:",
        len(right_boxes)
    )

    print(
        "TOTAL boxes:",
        len(left_boxes) +
        len(right_boxes)
    )

    print("=" * 60)

    return {
        "left": left_boxes,
        "right": right_boxes
    }


# =========================================================
# 6. Run on all images
# =========================================================

if __name__ == "__main__":

    image_paths = glob.glob(
        "data/processed/*_denoised.png"
    )

    if not image_paths:

        print(
            "لم يتم العثور على أي صور في "
            "data/processed/"
        )

    for image_path in image_paths:

        try:

            detect_table(
                image_path
            )

        except Exception as e:

            print()
            print(
                f"Error processing {image_path}:"
            )

            print(e)