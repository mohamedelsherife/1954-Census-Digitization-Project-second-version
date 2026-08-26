import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from PIL import Image


def display(im_path_or_array):
    """Displays an image, whether it's a file path or a numpy array."""
    dpi = 80
    if isinstance(im_path_or_array, str):
        im_data = plt.imread(im_path_or_array)
    else:
        im_data = im_path_or_array

    height, width = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()


def process_image(
    image_path,
    output_dir="data/processed",
    block_size=31,
    c_value=15,
    min_speckle_size=4,
    show_steps=False,
):
    """
    Cleans a scanned document image and prepares it for reading/extraction:
    1) Convert to grayscale
    2) Adaptive Threshold (local binarization, unaffected by dark edges)
    3) Remove noise (small scattered dots/speckles) via Connected Components

    Returns: path to the final cleaned image (denoised.png)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    if show_steps:
        display(image_path)

    # 1) Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_dir, f"{base_name}_gray.jpg")
    cv2.imwrite(gray_path, gray_image)
    if show_steps:
        display(gray_path)

    # 2) Light smoothing + Adaptive Threshold
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    im_bw = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c_value,
    )
    bw_path = os.path.join(output_dir, f"{base_name}_bw.jpg")
    cv2.imwrite(bw_path, im_bw)
    if show_steps:
        display(bw_path)

    # 3) Remove noise via Connected Components
    _, im_bw_strict = cv2.threshold(im_bw, 127, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(im_bw_strict)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    cleaned = np.zeros_like(inverted)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_speckle_size:
            cleaned[labels == i] = 255

    denoised = cv2.bitwise_not(cleaned)

    denoised_path = os.path.join(output_dir, f"{base_name}_denoised.png")
    cv2.imwrite(denoised_path, denoised)
    if show_steps:
        display(denoised_path)

    return denoised_path


def process_folder(
    input_dir,
    output_dir="data/processed",
    pattern="*.jpg",
    **kwargs,
):
    """Applies process_image to all images in a given folder."""
    image_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not image_paths:
        print(f"No images matching pattern {pattern} found in {input_dir}")
        return []

    results = []
    for path in image_paths:
        print(f"Processing: {path}")
        try:
            out_path = process_image(path, output_dir=output_dir, **kwargs)
            results.append(out_path)
            print(f"  -> Saved to: {out_path}")
        except Exception as e:
            print(f"  !! Error while processing {path}: {e}")

    return results


def load_image_as_array(path):
    """تحميل الصورة وتحويلها الى numpy array"""
    img = Image.open(path)
    return np.array(img)


def crop_image(img_array, top, bottom, left, right):
    """
    قص جزء من الصورة بالاحداثيات (بالبكسل)
    top, bottom : حدود القص على المحور الرأسي (y)
    left, right : حدود القص على المحور الافقي (x)
    """
    return img_array[top:bottom, left:right]


def save_array_as_image(img_array, out_path):
    """حفظ numpy array كصورة"""
    Image.fromarray(img_array).save(out_path)


def show_with_grid(img_array, step=100, figsize=(15, 10)):
    """عرض الصورة مع خطوط شبكة كل 'step' بكسل لتحديد الاحداثيات بالعين"""
    h, w = img_array.shape[:2]
    plt.figure(figsize=figsize)
    plt.imshow(img_array, cmap="gray")
    plt.xticks(np.arange(0, w, step), rotation=90)
    plt.yticks(np.arange(0, h, step))
    plt.grid(color="red", linestyle="--", linewidth=0.5)
    plt.show()


def split_left_right(img_array, mid_offset=0):
    """
    يقص الصورة الى نصفين: يمين ويسار، بناءً على نقطة المنتصف.

    - mid_offset: قيمة تصحيح يدوية (بالبكسل) لتحريك نقطة القص
                  يمين (+) أو يسار (-) لو الصفحة مش مظبوطة بالظبط في النص

    Returns: (left_half, right_half)
    """
    h, w = img_array.shape[:2]
    mid = (w // 2) + mid_offset

    left_half = crop_image(img_array, top=0, bottom=h, left=0, right=mid)
    right_half = crop_image(img_array, top=0, bottom=h, left=mid, right=w)

    return left_half, right_half


def crop_header_table(img_array, crops_template):
    """
    يقص صورة واحدة (سواء صفحة كاملة او نصف صفحة) الى اجزائها
    (header_part, table_part, ...) حسب crops_template.

    crops_template format:
        { "part_name": (top, bottom, left, right), ... }
    استخدم None لأي قيمة عشان يتحسب تلقائيا.

    Returns: dict { "part_name": cropped_array, ... }
    """
    h, w = img_array.shape[:2]
    results = {}

    for name, (top, bottom, left, right) in crops_template.items():
        resolved_top = top if top is not None else 0
        resolved_bottom = bottom if bottom is not None else h
        resolved_left = left if left is not None else 0
        resolved_right = right if right is not None else w

        results[name] = crop_image(
            img_array, resolved_top, resolved_bottom, resolved_left, resolved_right
        )

    return results


def crop_folder(
    input_dir,
    output_dir="data/croped",
    pattern="*.png",
    needs_split=None,
    split_offsets=None,
    custom_crops=None,
    default_crops=None,
):
    """
    Pipeline كامل للقص، بيتعامل مع كل صورة حسب حالتها:

    1) لو اسم الصورة موجود في needs_split (وقيمته True):
       - يقسمها الاول يمين/يسار (split_left_right)
       - وبعدين يقص كل نصف لـ header_part / table_part (او اي اجزاء تانية)

    2) لو الصورة مش محتاجة قص يمين/يسار (مش موجودة في needs_split، او قيمتها False):
       - يقص الصورة الكاملة مباشرة لـ header_part / table_part

    Parameters:
    - needs_split: dict {اسم_الصورة_بدون_الامتداد: True/False}
                    يحدد هل الصورة محتاجة قص يمين/يسار قبل قص header/table
    - split_offsets: dict {اسم_الصورة_بدون_الامتداد: offset}
                      لضبط نقطة المنتصف لصور معينة (اختياري)
    - custom_crops: dict {اسم_الصورة_بدون_الامتداد: crops_template}
                     احداثيات مخصصة لقص header/table لصورة (او نصف صورة) معينة
    - default_crops: crops_template افتراضي لأي صورة/نصف مش موجود في custom_crops
    """
    os.makedirs(output_dir, exist_ok=True)

    if needs_split is None:
        needs_split = {}
    if split_offsets is None:
        split_offsets = {}
    if custom_crops is None:
        custom_crops = {}
    if default_crops is None:
        default_crops = {
            "header_part": (0, 750, 0, None),
            "table_part":  (750, None, 0, None),
        }

    image_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not image_paths:
        print(f"No images matching pattern {pattern} found in {input_dir}")
        return

    print(f"Found {len(image_paths)} image(s) to crop")

    for input_path in image_paths:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        img_array = load_image_as_array(input_path)
        h, w = img_array.shape[:2]
        print(f"\nProcessing: {input_path}")
        print(f"image shape:   {h} x {w}")

        # ---- تحديد هل هذه الصورة محتاجة قص يمين/يسار ----
        if needs_split.get(base_name, False):
            offset = split_offsets.get(base_name, 0)
            left_half, right_half = split_left_right(img_array, mid_offset=offset)
            parts_to_process = {
                f"{base_name}_left": left_half,
                f"{base_name}_right": right_half,
            }
            print(f"  -> Split into left/right (offset={offset})")
        else:
            parts_to_process = {
                base_name: img_array,
            }

        # ---- قص كل جزء (سواء نصف صورة او الصورة كاملة) لـ header/table ----
        for part_name, part_array in parts_to_process.items():
            crops_template = custom_crops.get(part_name, default_crops)
            sub_crops = crop_header_table(part_array, crops_template)

            for sub_name, cropped in sub_crops.items():
                out_path = os.path.join(output_dir, f"{part_name}_{sub_name}.png")
                save_array_as_image(cropped, out_path)
                print(f"saved: {out_path}  |  shape: {cropped.shape[:2]}")

    print("\nDone cropping and saving all images")


if __name__ == "__main__":
    # ============================================
    # 1. تنظيف الصور (denoising) لكل الصور في data/raw
    # ============================================
    results = process_folder("data/raw", pattern="*.jpg")
    print("Final images:", results)

    # ============================================
    # 2. قص كل الصور المعالجة في data/processed
    # ============================================

    # حدد هنا فقط الصور اللي محتاجة قص يمين/يسار (صفحتين مفتوحتين/spread)
    # اي صورة مش مكتوبة هنا هتتعامل تلقائيًا كصفحة واحدة (بدون قص يمين/يسار)
    needs_split = {
        "1954-P000001_page-0001_denoised": True,   # صفحتين مفتوحتين -> هتتقسم يمين/يسار
        # "1954-P000002_page-0001_denoised" مش موجودة هنا -> هتتعامل كصفحة واحدة تلقائيًا
    }

    # (اختياري) تعديل نقطة المنتصف لصور معينة لو الانحناء مش بالظبط في النص
    split_offsets = {
        # "1954-P000001_page-0001_denoised": 20,
    }

    # احداثيات قص header/table مخصصة لكل جزء
    # - للصور اللي اتقسمت: الاسم بيبقى <original_name>_left او <original_name>_right
    # - للصور اللي ماتقسمتش: الاسم بيفضل زي ما هو <original_name>
    custom_crops = {
        "1954-P000001_page-0001_denoised_left": {
            "header_part": (0, 700, 0, None),
            "table_part":  (700, None, 0, None),
        },
        "1954-P000001_page-0001_denoised_right": {
            "header_part": (0, 700, 0, None),
            "table_part":  (700, None, 0, None),
        },
        "1954-P000002_page-0001_denoised": {
            "header_part": (0, 750, 0, None),
            "table_part":  (750, None, 0, None),
        },
    }

    # قص افتراضي لأي جزء مش موجود في custom_crops
    default_crops = {
        "header_part": (0, 750, 0, None),
        "table_part":  (750, None, 0, None),
    }

    crop_folder(
        input_dir="data/processed",
        output_dir="data/croped",
        pattern="*_denoised.png",
        needs_split=needs_split,
        split_offsets=split_offsets,
        custom_crops=custom_crops,
        default_crops=default_crops,
    )