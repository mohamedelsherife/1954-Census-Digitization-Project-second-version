import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob


def display(im_path_or_array):
    """تعرض صورة سواء كانت مسار ملف أو مصفوفة numpy."""
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
    تنظّف صورة وثيقة ممسوحة (scanned) وتجهزها للقراءة/الاستخراج:
    1) تحويل لرمادي
    2) Adaptive Threshold (ثنائية محلية، غير متأثرة بالحواف السوداء)
    3) إزالة الضوضاء (النقاط الصغيرة المتناثرة) عبر Connected Components

    المعاملات (parameters):
    - image_path: مسار صورة الإدخال
    - output_dir: مجلد حفظ النتائج
    - block_size: حجم النافذة المحلية لـ adaptiveThreshold (رقم فردي)
    - c_value: القيمة المطروحة من المتوسط بـ adaptiveThreshold
    - min_speckle_size: أقل مساحة (بكسل) تعتبر جزء من نص حقيقي
    - show_steps: لو True يعرض كل خطوة بالتفصيل

    يرجع: مسار الصورة النهائية المنظّفة (denoised.png)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # قراءة الصورة
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"تعذّر قراءة الصورة: {image_path}")

    if show_steps:
        display(image_path)

    # 1) تحويل لرمادي
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_dir, f"{base_name}_gray.jpg")
    cv2.imwrite(gray_path, gray_image)
    if show_steps:
        display(gray_path)

    # 2) تنعيم خفيف + Adaptive Threshold
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

    # 3) إزالة الضوضاء عبر Connected Components
    _, im_bw_strict = cv2.threshold(im_bw, 127, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(im_bw_strict)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    cleaned = np.zeros_like(inverted)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_speckle_size:
            cleaned[labels == i] = 255

    denoised = cv2.bitwise_not(cleaned)

    # الحفظ بصيغة PNG إلزامي (JPEG يكسر ثنائية الصورة ويرجع الضوضاء)
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
    """
    تطبّق process_image على كل الصور بمجلد معين.
    مثال: process_folder("data/raw", pattern="*.jpg")
    """
    image_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not image_paths:
        print(f"لا توجد صور مطابقة للنمط {pattern} داخل {input_dir}")
        return []

    results = []
    for path in image_paths:
        print(f"معالجة: {path}")
        try:
            out_path = process_image(path, output_dir=output_dir, **kwargs)
            results.append(out_path)
            print(f"  -> تم الحفظ في: {out_path}")
        except Exception as e:
            print(f"  !! خطأ أثناء معالجة {path}: {e}")

    return results


if __name__ == "__main__":
    # process folder:
    results = process_folder("data/raw", pattern="*.jpg")

    print("الصورة النهائية:", results)

    # مثال 2: معالجة كل صور مجلد كامل (فعّليه عند الحاجة)
    # results = process_folder("data/raw", pattern="*.jpg")
    # print(f"تمت معالجة {len(results)} صورة")