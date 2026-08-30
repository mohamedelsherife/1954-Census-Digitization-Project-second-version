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
    min_speckle_size=6,          # <-- كان 4، زودناها عشان تشيل بقايا تشويش اكتر
    show_steps=False,
):
    """
    Cleans a scanned document image and prepares it for reading/extraction:
    1) Convert to grayscale
    2) Adaptive Threshold (local binarization, unaffected by dark edges)
    3) Remove noise (small scattered dots/speckles) via Connected Components

    Returns: path to the final cleaned image (denoised.png)

    ملاحظة (v2): min_speckle_size اتزودت من 4 لـ 6 كافتراضي، لأن الملاحظة
    العملية على الصور دي إن فيه بقايا نقط تشويش صغيرة بعد denoise بالقيمة
    القديمة. لو لسه فيه تشويش زيادة عندك، جرب تزودها لـ 8-10، ولو لاحظت
    إن حروف صغيرة أو نقط فوق الحروف (زي التنقيط العربي) بتتشال بالغلط،
    رجعها لـ 4-5.
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


def crop_top_margin(img_array, cutoff_y):
    """
    يقص شريط من اعلى الصورة (مثلا لازالة اثر طية/تمزق حافة الورقة
    الظاهر فوق الجدول، واللي مش جزء من بيانات الكشف نفسها).

    - cutoff_y: كل حاجة فوق السطر ده هتتقص وتتشال، وكل حاجة من السطر ده لتحت هتفضل زي ما هي

    ملحوظة: القيمة دي بتختلف من صورة لصورة حسب مكان الطية، فينفع تحددها
    يدويا لكل صورة (باستخدام show_with_grid لتحديد الاحداثية بالعين),
    او تسيبها 0 لو الصورة مفيهاش طية اصلا.
    """
    h, w = img_array.shape[:2]
    cutoff_y = max(0, min(cutoff_y, h))
    return crop_image(img_array, top=cutoff_y, bottom=h, left=0, right=w)


def enhance_clarity(img_array, scale=4.0, sharpen_amount=1.5, blur_sigma=1.3):
    """
    يوضح الكلام في الصورة (خصوصا الخط اليدوي الصغير) عن طريق:
    1) تكبير الصورة (Upscale) باستخدام INTER_LANCZOS4 - بيحافظ على حواف
       الحروف بدقة اعلى من INTER_CUBIC على الصور الثنائية اللون (ابيض/اسود)
       زي دي، ومبيدّيش الشكل المتعرج (jagged/staircase) اللي بيظهر مع CUBIC
    2) شحذ خفيف (Unsharp Masking) - بيخلي الحروف وخطوط الجدول ابين واوضح
       من غير ما يبالغ ويدي هالة (halo) حوالين الحروف

    Parameters:
    - scale: معامل التكبير (4.0 = تكبير الصورة لـ 4 اضعاف حجمها - افضل توازن
             بين الوضوح وحجم الملف الناتج بعد التجربة على عينات فعلية)
    - sharpen_amount: قوة الشحذ (اكبر = شحذ اقوى؛ القيمة الافتراضية 1.5 مجربة
                      ومناسبة لتفادي الهالات حوالين الحروف)
    - blur_sigma: درجة التمويه المستخدمة كأساس لعملية unsharp masking

    Returns: الصورة بعد التكبير والشحذ (بنفس عدد القنوات - grayscale او ملونة)

    مهم (v2): الدالة دي لازم تفضل آخر خطوة في الـ pipeline (بعد deskew وبعد
    القص النهائي)، لأن تشغيل deskew على صورة مكبّرة 4x بيبوّظ اداء
    HoughLinesP (الخطوط بتبقى ضخمة وبعيدة عن قيم الباراميترات المعتادة).
    الترتيب في crop_folder تحته بيضمن كده تلقائيًا.
    """
    h, w = img_array.shape[:2]
    new_size = (int(w * scale), int(h * scale))

    upscaled = cv2.resize(img_array, new_size, interpolation=cv2.INTER_LANCZOS4)

    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=blur_sigma)
    sharpened = cv2.addWeighted(upscaled, sharpen_amount, blurred, -(sharpen_amount - 1.0), 0)

    return sharpened


# ============================================================
# ===============  دوال تصحيح الميلان (Deskew)  ===============
# ============================================================

def detect_skew_angle(
    img_array,
    angle_limit=15,
    canny_low=50,
    canny_high=150,
    hough_threshold=70,            # <-- كان 200، قللناها عشان يكتشف خطوط جدول متقطعة/اقل كثافة
    min_line_length_ratio=0.10,    # <-- كان 0.25، قللناها لان النصف بعد split بيبقى اضيق
    max_line_gap=50,               # <-- كان تابت جوه الكود بقيمة 20، بقى بارامتر قابل للتعديل
    min_lines_required=5,          # جديد: حد ادنى لعدد الخطوط عشان نثق في النتيجة
    label="",
    verbose=True,
):
    """
    يكتشف زاوية ميلان الصورة اعتماداً على خطوط الجدول (وليس النص).

    الفكرة:
    - نستخدم Canny لايجاد الحواف
    - نستخدم HoughLinesP لايجاد الخطوط المستقيمة (خطوط الجدول)
    - نحسب زاوية كل خط، ونستبعد الخطوط اللي زاويتها بعيدة اوي عن الافقي/الرأسي
      (عشان منتأثرش بخطوط حواف الورقة الممزقة او كلام مكتوب بخط مايل)
    - نرجع متوسط (median) الزوايا القريبة من الافقي كزاوية التصحيح

    Parameters:
    - angle_limit: اقصى زاوية ميلان متوقعة (بالدرجات) - أي خط زاويته اكبر من كده يتجاهل
    - hough_threshold: عدد الاصوات (votes) المطلوب في Hough space عشان يتحسب الخط
                       موجود. (v2) اتقللت من 200 لـ 70 لان القيمة القديمة كانت
                       بترفض كل الخطوط تقريبا في صور فيها تشويش/انقطاعات كتير.
    - min_line_length_ratio: اقل طول للخط (نسبة من عرض الصورة) عشان يتحسب
                             (خطوط الجدول الافقية عادة طويلة). (v2) اتقللت من
                             0.25 لـ 0.10 عشان تناسب الانصاف الاضيق بعد split.
    - max_line_gap: اقصى فجوة (بالبكسل) بين نقطتين عشان لسه يتحسبوا خط واحد
                    متصل. (v2) بقت بارامتر بدل ما كانت قيمة تابتة جوه الكود.
    - min_lines_required: (جديد) لو عدد الخطوط المقبولة اقل من الرقم ده،
                          الدالة بترفض تاخد قرار وترجع 0.0 مع تحذير، بدل ما
                          تاخد قرار غير موثوق بناء على خط او اتنين بس.
    - label: (جديد) اسم اختياري يتطبع في رسائل الـ verbose، مفيد لما تشغل
             الدالة دي على كذا صورة وعايز تعرف كل نتيجة تخص مين.
    - verbose: (جديد) لو True، بيطبع تفاصيل تشخيصية (عدد الخطوط, مدى الزوايا..)
              مفيد جدا للتأكد ان القرار المتخذ موثوق قبل ما تتطبق على الصورة.

    Returns: angle (float) بالدرجات. موجب = ميل عكس عقارب الساعة, سالب = مع عقارب الساعة
             لو مفيش خطوط كافية اتلاقت (اقل من min_lines_required), بيرجع 0.0
    """
    prefix = f"[{label}] " if label else ""

    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    h, w = gray.shape[:2]
    min_line_length = int(w * min_line_length_ratio)

    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None or len(lines) == 0:
        if verbose:
            print(f"  {prefix}!! لم يتم العثور على خطوط كافية لتحديد الميلان - سيتم تجاهل التصحيح")
        return 0.0

    angles = []
    weights = []
    for line in lines:
        # بعض نسخ OpenCV بترجع شكل الخط كـ [[x1,y1,x2,y2]] وبعضها [x1,y1,x2,y2]
        # نستخدم reshape(-1) عشان نضمن ان الشكل دايما flat مهما كانت النسخة
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        length = float(np.hypot(dx, dy))

        # نهتم فقط بالخطوط القريبة من الافقي (خطوط الجدول الافقية)
        if abs(angle) <= angle_limit:
            angles.append(angle)
            weights.append(length)

    if not angles:
        if verbose:
            print(f"  {prefix}!! لم يتم العثور على خطوط افقية واضحة - سيتم تجاهل التصحيح")
        return 0.0

    # (v2) لو عدد الخطوط المقبولة اقل من الحد الادنى، منثقش في القرار
    if len(angles) < min_lines_required:
        if verbose:
            print(
                f"  {prefix}!! عدد الخطوط الموثوقة ({len(angles)}) اقل من الحد الادنى "
                f"({min_lines_required}) - سيتم تجاهل التصحيح لتفادي قرار غير دقيق"
            )
        return 0.0

    # نستخدم "weighted median" بدل median العادي:
    # الخطوط الطويلة (زي حدود الجدول الحقيقية) توزنها اكبر من الخطوط القصيرة
    # (زي خطوط طيات/تمزقات حواف الورقة اللي ممكن تكون بزاوية مختلفة تماما
    # وتخدع median العادي رغم انها قصيرة وغير موثوقة)
    angles_arr = np.array(angles)
    weights_arr = np.array(weights)

    order = np.argsort(angles_arr)
    angles_sorted = angles_arr[order]
    weights_sorted = weights_arr[order]
    cum_weights = np.cumsum(weights_sorted)
    cutoff = weights_sorted.sum() / 2.0
    idx = np.searchsorted(cum_weights, cutoff)
    idx = min(idx, len(angles_sorted) - 1)

    weighted_median_angle = float(angles_sorted[idx])

    if verbose:
        print(
            f"  {prefix}عدد الخطوط الموثوقة: {len(angles)} | "
            f"مدى الزوايا: [{angles_arr.min():.2f}, {angles_arr.max():.2f}] | "
            f"الزاوية المختارة (weighted median): {weighted_median_angle:.3f}°"
        )

    return weighted_median_angle


def rotate_image(img_array, angle, border_value=255):
    """
    يدور الصورة بزاوية معينة حول مركزها، مع الحفاظ على كل الصورة
    (توسيع الابعاد لو لزم الامر عشان محدش يتقص من الاطراف).

    - border_value: لون الحواف الجديدة الناتجة عن الدوران (255 = ابيض, مناسب للصور بعد denoise)
    """
    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # حساب الابعاد الجديدة عشان الصورة متتقصش من الاطراف بعد الدوران
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # تعديل مصفوفة الدوران عشان تاخد في الاعتبار الابعاد الجديدة
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        img_array,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated


def deskew_image(
    img_array,
    angle_limit=15,
    min_angle_to_apply=0.1,
    hough_threshold=70,
    min_line_length_ratio=0.10,
    max_line_gap=50,
    min_lines_required=5,
    label="",
    show_angle=True,
):
    """
    الدالة الرئيسية للتصحيح: تكتشف زاوية الميلان وتصححها.

    - min_angle_to_apply: لو الزاوية المكتشفة اصغر من كده، متعملش دوران خالص
                          (عشان منضيعش جودة الصورة بدوران غير ضروري لزاوية شبه معدومة)
    - باقي الباراميترات (hough_threshold, min_line_length_ratio, max_line_gap,
      min_lines_required, label) بتتمرر مباشرة لـ detect_skew_angle، شوف
      توثيقها هناك.

    Returns: img_array بعد التصحيح (او نفس الصورة لو الزاوية صغيرة جدا او متلاقتش)
    """
    angle = detect_skew_angle(
        img_array,
        angle_limit=angle_limit,
        hough_threshold=hough_threshold,
        min_line_length_ratio=min_line_length_ratio,
        max_line_gap=max_line_gap,
        min_lines_required=min_lines_required,
        label=label,
        verbose=show_angle,
    )

    if abs(angle) < min_angle_to_apply:
        return img_array

    return rotate_image(img_array, angle)


# ============================================================
# =====================  نهاية دوال Deskew  ===================
# ============================================================


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
    output_dir="data/cropped",
    pattern="*.png",
    needs_split=None,
    split_offsets=None,
    custom_crops=None,
    default_crops=None,
    apply_deskew=True,
    deskew_params=None,
    top_margin_crops=None,
    apply_enhance=False,
    enhance_scale=2.0,
):
    """
    Pipeline كامل للقص، بيتعامل مع كل صورة حسب حالتها:

    1) لو اسم الصورة موجود في needs_split (وقيمته True):
       - يقسمها الاول يمين/يسار (split_left_right)
       - يشيل شريط اعلى الصورة لو محدد في top_margin_crops (لازالة اثر طية/تمزق)
       - يصحح ميلان كل نصف على حدة (deskew_image) [لو apply_deskew=True]
       - وبعدين يقص كل نصف لـ header_part / table_part (او اي اجزاء تانية)

    2) لو الصورة مش محتاجة قص يمين/يسار (مش موجودة في needs_split، او قيمتها False):
       - يشيل شريط اعلى الصورة لو محدد في top_margin_crops
       - يصحح ميلان الصورة كاملة [لو apply_deskew=True]
       - يقص الصورة الكاملة مباشرة لـ header_part / table_part

    الترتيب مهم وبيفضل زي ما هو (v2): split -> crop_top_margin -> deskew ->
    crop_header_table -> enhance (اخر حاجة). تشغيل enhance قبل deskew كان
    بيبوظ اداء اكتشاف الميلان (Hough) لان الصورة بتبقى مكبرة اربع اضعاف.

    Parameters:
    - needs_split: dict {اسم_الصورة_بدون_الامتداد: True/False}
                    يحدد هل الصورة محتاجة قص يمين/يسار قبل قص header/table
    - split_offsets: dict {اسم_الصورة_بدون_الامتداد: offset}
                      لضبط نقطة المنتصف لصور معينة (اختياري)
    - custom_crops: dict {اسم_الصورة_بدون_الامتداد: crops_template}
                     احداثيات مخصصة لقص header/table لصورة (او نصف صورة) معينة
    - default_crops: crops_template افتراضي لأي صورة/نصف مش موجود في custom_crops
    - apply_deskew: لو True، يصحح ميلان كل جزء (بعد split, قبل crop_header_table)
    - deskew_params: (جديد) dict بباراميترات deskew_image الافتراضية لكل
                      الصور، مثال:
                      {"hough_threshold": 70, "min_line_length_ratio": 0.10,
                       "max_line_gap": 50, "min_lines_required": 5}
                      لو عايز قيم مختلفة لصورة معينة، استخدم
                      per_image_deskew_params بدلها.
    - top_margin_crops: dict {اسم_الجزء: cutoff_y}
                        بيشيل شريط من اعلى الجزء المحدد (بعد split, قبل deskew)
                        لازالة اثر طيات/تمزقات حواف الورقة الظاهرة فوق الجدول.
                        الاسم هنا لازم يبقى بعد الـ split لو الصورة اتقسمت
                        (يعني <original_name>_left او <original_name>_right)
                        او الاسم الاصلي لو مافيش split.
                        مثال: {"1954-P000001_page-0001_denoised_left": 190}
    - apply_enhance: لو True، يوضح الكلام (تكبير + شحذ) لكل جزء بعد القص النهائي
                     (header_part / table_part) - ده بيتطبق آخر حاجة عشان
                     منكبرش/نشحذش بيكسلات هتتقص وتتشال بعد كده
    - enhance_scale: معامل التكبير المستخدم في enhance_clarity (افتراضي 2.0)
    """
    os.makedirs(output_dir, exist_ok=True)

    if needs_split is None:
        needs_split = {}
    if split_offsets is None:
        split_offsets = {}
    if custom_crops is None:
        custom_crops = {}
    if top_margin_crops is None:
        top_margin_crops = {}
    if deskew_params is None:
        deskew_params = {
            "hough_threshold": 70,
            "min_line_length_ratio": 0.10,
            "max_line_gap": 50,
            "min_lines_required": 5,
        }
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

            left_name = f"{base_name}_left"
            right_name = f"{base_name}_right"

            # ---- شيل شريط الطية/التمزق من اعلى كل نصف (لو محدد) ----
            if left_name in top_margin_crops:
                cutoff = top_margin_crops[left_name]
                print(f"  -> قص شريط علوي من {left_name} (cutoff_y={cutoff})")
                left_half = crop_top_margin(left_half, cutoff)

            if right_name in top_margin_crops:
                cutoff = top_margin_crops[right_name]
                print(f"  -> قص شريط علوي من {right_name} (cutoff_y={cutoff})")
                right_half = crop_top_margin(right_half, cutoff)

            if apply_deskew:
                print("  -> تصحيح ميلان النصف الايسر:")
                left_half = deskew_image(left_half, label=left_name, **deskew_params)
                print("  -> تصحيح ميلان النصف الايمن:")
                right_half = deskew_image(right_half, label=right_name, **deskew_params)

            parts_to_process = {
                left_name: left_half,
                right_name: right_half,
            }
            print(f"  -> Split into left/right (offset={offset})")
        else:
            if base_name in top_margin_crops:
                cutoff = top_margin_crops[base_name]
                print(f"  -> قص شريط علوي من {base_name} (cutoff_y={cutoff})")
                img_array = crop_top_margin(img_array, cutoff)

            if apply_deskew:
                print("  -> تصحيح ميلان الصورة:")
                img_array = deskew_image(img_array, label=base_name, **deskew_params)

            parts_to_process = {
                base_name: img_array,
            }

        # ---- قص كل جزء (سواء نصف صورة او الصورة كاملة) لـ header/table ----
        for part_name, part_array in parts_to_process.items():
            crops_template = custom_crops.get(part_name, default_crops)
            sub_crops = crop_header_table(part_array, crops_template)

            for sub_name, cropped in sub_crops.items():
                if apply_enhance:
                    cropped = enhance_clarity(cropped, scale=enhance_scale)

                out_path = os.path.join(output_dir, f"{part_name}_{sub_name}.png")
                save_array_as_image(cropped, out_path)
                print(f"saved: {out_path}  |  shape: {cropped.shape[:2]}")

    print("\nDone cropping and saving all images")


if __name__ == "__main__":
    # ============================================
    # 1. تنظيف الصور (denoising) لكل الصور في data/raw
    # ============================================
    results = process_folder(
        "data/raw",
        pattern="*.jpg",
        min_speckle_size=6,   # (v2) كانت 4 - جرب 8-10 لو لسه فيه تشويش
    )
    print("Final images:", results)

    # ============================================
    # 2. قص كل الصور المعالجة في data/processed
    #    (بيتم تصحيح الميلان تلقائيا لكل جزء قبل القص)
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

    # (اختياري) شيل شريط اعلى صورة معينة (بعد split, قبل deskew) لازالة
    # اثر طيات/تمزقات حواف الورقة الظاهرة فوق الجدول.
    # حدد القيمة بالعين باستخدام show_with_grid() على الصورة، ثم اكتب هنا
    # الاسم بعد الـ split (مثال: _left / _right) لو الصورة اتقسمت
    top_margin_crops = {
        "1954-P000001_page-0001_denoised_left": 190,
        # "1954-P000001_page-0001_denoised_right": 0,  # مثال: لو مفيش طية في اليمين
    }

    # (v2 - جديد) باراميترات اكتشاف الميلان. القيم دي اتظبطت بناء على تشخيص
    # فعلي اظهر ان القيم القديمة (threshold=200, ratio=0.25) كانت بترفض
    # كل الخطوط تقريبا. لو لسه الميلان مش بيتصحح صح، قلل hough_threshold
    # اكتر (مثلا 40-50) او زود max_line_gap.
    deskew_params = {
        "hough_threshold": 70,
        "min_line_length_ratio": 0.10,
        "max_line_gap": 50,
        "min_lines_required": 5,
    }

    crop_folder(
        input_dir="data/processed",
        output_dir="data/cropped",
        pattern="*_denoised.png",
        needs_split=needs_split,
        split_offsets=split_offsets,
        custom_crops=custom_crops,
        default_crops=default_crops,
        apply_deskew=True,   # غيّرها لـ False لو عايز ترجع للسلوك القديم بدون تصحيح ميلان
        deskew_params=deskew_params,
        top_margin_crops=top_margin_crops,
        apply_enhance=True,   # يوضح الكلام (تكبير + شحذ) - غيّرها لـ False لو مش محتاجها
        enhance_scale=4.0,    # جرب 3.0 لو حجم الملفات كبير اوي وعايز توازن افضل
    )