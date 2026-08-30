"""
htr.py — الاتصال الفعلي بموديل Kraken (Muharaf)
====================================================
يستقبل صورة خلية *جاهزة* (بعد prepare_cell)، ويرجّع النص المقروء.
"""

import cv2
from PIL import Image
from kraken.containers import Segmentation, BBoxLine
from kraken.lib import models
from kraken import rpred

# ===== عدّلوا هذا المسار حسب مكان الملف عندكم فعلياً =====
MODEL_PATH = "muharaf_rec_best/muharaf_rec_best.mlmodel"

# يتحمّل مرة وحدة بس وقت استيراد الملف — مو كل مرة تستدعون الدالة
# (تحميل الموديل بطيء، ما نبي نكرره لكل خلية)
print("جاري تحميل موديل Kraken...")
rec_model = models.load_any(MODEL_PATH)
print("تم تحميل الموديل.")


def predict_handwritten_text(cell_img) -> str:
    """
    cell_img: صورة OpenCV (ndarray) — نفس الصورة الجاهزة اللي
    ترجعها prepare_cell["image"]، مو مسار ملف.
    """
    # حوّل من OpenCV (BGR) لـ PIL (RGB) — Kraken يتوقع PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))

    # ابنوا "خط" وهمي يغطي الصورة كاملة — بما إنها أصلاً خلية وحدة مقصوصة
    line = BBoxLine(
        id="cell",
        bbox=(0, 0, pil_img.width, pil_img.height),
        text_direction="horizontal-rl",  # عربي: يمين لليسار
    )
    seg = Segmentation(
        type="bbox",
        imagename="cell",
        text_direction="horizontal-rl",
        script_detection=False,
        lines=[line],
    )

    records = list(rpred.rpred(network=rec_model, im=pil_img, bounds=seg))
    return records[0].prediction if records else ""
from pathlib import Path
import cv2
from PIL import Image
from kraken.lib import models
from kraken.rpred import rpred

# تحديد مسار النموذج تلقائياً
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "muharaf_rec_best" / "muharaf_rec_best.mlmodel"

# تحميل النموذج عند استيراد الملف
print("جاري تحميل نموذج Kraken...")
htr_model = models.load_any(str(MODEL_PATH))

from pathlib import Path
import cv2
from PIL import Image
from kraken.lib import models
from kraken.rpred import rpred
from kraken.containers import Segmentation, BBoxLine

# تحديد مسار النموذج
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "muharaf_rec_best" / "muharaf_rec_best.mlmodel"

print("جاري تحميل نموذج Kraken...")
htr_model = models.load_any(str(MODEL_PATH))

def predict_cell_text(cell_crop):
    """
    تستقبل صورة الخلية وتستخرج النص العربي باستخدام كائن Segmentation المعتمد
    """
    if cell_crop is None or cell_crop.size == 0:
        return ""

    # تحويل الصورة من BGR إلى RGB
    if len(cell_crop.shape) == 3:
        cell_crop = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(cell_crop)
    width, height = pil_img.size

    # إنشاء كائن التقسيم لتفادي خطأ AttributeError
    bounds = Segmentation(
        type='bbox',
        imagename='',
        text_direction='horizontal-tb',
        script_detection=False,
        lines=[BBoxLine(id='line_0', bbox=(0, 0, width, height))]
    )

    # تشغيل عملية التعرف
    prediction = rpred(htr_model, pil_img, bounds)
    recognized_text = "".join([rec.prediction for rec in prediction])
    return recognized_text.strip()