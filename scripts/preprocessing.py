import cv2
from matplotlib import pyplot as plt
import numpy as np
image_file = "data/raw/1954-P000001_page-0001.jpg"
img = cv2.imread(image_file)

#https://stackoverflow.com/questions/28816046/
#displaying-different-images-with-actual-size-in-matplotlib-subplot
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

display(image_file)

# to make image gray
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(img)
cv2.imwrite("data/processed/gray.jpg", gray_image)

#display gray image
display("data/processed/gray.jpg")

blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
 
# Adaptive Threshold — يحسب عتبة محلية لكل منطقة صغيرة
# غير متأثر بالحواف السوداء الكبيرة حول الوثيقة (بخلاف Otsu)
im_bw = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=31,   # حجم النافذة المحلية - جرب 21 إلى 41
    C=15            # مقدار الطرح من المتوسط - جرب 10 إلى 20
)
 
cv2.imwrite("data/processed/bw_image.jpg", im_bw)
display("data/processed/bw_image.jpg")



# إزالة الضوضاء باستخدام Connected Components
# (يشيل بس البقع الصغيرة جداً، ويحافظ بدقة على الحروف)
# ============================================
 
# نعيد تثبيت الصورة كثنائية صارمة (0 أو 255) لتفادي أي تدرّج رمادي
# قد ينتج من ضغط JPEG - مهم جداً قبل استخدام connectedComponents
_, im_bw_strict = cv2.threshold(im_bw, 127, 255, cv2.THRESH_BINARY)
 
# نعكس الصورة مؤقتاً: النص أبيض على خلفية سودة
# (connectedComponents يحسب المكونات المتصلة من البكسلات البيضاء)
inverted = cv2.bitwise_not(im_bw_strict)
n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
 
min_size = 4  # أقل مساحة (بالبكسل) نعتبرها جزء من نص حقيقي
cleaned = np.zeros_like(inverted)
for i in range(1, n_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_size:
        cleaned[labels == i] = 255
 
denoised = cv2.bitwise_not(cleaned)  # نرجعها لوضعها الطبيعي (نص أسود على أبيض)
 
# نحفظ بصيغة PNG (بدون فقدان جودة) للحفاظ على ثنائية الصورة
cv2.imwrite("data/processed/denoised.png", denoised)
display("data/processed/denoised.png")



