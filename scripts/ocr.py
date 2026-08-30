import pytesseract
from PIL import Image

<<<<<<< HEAD
croped_image = Image.open('data/cropped/1954-P000002_page-0001_denoised_header_part.png')
text = pytesseract.image_to_string(croped_image, lang='ara', config='--psm 4')
print(text)
import pytesseract
from PIL import Image

cropped_image = Image.open(
    'data/cropped/1954-P000002_page-0001_denoised_header_part.png'
)
text = pytesseract.image_to_string(cropped_image, lang='ara', config='--psm 4')
print(text)

=======
# بدل ما تقرأ الصورة كاملة، اقرأ كل خانة على حدة
regions = {
    "city_name": (1750, 250, 2100, 300),   # (left, top, right, bottom) تقريبية
    "center_name": (1500, 250, 1750, 300),
    "village_name": (1250, 250, 1500, 300),
}

img = Image.open("data/cropped/1954-P000002_page-0001_denoised_header_part.png")

for name, box in regions.items():
    cropped = img.crop(box)
    text = pytesseract.image_to_string(cropped, lang="ara", config="--psm 7")
    print(f"{name}: {text.strip()}")
>>>>>>> 2b9bb2cc0fc630c64ee6527de8772b38b7c16856
