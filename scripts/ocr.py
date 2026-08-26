import pytesseract
from PIL import Image

croped_image = Image.open('data/cropped/1954-P000002_page-0001_denoised_header_part.png')
text = pytesseract.image_to_string(croped_image, lang='ara', config='--psm 4')
print(text)