<<<<<<< HEAD
import cv2
import easyocr
import matplotlib.pyplot as plt



#read image should work on any device
image= cv2.imread('data/processed/1954-P000002_page-0001_denoised.png')
img = cv2.imread('image')




#the reading code 
reader = easyocr.Reader(['arb'],)
reader.readtext(img)
=======
import pytesseract
from PIL import Image

croped_image = Image.open('data/cropped/1954-P000002_page-0001_denoised_header_part.png')
text = pytesseract.image_to_string(croped_image, lang='ara', config='--psm 4')
print(text)
>>>>>>> 446107fa8728c5a72be62f7a106a3da39560e9c3
