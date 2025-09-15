import cv2, os
from PIL import Image
from rembg import remove

image_path = "photos/photo2.jpg"

image = cv2.imread(image_path)

no_bckg = remove(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
mask = thresh.copy()

image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
mask_pil = Image.fromarray(mask)

image_pil.putalpha(mask_pil)
filename, ext = os.path.splitext(image_path)
output_path = f"{filename}_no_bg.png"
no_bckg.save(output_path)
