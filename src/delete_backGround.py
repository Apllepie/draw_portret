import cv2, os
from PIL import Image
from rembg import remove


image_path = "simple_photos/czapka.png"

image = cv2.imread(image_path)

no_bckg = remove(image)
cv2.imshow("No Background",no_bckg)

filename, ext = os.path.splitext(image_path)
output_path = f"{filename}_no_bg.png"

cv2.imwrite(output_path, no_bckg)
