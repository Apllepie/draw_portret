import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

PATH_ADD = "simple_photos/hat.png"
PATH_BASE = "simple_photos/photo3.png"
#PATH_BASE = "photos/photo4.jpg"


def read_image(path):
    if os.path.exists(path):
        # read as color image, then convert BGR->RGB for matplotlib
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Failed to read '{path}'.")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

def find_head(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
    new = image.copy()
    for (x, y, w, h) in faces:
        
        cv2.rectangle(new, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return new, faces

def add_hat_to_head(base, add, faces):
    for (x, y, w, h) in faces:
        # Resize the hat to fit the width of the detected face
        scale_factor = 2  # Scale factor to make the hat larger than the face width
        hat_width = int(w * scale_factor)  
        hat_height = int(add.shape[0] * (hat_width / add.shape[1]))  
        
        resized_hat = cv2.resize(add, (hat_width, hat_height)) 

        # Calculate position: place the hat slightly above the detected face
        # Центрируем шляпу по лицу
        face_center_x = x + w // 2
        x1 = face_center_x - hat_width // 2
        x2 = x1 + hat_width
        y1 = y - hat_height + hat_height // 5  # Смещаем вверх на половину высоты шляпы
        y2 = y1 + hat_height

        # Проверяем границы и расширяем изображение если нужно
        expand_top = max(0, -y1)
        expand_bottom = max(0, y2 - base.shape[0])
        expand_left = max(0, -x1)
        expand_right = max(0, x2 - base.shape[1])

        # Расширяем изображение если нужно
        if expand_top > 0 or expand_bottom > 0 or expand_left > 0 or expand_right > 0:
            base = cv2.copyMakeBorder(base, 
                                    top=expand_top,
                                    bottom=expand_bottom, 
                                    left=expand_left,
                                    right=expand_right,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=[255, 255, 255])
            
            # Корректируем координаты после расширения
            x1 += expand_left
            x2 += expand_left
            y1 += expand_top
            y2 += expand_top

        # Убеждаемся, что координаты в пределах изображения
        x1 = max(0, x1)
        y1 = max(0, y1) 
        x2 = min(base.shape[1], x2)
        y2 = min(base.shape[0], y2)

        # Проверяем, что область имеет положительные размеры
        if x2 <= x1 or y2 <= y1:
            continue

        # Обрезаем шляпу под размер области
        actual_width = x2 - x1
        actual_height = y2 - y1
        hat_cropped = cv2.resize(resized_hat, (actual_width, actual_height))

        # Create a mask of the hat and its inverse mask
        gray_hat = cv2.cvtColor(hat_cropped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_hat, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the area of the hat in the base image
        roi = base[y1:y2, x1:x2]
        
        # Проверяем совпадение размеров
        if roi.shape[:2] != mask.shape:
            print(f"Size mismatch: ROI {roi.shape}, mask {mask.shape}")
            continue
            
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only region of hat from hat image.
        img_fg = cv2.bitwise_and(hat_cropped, hat_cropped, mask=mask)

        # Put hat in ROI and modify the base image
        dst = cv2.add(img_bg, img_fg)
        base[y1:y2, x1:x2] = dst

    return base




def main():
    base = read_image(PATH_BASE)
    add = read_image(PATH_ADD)
    if base is None or add is None:
        return

    detected, faces = find_head(base)
    if len(detected) == 0:
        print("No faces detected.")
        return
    # display base and add images side by side

    with_hat = add_hat_to_head(base.copy(), add, faces)
    root = os.path.splitext(PATH_BASE)[0]
    cv2.imwrite(f"{root}_with_hat.png", cv2.cvtColor(with_hat, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 2, 1)
    plt.imshow(base)
    plt.title("Base")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(add)
    plt.title("Add")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(detected)
    plt.title("Detected Faces")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(with_hat)
    plt.title("With Hat")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"✅ Done! add = {add.shape}")


if __name__ == "__main__":
    main()