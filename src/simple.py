import cv2
#import mediapipe as mp
import svgwrite
import numpy as np


# DEFAULT_PHOTOS_DIR = "photos"
# DEFAULT_SVG_DIR = "svg"
# D_PHOTO = "photo3.png"
# D_SVG = "edges2.svg"

DEFAULT_PHOTOS_DIR = "simple_photos"
DEFAULT_SVG_DIR = "simple_svg"
D_PHOTO = "fish.png"
D_SVG = "fish.svg"


def make_svg(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = edges.shape
    dwg = svgwrite.Drawing(f"{DEFAULT_SVG_DIR}/{D_SVG}", size=(w, h))

    for cnt in contours:
        if len(cnt) > 2:
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            
            # Адаптивный epsilon на основе соотношения периметра к площади
            if area > 0:
                roundness = (perimeter * perimeter) / (4 * np.pi * area)
                if roundness > 2:  # Более вытянутая/сложная форма
                    epsilon = 0.002 * perimeter
                else:  # Более округлая форма
                    epsilon = 0.008 * perimeter
            else:
                epsilon = 0.005 * perimeter
            
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            simplified_points = [(int(p[0][0]), int(p[0][1])) for p in approx]

            if len(simplified_points) > 2:
                simplified_points.append(simplified_points[0])

            print(f"Contour: {len(simplified_points)} points, roundness: {roundness:.2f}")
            dwg.add(dwg.polyline(simplified_points, stroke='black', fill='none', stroke_width=1))

    dwg.save()
    print("✅ SVG создан!")

def adaptive_canny(image, sigma=0.82):
    """Adaptive threshold selection for Canny"""
    # calculating the median of the pixel intensities
    median = np.median(image)

    # Compute lower and upper thresholds based on the median
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    
    return cv2.Canny(image, lower, upper)

def main():
    image = cv2.imread(f"{DEFAULT_PHOTOS_DIR}/{D_PHOTO}")
    if image is None:
        print("Error: Could not read image.")
        exit()

    cv2.imshow('Input Image', image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # upgrade contrast and brightness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_image)

    # Adaptive blurring based on image size
    h, w = enhanced.shape
    ksize = max(3, min(w, h) // 200)  # Adaptive kernel size
    if ksize % 2 == 0:  # Must be odd
        ksize += 1
    
    blur = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)

    # Use adaptive Canny
    edges = adaptive_canny(blur)
    cv2.imshow('Edge Detected Image', edges)

    make_svg(edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()