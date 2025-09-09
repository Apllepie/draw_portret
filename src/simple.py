import cv2
import mediapipe as mp
import svgwrite

image = cv2.imread('photos/photo4.jpg')
if image is None:
    print("Error: Could not read image.")
    exit()

cv2.imshow('Input Image', image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cv2.imshow('Grayscale Image', gray_image)

edges = cv2.Canny(gray_image, 100, 200)
cv2.imshow('Edge Detected Image', edges)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

h, w = edges.shape
dwg = svgwrite.Drawing("svg/edges2.svg", size=(w, h))

for cnt in contours:
    points = [(int(p[0][0]), int(p[0][1])) for p in cnt]
    if len(points) > 1:
        dwg.add(dwg.polyline(points, stroke='black', fill='none', stroke_width=1))

dwg.save()
print("✅ SVG создан!")

cv2.waitKey(0)
cv2.destroyAllWindows()