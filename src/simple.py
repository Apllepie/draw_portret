import cv2
import mediapipe as mp
import svgwrite

DEFAULT_PHOTOS_DIR = "photos"
DEFAULT_SVG_DIR = "svg"
D_PHOTO = "photo4.jpg"
D_SVG = "edges2.svg"

def make_svg(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = edges.shape
    dwg = svgwrite.Drawing(f"{DEFAULT_SVG_DIR}/{D_SVG}", size=(w, h))

    for cnt in contours:
        points = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        if len(points) > 1:
            dwg.add(dwg.polyline(points, stroke='black', fill='none', stroke_width=1))

    dwg.save()
    print("✅ SVG создан!")

def main():
    image = cv2.imread(f"{DEFAULT_PHOTOS_DIR}/{D_PHOTO}")
    if image is None:
        print("Error: Could not read image.")
        exit()

    cv2.imshow('Input Image', image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('Grayscale Image', gray_image)

    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

    edges = cv2.Canny(blur, 100, 200)
    cv2.imshow('Edge Detected Image', edges)

    make_svg(edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()