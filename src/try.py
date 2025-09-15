import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

#params
photo_path = "photos/photo2.jpg"
#photo_path = "simple_photos/fish.png"
LS_path = "LS/DRAW.LS"
SHOW_IMAGE = True
SHAPE_LENGTH_THRESHOLD = 10  # minimum number of points to consider a shape

def make_shapes(edge):
    """finds all shapes in the edge image and returns a list of points"""
    shape = []
    visited = np.zeros_like(edge, dtype=bool)
    height, width = edge.shape
    
    def bfs(start_x, start_y):
        """Breadth-First Search to find all connected points of a shape"""
        current_shape = []
        queue = deque([(start_x, start_y)])
        
        while queue:
            x, y = queue.popleft()
            
            if (x < 0 or x >= height or y < 0 or y >= width or 
                visited[x, y] or edge[x, y] == 0):
                continue
            
            visited[x, y] = True
            current_shape.append((x, y))
            
            # Check all 8 directions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < height and 0 <= new_y < width and 
                        not visited[new_x, new_y] and edge[new_x, new_y] != 0):
                        queue.append((new_x, new_y))
        
        return current_shape

    # Find all connected components
    for i in range(height):
        for j in range(width):
            if edge[i, j] != 0 and not visited[i, j]:
                current_shape = bfs(i, j)
                
                # Filter out small shapes
                if len(current_shape) > SHAPE_LENGTH_THRESHOLD:
                    shape.extend(current_shape)
                    shape.append("/")
    
    return shape

def check_graphicaly(shape):
    """draws the found shapes using matplotlib"""
    plt.figure(figsize=(12, 8))
    
    x = []
    y = []
    colors = plt.cm.tab10(np.linspace(0, 1, 20))
    figure_n = 0
    
    for point in shape:
        if point == "/":
            if x and y:
                color = colors[figure_n % len(colors)]
                plt.scatter(y, x, s=1, color=color, alpha=0.7)
                figure_n += 1
            x = []
            y = []
        else:
            px, py = point
            x.append(px)
            y.append(py)
    
    if x and y:
        color = colors[figure_n % len(colors)]
        plt.scatter(y, x, s=1, color=color, alpha=0.7)
        figure_n += 1
    
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.title(f"Found {figure_n} shapes")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print(f"Total shapes found: {figure_n}")

def main():
    image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {photo_path}")
        return
    
    # blur and edge detection
    image = cv2.GaussianBlur(image, (3, 3), 0)
    edge = cv2.Canny(image, 50, 150)

    if SHOW_IMAGE:
        cv2.imshow("Original", image)
        cv2.imshow("Edges", edge)

    shape = make_shapes(edge.copy())
    print(f"Total points found: {len([p for p in shape if p != '/'])}")
    check_graphicaly(shape)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()