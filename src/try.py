import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
PHOTO_PATH = "simple_photos/fish.png"
#PHOTO_PATH = "simple_photos/square.png"
#PHOTO_PATH = "simple_photos/photo3_with_hat.png"
LS_PATH = "LS/DRAW.LS"

# Image processing
SHAPE_LENGTH_THRESHOLD = 10
MAX_POINTS_PER_SHAPE = 400
ARC_STEP_MM = 0.8
MAX_SEG_LEN_MM = 25.0
COLLINEAR_ANGLE_DEG = 4.0

# FANUC parameters
SCALE_FACTOR = 0.1  # px -> mm
Z_UP, Z_DOWN = 10.0, 0.0
FEED_RATE = 200
BASE_X, BASE_Y = 0.0, 0.0
UT, UF = 9, 4


def resample_contour(pts, step_px):
    """Uniform point sampling along arc length."""
    if len(pts) < 2:
        return pts
    
    result = [pts[0]]
    accumulated = 0.0
    
    for i in range(1, len(pts)):
        segment = pts[i] - pts[i-1]
        seg_len = np.linalg.norm(segment)
        
        while accumulated + seg_len >= step_px and seg_len > 0:
            t = (step_px - accumulated) / seg_len
            new_point = pts[i-1] + segment * t
            result.append(new_point)
            pts[i-1] = new_point
            segment = pts[i] - pts[i-1]
            seg_len = np.linalg.norm(segment)
            accumulated = 0.0
        
        accumulated += seg_len
    
    return np.array(result, dtype=np.float32)


def limit_segment_length(pts, max_len_px):
    """Splits long segments into parts."""
    if len(pts) < 2:
        return pts
    
    result = [pts[0]]
    for i in range(1, len(pts)):
        p0, p1 = result[-1], pts[i]
        dist = np.linalg.norm(p1 - p0)
        
        if dist <= max_len_px:
            result.append(p1)
        else:
            n_segments = int(np.ceil(dist / max_len_px))
            for k in range(1, n_segments + 1):
                result.append(p0 + (p1 - p0) * (k / n_segments))
    
    return np.array(result, dtype=np.float32)


def merge_collinear(pts, angle_thr_deg):
    """Merges collinear segments."""
    if len(pts) <= 2:
        return pts
    
    result = [pts[0]]
    curr = pts[1]
    prev_angle = np.arctan2(*(pts[1] - pts[0])[::-1])
    
    for i in range(2, len(pts)):
        angle = np.arctan2(*(pts[i] - curr)[::-1])
        angle_diff = np.rad2deg(np.arctan2(np.sin(angle - prev_angle), 
                                           np.cos(angle - prev_angle)))
        
        if abs(angle_diff) < angle_thr_deg:
            curr = pts[i]
        else:
            result.append(curr)
            curr = pts[i]
            prev_angle = angle
    
    result.append(curr)
    return np.array(result, dtype=np.int32)


def extract_contours(edge_image):
    """Extracts and processes contours."""
    contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    shapes = []
    step_px = ARC_STEP_MM / SCALE_FACTOR
    max_seg_px = MAX_SEG_LEN_MM / SCALE_FACTOR
    
    for cnt in contours:
        if len(cnt) <= SHAPE_LENGTH_THRESHOLD:
            continue
        
        # Process contour
        pts = cnt[:, 0, :]
        pts = resample_contour(pts.astype(np.float32), step_px)
        pts = merge_collinear(pts.astype(np.int32), COLLINEAR_ANGLE_DEG)
        pts = limit_segment_length(pts.astype(np.float32), max_seg_px).astype(np.int32)
        
        # Limit number of points
        if len(pts) > MAX_POINTS_PER_SHAPE:
            stride = int(np.ceil(len(pts) / MAX_POINTS_PER_SHAPE))
            pts = pts[::stride]
        
        shapes.extend([(int(y), int(x)) for x, y in pts])
        shapes.append("/")
    
    return shapes


def save_fanuc_program(shapes, output_path):
    """Saves program in FANUC LS format."""
    positions = []
    
    with open(output_path, 'w') as f:
        # Header
        f.write("/PROG DRAW\n/ATTR\nOWNER\t\t= MNEDITOR;\n")
        f.write("COMMENT\t\t= \"Portrait Drawing\";\n/MN\n")
        
        line_num, pos_num, shape_idx = 1, 1, 0
        f.write(f"   {line_num}:   UFRAME_NUM={UF};\n"); line_num += 1
        f.write(f"   {line_num}:   UTOOL_NUM={UT};\n"); line_num += 1
        
        # Process contours
        current_shape = []
        for point in shapes:
            if point == "/":
                if len(current_shape) >= 2:
                    shape_idx += 1
                    f.write(f"   {line_num}:  !Shape {shape_idx} ;\n"); line_num += 1
                    
                    # Move to first point + lower pen
                    y0, x0 = current_shape[0]
                    x, y = BASE_X + x0 * SCALE_FACTOR, BASE_Y + y0 * SCALE_FACTOR
                    
                    f.write(f"   {line_num}:L P[{pos_num}] {FEED_RATE}mm/sec CNT0 ;\n")
                    positions.append([x, y, Z_UP]); line_num += 1; pos_num += 1
                    
                    f.write(f"   {line_num}:L P[{pos_num}] {FEED_RATE//2}mm/sec CNT0 ;\n")
                    positions.append([x, y, Z_DOWN]); line_num += 1; pos_num += 1
                    
                    # Draw contour (starting from second point)
                    for yy, xx in current_shape[1:]:
                        x, y = BASE_X + xx * SCALE_FACTOR, BASE_Y + yy * SCALE_FACTOR
                        f.write(f"   {line_num}:L P[{pos_num}] {FEED_RATE//4}mm/sec CNT0 ;\n")
                        positions.append([x, y, Z_DOWN]); line_num += 1; pos_num += 1
                    
                    # Lift pen (no duplicate position)
                    x_last, y_last = positions[-1][0], positions[-1][1]  # Take last position
                    f.write(f"   {line_num}:L P[{pos_num}] {FEED_RATE//2}mm/sec CNT0 ;\n")
                    positions.append([x_last, y_last, Z_UP]); line_num += 1; pos_num += 1
                
                current_shape = []
            else:
                current_shape.append(point)
        
        # Return to start
        if positions:
            f.write(f"   {line_num}:J P[1] {FEED_RATE}% CNT0 ;\n"); line_num += 1
        
        f.write(f"   {line_num}:  !End of program ;\n/POS\n")
        
        # Positions
        for i, (x, y, z) in enumerate(positions, 1):
            f.write(f"P[{i}]{{\n\tGP1:\n")
            f.write(f"\t UF : {UF}, UT : {UT},\tCONFIG : 'F U T, 0, 0, 0',\n")
            f.write(f"\t X = {x:.3f} mm,\tY = {y:.3f} mm,\tZ = {z:.3f} mm,\n")
            f.write("\t W = 59.855 deg,\tP = -83.452 deg,\tR = -153.620 deg\n};\n")
        
        f.write("/END\n")
    
    print(f"✅ Program saved: {output_path} ({shape_idx} contours)")
    return positions


def visualize_path(positions, animate=True, interval=20):
    """Visualizes robot path."""
    if not positions:
        return
    
    pts = np.array(positions)
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    z_threshold = (Z_DOWN + Z_UP) / 2
    
    fig, ax = plt.subplots(figsize=(10, 8))
    pad = 5.0
    ax.set_xlim(X.min() - pad, X.max() + pad)
    ax.set_ylim(Y.min() - pad, Y.max() + pad)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Robot Path")
    
    draw_line, = ax.plot([], [], 'r-', lw=1.8, label="Drawing")
    travel_line, = ax.plot([], [], 'gray', lw=1.2, ls='--', alpha=0.7, label="Travel")
    tip, = ax.plot([], [], 'bo', ms=6, label="Tool")
    ax.legend()
    
    draw_x, draw_y, travel_x, travel_y = [], [], [], []
    
    # FIXED: segment draws only if BOTH points are pen down
    segments = [(pts[i, 0], pts[i, 1], pts[i+1, 0], pts[i+1, 1],
                 pts[i, 2] <= z_threshold and pts[i+1, 2] <= z_threshold)
                for i in range(len(pts) - 1)]
    
    def update(frame):
        x0, y0, x1, y1, is_draw = segments[frame]
        
        if is_draw:
            # Add break if this is start of new line
            if draw_x and (frame == 0 or not segments[frame-1][4]):
                draw_x.append(np.nan)
                draw_y.append(np.nan)
            draw_x.extend([x0, x1])
            draw_y.extend([y0, y1])
            draw_line.set_data(draw_x, draw_y)
        else:
            # Add break if this is start of new travel
            if travel_x and frame > 0 and segments[frame-1][4]:
                travel_x.append(np.nan)
                travel_y.append(np.nan)
            travel_x.extend([x0, x1])
            travel_y.extend([y0, y1])
            travel_line.set_data(travel_x, travel_y)
        
        tip.set_data([x1], [y1])
        return draw_line, travel_line, tip
    
    if animate:
        anim = animation.FuncAnimation(fig, update, frames=len(segments), 
                                      interval=interval, blit=True, repeat=False)
        plt.show()
    else:
        # Non-animated mode - draw everything at once
        for frame in range(len(segments)):
            update(frame)
        plt.show()


def main():
    # Load and process image
    image = cv2.imread(PHOTO_PATH, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Error loading: {PHOTO_PATH}")
        return
    
    image = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(image, 50, 150)
    
    cv2.imshow("Original", image)
    cv2.imshow("Contours", edges)
    
    # Generate trajectory
    shapes = extract_contours(edges)
    print(f"Found points: {len([p for p in shapes if p != '/'])}")
    
    # Save and visualize
    positions = save_fanuc_program(shapes, LS_PATH)
    visualize_path(positions)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()