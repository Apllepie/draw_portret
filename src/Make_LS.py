import xml.etree.ElementTree as ET


INPUT_SVG = 'svg/edges2.svg'
OUTPUT_LS = 'LS/DRAW.LS'

# INPUT_SVG = 'simple_svg/fish.svg' 
# OUTPUT_LS = 'LS/DRAW.LS'

SCALE = 0.25  # Scale factor for coordinates
offset_x = 100  # Offset for x coordinates
offset_y = 100  # Offset for y coordinates
z_up = 50  # Z height for travel moves
z_down = 0  # Z height for drawing moves

JOINT_SPEED = 150  # Joint speed 

INVERT_Y = True  # Flag for invert Y coordinate


def take_points_from_svg():
    tree = ET.parse(INPUT_SVG)
    if tree is None:
        print("Error: Could not read SVG.")
        exit()
    root = tree.getroot()

    # take height for invert y
    svg_height = float(root.get('height', 600))
    
    all_points = []


    for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
        points_str = polyline.get('points')
        print(f"Points string: {points_str}")
        if points_str:
            points = []
            for point in points_str.strip().split(' '):
                if ',' in point:
                    x_str, y_str = point.split(',')
                    try:
                        x = float(x_str) * SCALE + offset_x
                        y_coord = float(y_str)
                        
                        # Инвертируем Y если нужно
                        if INVERT_Y:
                            y_coord = svg_height - y_coord
                            
                        y = y_coord * SCALE + offset_y
                        points.append((x, y))
                    except ValueError:
                        continue
            if points:
                all_points.append(points)
    print (f"points : {len(all_points)}")
    return all_points

def write_ls_file(all_points):
    ls_line = []
    ls_line.append("/PROG DRAW")
    ls_line.append("/ATTR")
    ls_line.append("/OWNER       = MNEDITOR;")
    ls_line.append("/MN")

    point_id = 1
    pos_header = []
    line_num = 1  

    for points in all_points:
        if len(points) < 2:
            continue
        
        # Move to the first point with pen up
        x, y = points[0]
        ls_line.append(f"{line_num}:J P[{point_id}] 100% FINE ;")
        pos_header.append(f"P[{point_id}] {{X {x:.1f} Y {y:.1f} Z {z_up:.1f} W 0 P 0 R 0}};")
        point_id += 1
        line_num += 1

        # Move down to start drawing
        ls_line.append(f"{line_num}:L P[{point_id}] {JOINT_SPEED}mm/sec FINE ;")
        pos_header.append(f"P[{point_id}] {{X {x:.1f} Y {y:.1f} Z {z_down:.1f} W 0 P 0 R 0}};")
        point_id += 1
        line_num += 1

        # Draw the rest of the points
        for x, y in points[1:]:
            ls_line.append(f"{line_num}:L P[{point_id}] {JOINT_SPEED}mm/sec FINE ;")
            pos_header.append(f"P[{point_id}] {{X {x:.1f} Y {y:.1f} Z {z_down:.1f} W 0 P 0 R 0}};")
            point_id += 1
            line_num += 1

        # Move up after finishing
        ls_line.append(f"{line_num}:L P[{point_id}] {JOINT_SPEED}mm/sec FINE ;")
        pos_header.append(f"P[{point_id}] {{X {x:.1f} Y {y:.1f} Z {z_up:.1f} W 0 P 0 R 0}};")
        point_id += 1
        line_num += 1

    ls_line.append("/POS")
    ls_line.extend(pos_header)
    ls_line.append("/END")

    with open(OUTPUT_LS, 'w') as f:
        f.write('\n'.join(ls_line) + '\n')

    print(f"✅ file {OUTPUT_LS} created!")


def main():
    all_points = take_points_from_svg()
    write_ls_file(all_points)


if __name__ == "__main__":
    main()