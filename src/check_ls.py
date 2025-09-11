import re
import matplotlib.pyplot as plt

DEFAULT_PATH = "LS/DRAW.LS"


def parse_ls(file_path):
    points = {}
    sequence = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # --- Сначала достаем точки из /POS ---
    pos_section = False
    for line in lines:
        if line.startswith("/POS"):
            pos_section = True
            continue
        if line.startswith("/END"):
            pos_section = False
        if pos_section and "P[" in line:
            match = re.search(r"P\[(\d+)\].*X ([\d\.\-]+) Y ([\d\.\-]+) Z ([\d\.\-]+)", line)
            if match:
                idx, x, y, z = match.groups()
                points[int(idx)] = (float(x), float(y), float(z))

    # --- Теперь достаем порядок движения из /MN ---
    mn_section = False
    for line in lines:
        if line.startswith("/MN"):
            mn_section = True
            continue
        if line.startswith("/POS"):
            mn_section = False
        if mn_section:
            match = re.search(r"[JL] P\[(\d+)\]", line)
            if match:
                sequence.append(int(match.group(1)))

    return points, sequence


def plot_path(points, sequence):
    xs, ys = [], []

    for idx in sequence:
        if idx in points:
            x, y, z = points[idx]
            xs.append(x)
            ys.append(y)

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o")
    plt.title("Fanuc LS Path (XY projection)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
 

    file_path = DEFAULT_PATH
    points, sequence = parse_ls(file_path)
    print("Последовательность точек:", sequence)
    plot_path(points, sequence)
