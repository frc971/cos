#!/usr/bin/env python3
"""Draw a robot-frame floor circle and velocity goal on a camera image."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def matmul(a, b):
    return [sum(a[i][k] * b[k] for k in range(3)) for i in range(3)]


def matrix_multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def rotation(extrinsics):
    rx, ry, rz = [
        math.radians(extrinsics[f"rotation_{axis}"]) for axis in "xyz"
    ]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    my = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    mz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    return matrix_multiply(matrix_multiply(mz, my), mx)


def distort_and_project(point, intrinsics, extrinsics, robot_to_camera, optical_axis):
    origin = [extrinsics[f"translation_{axis}"] for axis in "xyz"]
    relative = [point[i] - origin[i] for i in range(3)]
    local = matmul(robot_to_camera, relative)

    axis_sign = 1.0 if optical_axis == "positive-x" else -1.0
    depth = axis_sign * local[0]
    if depth <= 0.01:
        return None
    # OpenCV image axes are right/down; WPILib camera axes are forward/left/up.
    x, y = -axis_sign * local[1] / depth, -axis_sign * local[2] / depth
    r2 = x * x + y * y
    radial = 1 + intrinsics["k1"] * r2 + intrinsics["k2"] * r2**2 + intrinsics["k3"] * r2**3
    xd = x * radial + 2 * intrinsics["p1"] * x * y + intrinsics["p2"] * (r2 + 2 * x * x)
    yd = y * radial + intrinsics["p1"] * (r2 + 2 * y * y) + 2 * intrinsics["p2"] * x * y
    return (
        intrinsics["fx"] * xd + intrinsics["cx"],
        intrinsics["fy"] * yd + intrinsics["cy"],
    )


def clip_line(p0, p1, width, height):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    low, high = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, width - 1 - x0), (-dy, y0), (dy, height - 1 - y0)):
        if p == 0:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                low = max(low, t)
            else:
                high = min(high, t)
    if low > high:
        return None
    return ((x0 + low * dx, y0 + low * dy), (x0 + high * dx, y0 + high * dy))


def label(draw, xy, text, fill, font):
    box = draw.textbbox(xy, text, font=font, stroke_width=1)
    draw.rounded_rectangle((box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3), radius=4, fill=(0, 0, 0, 190))
    draw.text(xy, text, fill=fill, font=font, stroke_width=1, stroke_fill=(0, 0, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("camera", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument(
        "--optical-axis",
        choices=("positive-x", "negative-x"),
        default="negative-x",
    )
    parser.add_argument("--translation-x", type=float, default=0.0)
    parser.add_argument("--translation-y", type=float, default=0.0)
    parser.add_argument("--translation-z", type=float, default=0.4)
    parser.add_argument("--rotation-x", type=float, default=180.0)
    parser.add_argument("--rotation-y", type=float, default=-8.0)
    parser.add_argument("--rotation-z", type=float, default=180.0)
    parser.add_argument(
        "--rotate-180",
        action="store_true",
        help="Display an upside-down camera stream upright.",
    )
    parser.add_argument(
        "--projection-in-raw",
        action="store_true",
        help="Rotate projected raw-sensor pixels along with an upside-down frame.",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Write only the transparent annotation layer.",
    )
    args = parser.parse_args()

    config = json.loads(args.camera.read_text())
    intrinsics = config["intrinsics"]
    extrinsics = dict(config["extrinsics"])
    extrinsics.update(
        translation_x=args.translation_x,
        translation_y=args.translation_y,
        translation_z=args.translation_z,
        rotation_x=args.rotation_x,
        rotation_y=args.rotation_y,
        rotation_z=args.rotation_z,
    )
    robot_to_camera = transpose(rotation(extrinsics))

    image = Image.open(args.image).convert("RGBA")
    if args.rotate_180:
        image = image.rotate(180)

    def display_pixel(pixel):
        if pixel is not None and args.rotate_180 and args.projection_in_raw:
            return (image.width - 1 - pixel[0], image.height - 1 - pixel[1])
        return pixel

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=18)

    target = (args.target_x, args.target_y)
    points = []
    for i in range(721):
        angle = 2 * math.pi * i / 720
        floor_point = (
            target[0] + args.radius * math.cos(angle),
            target[1] + args.radius * math.sin(angle),
            0.0,
        )
        points.append(
            display_pixel(
                distort_and_project(
                    floor_point, intrinsics, extrinsics, robot_to_camera, args.optical_axis
                )
            )
        )

    visible_segments = []
    for first, second in zip(points, points[1:]):
        if first is None or second is None:
            continue
        segment = clip_line(first, second, image.width, image.height)
        if segment is not None:
            visible_segments.append(segment)
            draw.line(segment, fill=(40, 255, 70, 235), width=7)
            draw.line(segment, fill=(255, 255, 255, 230), width=2)

    target_pixel = display_pixel(
        distort_and_project(
            (target[0], target[1], 0.0),
            intrinsics,
            extrinsics,
            robot_to_camera,
            args.optical_axis,
        )
    )
    if target_pixel is not None:
        x, y = target_pixel
        if 0 <= x < image.width and 0 <= y < image.height:
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=(255, 40, 210, 245), outline="white", width=3)
            label(draw, (min(x + 16, image.width - 235), max(8, y - 14)), "velocity goal / cluster center", (255, 80, 220, 255), font)
        else:
            edge_x = min(max(x, 18), image.width - 18)
            edge_y = image.height - 8 if y >= image.height else 8
            direction = 1 if y >= image.height else -1
            draw.polygon(
                ((edge_x, edge_y), (edge_x - 12, edge_y - 18 * direction),
                 (edge_x + 12, edge_y - 18 * direction)),
                fill=(255, 40, 210, 245),
                outline="white",
            )
            label(
                draw,
                (min(edge_x + 18, image.width - 270), image.height - 34 if direction > 0 else 12),
                f"velocity goal off-screen (u={x:.0f}, v={y:.0f})",
                (255, 80, 220, 255),
                font,
            )

    if visible_segments:
        endpoints = [p for segment in visible_segments for p in segment]
        anchor = min(endpoints, key=lambda p: abs(p[0] - image.width * 0.18) + abs(p[1] - image.height * 0.72))
        label(
            draw,
            (min(anchor[0] + 10, image.width - 210), max(8, anchor[1] - 30)),
            f"{args.radius:.1f} m floor radius",
            (70, 255, 90, 255),
            font,
        )

    label(
        draw,
        (12, 12),
        f"second_bot/color intrinsics | mount: ({args.translation_x:.3f}, "
        f"{args.translation_y:.3f}, {args.translation_z:.3f}) m, "
        f"rpy=({args.rotation_x:g}, {args.rotation_y:g}, {args.rotation_z:g}) deg",
        "white",
        font,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_only:
        overlay.save(args.output)
    else:
        Image.alpha_composite(image, overlay).convert("RGB").save(
            args.output, quality=95
        )
    print(f"wrote {args.output}; visible circle segments={len(visible_segments)}; target_pixel={target_pixel}")


if __name__ == "__main__":
    main()
