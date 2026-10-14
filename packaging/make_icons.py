"""Draw Athena's app icon with Pillow and write it for every platform.

The artwork matches ``src/destiny_bot/web/icon.svg`` (a blue tile with a gold laurel
wreath, an open book and sound waves rising from it). Sizes of 32 px and below use
a simplified drawing without the laurel and with bolder strokes, so the icon stays
legible in the taskbar and title bar.

Writes into ``packaging/icons/``:
    icon.ico       16, 20, 24, 32, 40, 48, 64, 128 and 256 px (Windows exe, shortcuts, installer)
    icon.icns      16 to 1024 px (macOS app bundle)
    icon-512.png   512 px (Linux desktop entries, README)

Usage: python packaging/make_icons.py [--out DIR]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
VIEWBOX = 64.0
SUPERSAMPLE = 4

TOP = (0x3A, 0x78, 0xA0)
BOTTOM = (0x24, 0x4F, 0x6B)
GOLD = (0xEC, 0xD5, 0x8F)
WHITE = (0xFF, 0xFF, 0xFF)
PAGE_SHADE = (0xDC, 0xEB, 0xF4)

# Laurel leaves on the left branch: (x, y, rotation in degrees), as in icon.svg.
LEAVES = [
    (21.0, 53.7, -190), (21.0, 53.7, -114), (13.7, 47.8, -167), (13.7, 47.8, -91),
    (9.4, 39.5, -144), (9.4, 39.5, -68), (8.7, 30.1, -121), (8.7, 30.1, -45),
    (11.3, 22.0, -100), (11.3, 22.0, -24), (12.5, 19.9, -56),
]  # fmt: skip
WREATH_CENTER = (32.0, 33.0)
WREATH_RADIUS = 23.5
WAVE_CENTER = (32.0, 27.9)
WAVES = [(5.2, 1.0), (9.7, 0.85), (14.2, 0.68)]  # (radius, opacity)
WAVE_HALF_ANGLE = 45.4


def _quad(p0, p1, p2, steps=12):
    return [
        ((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0],
         (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1])
        for t in (i / steps for i in range(steps + 1))
    ]  # fmt: skip


def _cubic(p0, p1, p2, p3, steps=24):
    points = []
    for i in range(steps + 1):
        t = i / steps
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3
        points.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return points


def _leaf(x, y, degrees):
    """The lens-shaped leaf ``M0 0 Q3.6 -2.3 7.2 0 Q3.6 2.3 0 0Z``, rotated then moved to (x, y)."""
    outline = _quad((0, 0), (3.6, -2.3), (7.2, 0)) + _quad((7.2, 0), (3.6, 2.3), (0, 0))[1:]
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    return [(x + px * cos - py * sin, y + px * sin + py * cos) for px, py in outline]


def _page(mirror: bool, scale: float = 1.0, lift: float = 0.0):
    """One page of the open book (the left page, or the right page when mirrored)."""
    pts = _cubic((32, 31.2), (28.1, 28.4), (22.9, 27.6), (17, 28.7))
    pts += [(17, 45.3)]
    pts += _cubic((17, 45.3), (22.9, 44.2), (28.1, 45.0), (32, 47.8))[1:]
    cx, cy = 32.0, 38.0
    pts = [(cx + (px - cx) * scale, cy + (py - cy) * scale - lift) for px, py in pts]
    return [(64 - px, py) if mirror else (px, py) for px, py in pts]


def _arc_points(center, radius, start_deg, end_deg, steps=48):
    return [
        (center[0] + radius * math.cos(math.radians(a)), center[1] + radius * math.sin(math.radians(a)))
        for a in (start_deg + (end_deg - start_deg) * i / steps for i in range(steps + 1))
    ]


def _stroke(draw: ImageDraw.ImageDraw, points, width: float, fill) -> None:
    """A polyline with round joins and caps."""
    draw.line(points, fill=fill, width=max(1, round(width)), joint="curve")
    r = width / 2
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def _arc(draw: ImageDraw.ImageDraw, center, radius: float, width: float, fill) -> None:
    """A thick arc around ``center`` spanning WAVE_HALF_ANGLE either side of straight up, round caps."""
    x, y = center
    outer = radius + width / 2
    start, end = 270 - WAVE_HALF_ANGLE, 270 + WAVE_HALF_ANGLE
    draw.arc((x - outer, y - outer, x + outer, y + outer), start, end, fill=fill, width=max(1, round(width)))
    for angle in (start, end):
        ex, ey = x + radius * math.cos(math.radians(angle)), y + radius * math.sin(math.radians(angle))
        r = width / 2
        draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=fill)


def render(size: int) -> Image.Image:
    """The icon at ``size`` x ``size`` pixels (RGBA)."""
    simple = size <= 32
    px = size * SUPERSAMPLE
    k = px / VIEWBOX

    def sc(points):
        return [(x * k, y * k) for x, y in points]

    # Background tile: vertical gradient inside a rounded square (rx 15 of 64).
    gradient = Image.new("RGBA", (1, px))
    for y in range(px):
        t = y / max(px - 1, 1)
        gradient.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM)) + (255,))
    gradient = gradient.resize((px, px))
    mask = Image.new("L", (px, px), 0)
    inset = 0 if size > 32 else 0.5 * k  # a hair of margin keeps tiny icons from looking cropped
    ImageDraw.Draw(mask).rounded_rectangle(
        (inset, inset, px - 1 - inset, px - 1 - inset), radius=(13 if simple else 15) * k, fill=255
    )
    image = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    image.paste(gradient, (0, 0), mask)

    overlay = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if not simple:
        start = math.degrees(math.atan2(56.3 - WREATH_CENTER[1], 29.1 - WREATH_CENTER[0]))
        end = math.degrees(math.atan2(22.0 - WREATH_CENTER[1], 11.3 - WREATH_CENTER[0])) + 360
        branch = _arc_points(WREATH_CENTER, WREATH_RADIUS, start, end)
        for mirror in (False, True):
            flip = (lambda pts: [(64 - x, y) for x, y in pts]) if mirror else (lambda pts: pts)
            _stroke(draw, sc(flip(branch)), 1.6 * k, GOLD + (255,))
            for leaf in LEAVES:
                draw.polygon(sc(flip(_leaf(*leaf))), fill=GOLD + (255,))

    # Sound waves: concentric arcs above the book, fading outwards. Small sizes get
    # fewer, bolder arcs and a bigger book so the shapes survive at 16 px.
    if size <= 20:
        waves, wave_width, book_scale, drop, rise = [(6.0, 1.0), (15.0, 1.0)], 5.0, 1.3, 4.0, 1.5
    elif simple:
        waves, wave_width, book_scale, drop, rise = [(5.5, 1.0), (10.5, 1.0), (15.5, 0.85)], 3.8, 1.25, 2.5, 0.0
    else:
        waves, wave_width, book_scale, drop, rise = WAVES, 3.0, 1.0, 0.0, 0.0
    cx, cy = WAVE_CENTER[0], WAVE_CENTER[1] + drop * 0.5 - rise
    for radius, opacity in waves:
        layer = Image.new("RGBA", (px, px), (0, 0, 0, 0))
        _arc(ImageDraw.Draw(layer), (cx * k, cy * k), radius * k, wave_width * k, WHITE + (round(255 * opacity),))
        overlay = Image.alpha_composite(overlay, layer)
    draw = ImageDraw.Draw(overlay)

    # The open book.
    draw.polygon(sc(_page(False, book_scale, -drop)), fill=WHITE + (255,))
    draw.polygon(sc(_page(True, book_scale, -drop)), fill=PAGE_SHADE + (255,))

    image = Image.alpha_composite(image, overlay)
    image.putalpha(Image.composite(image.getchannel("A"), Image.new("L", (px, px), 0), mask))
    return image.resize((size, size), Image.Resampling.LANCZOS)


ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
ICNS_SIZES = [16, 32, 64, 128, 256, 512, 1024]


def write_icons(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    ico = [render(s) for s in ICO_SIZES]
    ico_path = out / "icon.ico"
    ico[-1].save(ico_path, format="ICO", sizes=[(s, s) for s in ICO_SIZES], append_images=ico[:-1])
    icns = [render(s) for s in ICNS_SIZES]
    icns_path = out / "icon.icns"
    icns[-1].save(icns_path, format="ICNS", append_images=icns[:-1])
    png_path = out / "icon-512.png"
    render(512).save(png_path, format="PNG", optimize=True)
    return [ico_path, icns_path, png_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "icons", help="output folder (default packaging/icons)")
    args = parser.parse_args(argv)
    for path in write_icons(args.out):
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
