"""Draws the featured image, a bar chart of mean whole record accuracy from the saved results (same figures as Table 1).

    python3 make_featured_chart.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
NAVY, GRAY, INK, MUTED, RULE, BG = (16, 42, 67), (148, 163, 184), (28, 36, 48), (100, 116, 139), (226, 232, 240), (255, 255, 255)
S = 2
W, H = 1600 * S, 900 * S
ROWS = [  # label, mean whole record accuracy from RESULTS.md and Table 1
    ("Qwen3 0.6B, trained", 70.35, NAVY),
    ("Haiku, with label definitions", 56.66, GRAY),
    ("GPT-6 Sol, original prompt", 52.66, GRAY),
    ("Haiku, original prompt", 47.86, GRAY),
]
f = lambda p, n: ImageFont.truetype(p, n * S)
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
left, right, top, rowh, barh = 520 * S, 1480 * S, 190 * S, 150 * S, 84 * S
scale = (right - left) / 80.0
d.text((100 * S, 70 * S), "Whole record accuracy on 2,974 test commands", font=f(BOLD, 40), fill=INK)
for t in (0, 20, 40, 60, 80):
    x = left + t * scale
    d.line([(x, top - 20 * S), (x, top + 4 * rowh - 40 * S)], fill=RULE, width=2 * S)
    d.text((x, top + 4 * rowh - 20 * S), str(t), font=f(FONT, 26), fill=MUTED, anchor="ma")
for i, (label, v, col) in enumerate(ROWS):
    y = top + i * rowh
    d.text((left - 30 * S, y + barh / 2), label, font=f(BOLD if i == 0 else FONT, 32), fill=INK, anchor="rm")
    d.rectangle([left, y, left + v * scale, y + barh], fill=col)
    d.text((left + v * scale + 20 * S, y + barh / 2), f"{v:.2f}", font=f(BOLD, 36), fill=INK if i else NAVY, anchor="lm")
img = img.resize((1600, 900), Image.LANCZOS)
out = HERE / "figures" / "featured_chart.png"
img.save(out, optimize=True)
print(out, img.size)
