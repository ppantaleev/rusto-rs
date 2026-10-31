#!/usr/bin/env python3
"""Generate the synthetic document fixtures (page_text, table_3x4_merged).

All content is synthetic (invented Bulgarian text); no corpus pages are used.
Font: DejaVu Sans (Bitstream Vera licence + public-domain changes), which
permits embedding glyph renderings in images. Requires Python 3 + Pillow only.

Usage: gen_doc_fixtures.py --out DIR [--font PATH] [--font-bold PATH]
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1240, 1754
MARGIN = 20
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    os.path.expanduser("~/Library/Fonts/DejaVuSans.ttf"),
]


def find_font(explicit):
    cands = [explicit] if explicit else list(FONT_CANDIDATES)
    if not explicit:
        try:
            import matplotlib
            cands.append(os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf"))
        except Exception:
            pass
    for c in cands:
        if c and os.path.exists(c):
            return c
    print("DejaVuSans.ttf not found; pass --font PATH", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--font")
    ap.add_argument("--font-bold")
    a = ap.parse_args()
    regular = find_font(a.font)
    bold = a.font_bold or regular.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
    if not os.path.exists(bold):
        bold = regular
    f_title = ImageFont.truetype(bold, 48)
    f_body = ImageFont.truetype(regular, 24)
    f_cell = ImageFont.truetype(regular, 26)
    os.makedirs(a.out, exist_ok=True)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def text_box(x, y, s, font):
        d.text((x, y), s, font=font, fill="black")
        return d.textbbox((x, y), s, font=font)

    def union(boxes):
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]

    title = text_box(100, 100, "Доклад за строителни работи", f_title)

    paras = [
        ["Настоящият документ описва обема на строителните дейности",
         "по обекта през текущата година. Всички данни са условни",
         "и служат единствено за проверка на разпознаването на текст.",
         "Количествата са посочени в таблицата по-долу."],
        ["Изпълнението на работите се извършва на етапи в съответствие",
         "с утвърдения график. Отговорното лице следи за качеството",
         "на материалите и за спазването на сроковете. Забележки",
         "се вписват в дневника на обекта."],
    ]
    para_boxes = []
    y = 260
    for p in paras:
        boxes = []
        for line in p:
            boxes.append(text_box(100, y, line, f_body))
            y += 40
        para_boxes.append(union(boxes))
        y += 60

    # Table: 3 rows x 4 columns, 2 px ruled lines.
    tx0, ty0 = 100, y + 20
    col_w = [120, 380, 200, 240]
    row_h = 60
    xs = [tx0]
    for w in col_w:
        xs.append(xs[-1] + w)
    ys = [ty0 + i * row_h for i in range(4)]
    tx1, ty1 = xs[-1], ys[-1]
    LW = 2
    for yy in ys:
        d.line([(tx0, yy), (tx1, yy)], fill="black", width=LW)
    for i, xx in enumerate(xs):
        if i == 1:
            # Row 0 cell "Позиция" spans columns 0-1: no divider in row 0.
            d.line([(xx, ys[1]), (xx, ty1)], fill="black", width=LW)
        else:
            d.line([(xx, ty0), (xx, ty1)], fill="black", width=LW)

    def cell(r, c, s, span=1):
        x0, x1 = xs[c], xs[c + span]
        tw = d.textlength(s, font=f_cell)
        d.text(((x0 + x1 - tw) / 2, ys[r] + 14), s, font=f_cell, fill="black")

    cell(0, 0, "Позиция", 2)
    cell(0, 2, "Количество")
    cell(0, 3, "Цена")
    data = [["1", "Бетон", "12", "340,00"], ["2", "Арматура", "5", "1 200,00"]]
    for r, row in enumerate(data, start=1):
        for c, s in enumerate(row):
            cell(r, c, s)

    table = [tx0 - LW // 2 - 1, ty0 - LW // 2 - 1, tx1 + LW // 2 + 1, ty1 + LW // 2 + 1]
    page_meta = {"width": W, "height": H, "title": list(title),
                 "paragraphs": para_boxes, "table": table}
    img.save(os.path.join(a.out, "page_text.png"), optimize=True)
    with open(os.path.join(a.out, "page_text.json"), "w", encoding="utf-8") as f:
        json.dump(page_meta, f, ensure_ascii=False)

    crop = (table[0] - MARGIN, table[1] - MARGIN, table[2] + MARGIN, table[3] + MARGIN)
    timg = img.crop(crop)
    timg.save(os.path.join(a.out, "table_3x4_merged.png"), optimize=True)
    tmeta = {"width": timg.width, "height": timg.height, "num_rows": 3, "num_cols": 4,
             "merged": {"row": 0, "col": 0, "row_span": 1, "col_span": 2},
             "rows": [["Позиция", "Количество", "Цена"]] + data}
    with open(os.path.join(a.out, "table_3x4_merged.json"), "w", encoding="utf-8") as f:
        json.dump(tmeta, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
