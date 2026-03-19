import os
import json
import random
import base64
import zlib
import io
import numpy as np
import cv2
import shutil
from PIL import Image, ImageDraw, ImageFont

# Константы
ROOT_DIR = "/mnt/ssd4tb/project/BAMNet/segmentation_point(v2)"
OUTPUT_DIR = "./output_samples"
NUM_SAMPLES = 25
DPI = 300

# Цвета из meta.json (Supervisely)
CLASS_COLORS = {
    "mask": "#49BC4E",
    "AA1": "#E916E6",
    "AA2": "#AEFF01",
    "STJ1": "#DBFF00",
    "STJ2": "#AEFF01",
}

# Стиль бейджей Supervisely
BADGE_BG = (50, 50, 50, 200)       # тёмно-серый фон с прозрачностью
BADGE_RADIUS = 3                    # скругление углов
BADGE_PADDING_X = 6                 # горизонтальный отступ
BADGE_PADDING_Y = 3                 # вертикальный отступ
BADGE_ICON_SIZE = 8                 # размер иконки (треугольник из 3 точек)
BADGE_ICON_GAP = 5                  # отступ между иконкой и текстом
BADGE_FONT_SIZE = 12                # размер шрифта
POINT_RADIUS = 4                    # радиус точки-аннотации
MASK_CONTOUR_WIDTH = 2              # толщина контура маски
MASK_FILL_OPACITY = 100             # прозрачность заливки маски (~0.39)


def hex_to_rgb(hex_color):
    """Конвертация HEX цвета в RGB кортеж."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def decode_supervisely_bitmap(data):
    """Декодирование растровой маски Supervisely (base64 -> zlib -> PNG)."""
    raw = base64.b64decode(data)
    decompressed = zlib.decompress(raw)
    return Image.open(io.BytesIO(decompressed)).convert("L")


def load_font():
    """Загрузка шрифта для меток."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, BADGE_FONT_SIZE)
    return ImageFont.load_default()


def draw_supervisely_icon(draw, cx, cy, color_rgb, size=8):
    """
    Рисует иконку Supervisely (3 точки треугольником) — 
    стиль иконки класса в бейдже.
    Верхняя точка по центру, две нижних по бокам.
    """
    dot_r = max(1, size // 5)
    half = size // 2
    # Верхняя точка
    draw.ellipse([cx - dot_r, cy - half - dot_r, cx + dot_r, cy - half + dot_r],
                 fill=color_rgb + (255,))
    # Нижняя левая
    draw.ellipse([cx - half - dot_r, cy + half - dot_r - 1, cx - half + dot_r, cy + half + dot_r - 1],
                 fill=color_rgb + (255,))
    # Нижняя правая
    draw.ellipse([cx + half - dot_r, cy + half - dot_r - 1, cx + half + dot_r, cy + half + dot_r - 1],
                 fill=color_rgb + (255,))


def draw_supervisely_badge(draw, x, y, label, color_rgb, font, anchor="left", show_icon=True):
    """
    Рисует бейдж-метку в стиле Supervisely:
    [тёмный фон со скруглёнными углами] [цветная иконка (опционально)] [белый текст]
    """
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    badge_h = max(text_h, BADGE_ICON_SIZE) + BADGE_PADDING_Y * 2
    if show_icon:
        badge_w = BADGE_PADDING_X + BADGE_ICON_SIZE + BADGE_ICON_GAP + text_w + BADGE_PADDING_X
    else:
        badge_w = BADGE_PADDING_X + text_w + BADGE_PADDING_X

    if anchor == "center":
        x = x - badge_w // 2

    bg_rect = [x, y, x + badge_w, y + badge_h]
    draw.rounded_rectangle(bg_rect, radius=BADGE_RADIUS, fill=BADGE_BG)

    if show_icon:
        icon_cx = x + BADGE_PADDING_X + BADGE_ICON_SIZE // 2
        icon_cy = y + badge_h // 2
        draw_supervisely_icon(draw, icon_cx, icon_cy, color_rgb, BADGE_ICON_SIZE)
        text_x = x + BADGE_PADDING_X + BADGE_ICON_SIZE + BADGE_ICON_GAP
    else:
        text_x = x + BADGE_PADDING_X

    text_y = y + (badge_h - text_h) // 2 - 1
    draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)

    return badge_w, badge_h


def get_all_samples(root_dir):
    """Поиск всех пар изображение-аннотация."""
    samples = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            img_dir = os.path.join(entry.path, "img")
            ann_dir = os.path.join(entry.path, "ann")
            if os.path.exists(img_dir) and os.path.exists(ann_dir):
                for img_name in os.listdir(img_dir):
                    if img_name.endswith(".png"):
                        ann_name = img_name + ".json"
                        ann_path = os.path.join(ann_dir, ann_name)
                        if os.path.exists(ann_path):
                            samples.append({
                                "img_path": os.path.join(img_dir, img_name),
                                "ann_path": ann_path,
                                "name": img_name,
                            })
    return samples


def draw_supervisely_style(img, ann_data):
    """Отрисовка аннотаций в стиле Supervisely."""
    font = load_font()
    W, H = img.size

    # --- Слой 1: маски (заливка + контур) ---
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    mask_labels = []  # для последующей отрисовки бейджей масок

    for obj in ann_data.get("objects", []):
        if obj.get("geometryType") != "bitmap":
            continue
        class_title = obj.get("classTitle", "mask")
        rgb = hex_to_rgb(CLASS_COLORS.get(class_title, "#49BC4E"))

        bitmap = obj.get("bitmap", {})
        data = bitmap.get("data")
        origin = bitmap.get("origin")
        if not data or not origin:
            continue

        mask_pil = decode_supervisely_bitmap(data)
        mask_np = np.array(mask_pil)
        x0, y0 = origin[0], origin[1]

        # Заливка маски
        fill_layer = Image.new("RGBA", mask_pil.size, rgb + (MASK_FILL_OPACITY,))
        transparent = Image.new("RGBA", mask_pil.size, (0, 0, 0, 0))
        transparent.paste(fill_layer, (0, 0), mask_pil)
        overlay.paste(transparent, (x0, y0), transparent)

        # Контур маски (OpenCV)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_shifted = cnt + [x0, y0]
            pts = cnt_shifted.reshape(-1, 2).tolist()
            if len(pts) > 1:
                point_list = [tuple(p) for p in pts] + [tuple(pts[0])]
                overlay_draw.line(point_list, fill=rgb + (255,), width=MASK_CONTOUR_WIDTH)

        # Запоминаем позицию для бейджа маски (нижний-левый угол видимой области)
        ys, xs = np.where(mask_np > 0)
        if len(ys) > 0:
            # Берём нижнюю-левую точку маски
            max_y_idx = np.argmax(ys)
            label_x = x0 + int(xs[np.argmin(xs)]) - 10
            label_y = y0 + int(np.max(ys)) + 5
            # Ограничиваем позицию
            label_x = max(5, min(label_x, W - 80))
            label_y = min(label_y, H - 25)
            mask_labels.append((class_title, rgb, label_x, label_y))

    # Комбинирование
    img_rgba = img.convert("RGBA")
    combined = Image.alpha_composite(img_rgba, overlay)
    draw = ImageDraw.Draw(combined)

    # --- Слой 2: бейджи масок ---
    for class_title, rgb, lx, ly in mask_labels:
        draw_supervisely_badge(draw, lx, ly, class_title, rgb, font)

    # --- Слой 3: точки + бейджи точек ---
    for obj in ann_data.get("objects", []):
        if obj.get("geometryType") != "point":
            continue
        class_title = obj.get("classTitle", "point")
        rgb = hex_to_rgb(CLASS_COLORS.get(class_title, "#FFFFFF"))

        pts = obj.get("points", {}).get("exterior", [])
        if not pts:
            continue

        x, y = pts[0][0], pts[0][1]

        # Точка: чёрная обводка + цветная заливка + белый центр
        r = POINT_RADIUS
        draw.ellipse([x-r-1, y-r-1, x+r+1, y+r+1],
                     fill=(0, 0, 0, 255))
        draw.ellipse([x-r, y-r, x+r, y+r],
                     fill=rgb + (255,))
        draw.ellipse([x-1, y-1, x+1, y+1],
                     fill=(255, 255, 255, 255))

        # Бейдж справа от точки
        badge_x = x + r + 5
        badge_y = y - 10
        # Ограничиваем
        badge_y = max(2, badge_y)
        draw_supervisely_badge(draw, badge_x, badge_y, class_title, rgb, font, show_icon=False)

    return combined.convert("RGB")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print("Finding samples...")
    all_samples = get_all_samples(ROOT_DIR)
    print(f"Found {len(all_samples)} samples.")

    if len(all_samples) < NUM_SAMPLES:
        selected = all_samples
    else:
        selected = random.sample(all_samples, NUM_SAMPLES)

    for i, sample in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] Processing {sample['name']}...")
        try:
            img = Image.open(sample['img_path'])
            with open(sample['ann_path'], 'r') as f:
                ann_data = json.load(f)

            result = draw_supervisely_style(img, ann_data)

            out_name = f"sample_{i+1:02d}_{sample['name'].replace('.png', '.jpeg')}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            result.save(out_path, "JPEG", dpi=(DPI, DPI), quality=95)

            # Копируем оригинальный файл
            orig_name = f"orig_{i+1:02d}_{sample['name']}"
            orig_path = os.path.join(OUTPUT_DIR, orig_name)
            shutil.copy2(sample['img_path'], orig_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"Done. {len(selected)} images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
