import os
import json
import random
import base64
import zlib
import io
import numpy as np
import cv2
import shutil
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bamnet_paths import get_data_path

# Константы
DEFAULT_DATASET_ROOT = str(get_data_path("export_project", "segmentation_point"))
OUTPUT_DIR = str(ROOT_DIR / "publication" / "generated_samples")
NUM_SAMPLES = 25
DPI = 300
DEFAULT_EXPORT_SIZE = None

# Цвета из meta.json (Supervisely)
CLASS_COLORS = {
    #"mask": "#49BC4E",
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


def scale_px(value, render_scale, minimum=1):
    return max(minimum, int(round(value * render_scale)))


def load_font(font_size):
    """Загрузка шрифта для меток."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, font_size)
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


def draw_supervisely_badge(draw, x, y, label, color_rgb, font, style, anchor="left", show_icon=True):
    """
    Рисует бейдж-метку в стиле Supervisely:
    [тёмный фон со скруглёнными углами] [цветная иконка (опционально)] [белый текст]
    """
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    badge_h = max(text_h, style["badge_icon_size"]) + style["badge_padding_y"] * 2
    if show_icon:
        badge_w = (
            style["badge_padding_x"]
            + style["badge_icon_size"]
            + style["badge_icon_gap"]
            + text_w
            + style["badge_padding_x"]
        )
    else:
        badge_w = style["badge_padding_x"] + text_w + style["badge_padding_x"]

    if anchor == "center":
        x = x - badge_w // 2

    bg_rect = [x, y, x + badge_w, y + badge_h]
    draw.rounded_rectangle(bg_rect, radius=style["badge_radius"], fill=BADGE_BG)

    if show_icon:
        icon_cx = x + style["badge_padding_x"] + style["badge_icon_size"] // 2
        icon_cy = y + badge_h // 2
        draw_supervisely_icon(draw, icon_cx, icon_cy, color_rgb, style["badge_icon_size"])
        text_x = x + style["badge_padding_x"] + style["badge_icon_size"] + style["badge_icon_gap"]
    else:
        text_x = x + style["badge_padding_x"]

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


def patient_dir_from_image_name(image_name):
    patient_token = image_name.split("_", 1)[0]
    if not patient_token.isdigit():
        raise ValueError(f"Cannot derive patient id from image name: {image_name}")
    return f"{int(patient_token):03d}"


def get_single_sample(input_path, dataset_root):
    """Собирает одну пару image+annotation по имени файла."""
    image_name = os.path.basename(input_path)
    patient_dir = patient_dir_from_image_name(image_name)
    ann_path = os.path.join(dataset_root, patient_dir, "ann", image_name + ".json")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation not found for {image_name}: {ann_path}")

    return {
        "img_path": input_path,
        "ann_path": ann_path,
        "name": image_name,
    }


def compute_render_scale(img_size, export_size):
    if not export_size:
        return 1.0
    longest_side = max(img_size)
    if longest_side <= 0:
        return 1.0
    return float(export_size) / float(longest_side)


def draw_supervisely_style(img, ann_data, export_size=None):
    """Отрисовка аннотаций в стиле Supervisely."""
    orig_w, orig_h = img.size
    render_scale = compute_render_scale((orig_w, orig_h), export_size)
    if abs(render_scale - 1.0) > 1e-6:
        target_size = (
            max(1, int(round(orig_w * render_scale))),
            max(1, int(round(orig_h * render_scale))),
        )
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    W, H = img.size

    style = {
        "badge_radius": scale_px(BADGE_RADIUS, render_scale),
        "badge_padding_x": scale_px(BADGE_PADDING_X, render_scale),
        "badge_padding_y": scale_px(BADGE_PADDING_Y, render_scale),
        "badge_icon_size": scale_px(BADGE_ICON_SIZE, render_scale),
        "badge_icon_gap": scale_px(BADGE_ICON_GAP, render_scale),
        "badge_font_size": scale_px(BADGE_FONT_SIZE, render_scale, minimum=8),
        "point_radius": scale_px(POINT_RADIUS, render_scale),
        "mask_contour_width": scale_px(MASK_CONTOUR_WIDTH, render_scale),
    }
    font = load_font(style["badge_font_size"])

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
        if abs(render_scale - 1.0) > 1e-6:
            mask_pil = mask_pil.resize(
                (
                    max(1, int(round(mask_pil.size[0] * render_scale))),
                    max(1, int(round(mask_pil.size[1] * render_scale))),
                ),
                Image.Resampling.NEAREST,
            )
        mask_np = np.array(mask_pil)
        x0 = int(round(origin[0] * render_scale))
        y0 = int(round(origin[1] * render_scale))

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
                overlay_draw.line(point_list, fill=rgb + (255,), width=style["mask_contour_width"])

        # Запоминаем позицию для бейджа маски (нижний-левый угол видимой области)
        ys, xs = np.where(mask_np > 0)
        if len(ys) > 0:
            # Берём нижнюю-левую точку маски
            max_y_idx = np.argmax(ys)
            label_x = x0 + int(xs[np.argmin(xs)]) - 10
            label_y = y0 + int(np.max(ys)) + 5
            # Ограничиваем позицию
            label_x = max(5, min(label_x, W - scale_px(80, render_scale)))
            label_y = min(label_y, H - scale_px(25, render_scale))
            mask_labels.append((class_title, rgb, label_x, label_y))

    # Комбинирование
    img_rgba = img.convert("RGBA")
    combined = Image.alpha_composite(img_rgba, overlay)
    draw = ImageDraw.Draw(combined)

    # --- Слой 2: бейджи масок ---
    for class_title, rgb, lx, ly in mask_labels:
        draw_supervisely_badge(draw, lx, ly, class_title, rgb, font, style)

    # --- Слой 3: точки + бейджи точек ---
    for obj in ann_data.get("objects", []):
        if obj.get("geometryType") != "point":
            continue
        class_title = obj.get("classTitle", "point")
        rgb = hex_to_rgb(CLASS_COLORS.get(class_title, "#FFFFFF"))

        pts = obj.get("points", {}).get("exterior", [])
        if not pts:
            continue

        x = int(round(pts[0][0] * render_scale))
        y = int(round(pts[0][1] * render_scale))

        # Точка: чёрная обводка + цветная заливка + белый центр
        r = style["point_radius"]
        draw.ellipse([x-r-1, y-r-1, x+r+1, y+r+1],
                     fill=(0, 0, 0, 255))
        draw.ellipse([x-r, y-r, x+r, y+r],
                     fill=rgb + (255,))
        draw.ellipse([x-1, y-1, x+1, y+1],
                     fill=(255, 255, 255, 255))

        # Бейдж справа от точки
        badge_x = x + r + 5
        badge_y = y - scale_px(10, render_scale)
        # Ограничиваем
        badge_y = max(2, badge_y)
        draw_supervisely_badge(draw, badge_x, badge_y, class_title, rgb, font, style, show_icon=False)

    return combined.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Генерация примеров аннотированных изображений.")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Путь к Supervisely-датасету или к одному PNG-файлу.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Корень raw Supervisely-датасета для поиска ann/<image>.json в single-image режиме.",
    )
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Куда сохранить JPEG-примеры.")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Сколько примеров сохранить.")
    parser.add_argument(
        "--export-size",
        type=int,
        default=DEFAULT_EXPORT_SIZE,
        help="Целевой размер большей стороны итогового изображения. "
             "Если не задан, сохраняется исходный размер.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created output directory: {args.output}")

    if os.path.isfile(args.input):
        selected = [get_single_sample(args.input, args.dataset_root)]
        print(f"Single-image mode: {selected[0]['name']}")
    else:
        print("Finding samples...")
        all_samples = get_all_samples(args.input)
        print(f"Found {len(all_samples)} samples.")

        if len(all_samples) < args.num_samples:
            selected = all_samples
        else:
            selected = random.sample(all_samples, args.num_samples)

    for i, sample in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] Processing {sample['name']}...")
        try:
            img = Image.open(sample['img_path'])
            with open(sample['ann_path'], 'r') as f:
                ann_data = json.load(f)

            result = draw_supervisely_style(img, ann_data, export_size=args.export_size)

            if args.export_size:
                render_scale = compute_render_scale(img.size, args.export_size)
                if abs(render_scale - 1.0) > 1e-6:
                    img = img.resize(
                        (
                            max(1, int(round(img.size[0] * render_scale))),
                            max(1, int(round(img.size[1] * render_scale))),
                        ),
                        Image.Resampling.LANCZOS,
                    )

            if len(selected) == 1:
                out_name = f"sample_{sample['name'].replace('.png', '.jpeg')}"
                orig_name = f"orig_{sample['name']}"
            else:
                out_name = f"sample_{i+1:02d}_{sample['name'].replace('.png', '.jpeg')}"
                orig_name = f"orig_{i+1:02d}_{sample['name']}"
            out_path = os.path.join(args.output, out_name)
            result.save(out_path, "JPEG", dpi=(DPI, DPI), quality=95)

            # Копируем оригинальный файл
            orig_path = os.path.join(args.output, orig_name)
            if args.export_size:
                img.save(orig_path, dpi=(DPI, DPI))
            else:
                shutil.copy2(sample['img_path'], orig_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"Done. {len(selected)} images saved to {args.output}/")


if __name__ == "__main__":
    main()
