import os, sys, json, math, io, base64, zlib, shutil
from pathlib import Path
from itertools import chain
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from bamnet_paths import get_data_path

from manet_coords import (
    LitBoundaryAwareSystem,
    improved_softargmax2d,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

EXPORT_MASK = True

# ===================== конфиг =====================
CKPT_PATH = Path("runs/aortic_ring/boundary_aware_manet/20251116_214738_finish/checkpoints/ep020.ckpt")

# Корень датасета, где лежат подпапки 001, 002, ...
DATA_ROOT  = get_data_path("export_project")
# Какие подпапки обрабатывать
#'005', '007', '008', '009', '010', '018', '019', '020', '021', '022', '023', '026', '030', '031', '032', '033', '0,34', '035', '0,36', '001','024','034','036','038','039', '057', '075'
INCLUDE_FOLDERS = ['0084'] 

# Где лежат сами изображения внутри каждой подпапки
IMAGES_SUBDIR = "img"

# Корень проекта в формате Supervisely
OUT_ROOT   = get_data_path("export_project")
OUT_PRED   = get_data_path("pred_out")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Имена точек в порядке выходов модели
POINT_NAMES = ["AA1","AA2","STJ1","STJ2"]  # если P != 4, лишние/недостающие будут названы pt_i

# ==================================================

def next_pow2(x, base=32):
    return int(math.ceil(x / base) * base)

def preprocess_image(rgb_uint8):
    h, w = rgb_uint8.shape[:2]
    size = next_pow2(max(h, w), 32)
    img = cv2.resize(rgb_uint8, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0), (h, w)

@torch.no_grad()
def forward_and_post(model, x, orig_hw):
    (orig_h, orig_w) = orig_hw
    out = model(x)
    seg_logits = out["segmentation"]         # [B,1,H',W']
    heat_pred  = out["point_heatmaps"]       # [B,P,H',W']

    seg_prob = torch.sigmoid(seg_logits)
    seg_up = F.interpolate(
        seg_prob, size=(orig_h, orig_w),
        mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()

    beta = 12.0
    px_hm, py_hm = improved_softargmax2d(heat_pred, beta=beta, stable=True)
    px_hm = px_hm[0].cpu().numpy()
    py_hm = py_hm[0].cpu().numpy()

    _, P, hh, ww = heat_pred.shape
    scale_x = (orig_w - 1.0) / max(1.0, ww - 1)
    scale_y = (orig_h - 1.0) / max(1.0, hh - 1)
    xs = px_hm * scale_x
    ys = py_hm * scale_y

    conf = torch.sigmoid(heat_pred)[0].cpu().numpy().reshape(P, -1).max(axis=1)
    pts = [(float(xs[j]), float(ys[j]), float(conf[j])) for j in range(P)]
    return seg_up, pts

def draw_overlay(rgb_uint8, mask_prob, points_px, alpha=0.25):
    rgb = np.asarray(rgb_uint8, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.shape[2] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

    mask = np.nan_to_num(mask_prob, nan=0.0, posinf=1.0, neginf=0.0)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    vis = rgb.astype(np.float32)
    vis = vis * (1.0 - alpha * mask[..., None]) + \
          np.array([255, 128, 96], dtype=np.float32) * (alpha * mask[..., None])
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    H, W = mask.shape
    for (x, y, c) in points_px:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cx, cy = int(round(x)), int(round(y))
        if 0 <= cx < W and 0 <= cy < H:
            col = (0, 255, 0) if c >= 0.5 else (255, 165, 0)
            cv2.drawMarker(
                vis, (cx, cy), color=col,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=18, thickness=2, line_type=cv2.LINE_AA
            )
    return vis

# --- Упаковка маски в Supervisely bitmap (zlib + base64, как в оф. доке) ---
def mask_to_supervisely_bitmap(bin_mask_u8):
    # bin_mask_u8: 0 или 255; считаем ненулевое как True
    ys, xs = np.where(bin_mask_u8 > 0)
    if len(xs) == 0:
        crop = np.zeros((1, 1), np.uint8)
        origin = [0, 0]
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        crop = bin_mask_u8[y0:y1+1, x0:x1+1]
        origin = [int(x0), int(y0)]

    # в bool-маску
    bool_mask = crop > 0

    # как в docs: PIL + палитра + PNG + zlib.compress + base64
    img_pil = Image.fromarray(bool_mask.astype(np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    png_bytes = bytes_io.getvalue()
    data = base64.b64encode(zlib.compress(png_bytes)).decode('utf-8')

    return {"origin": origin, "data": data}

def make_point_obj(name, x, y):
    return {
        "classTitle": name,
        "description": "",
        "tags": [],
        "points": {
            "exterior": [[int(round(x)), int(round(y))]],
            "interior": []
        },
        "geometryType": "point",
        "shape": "point",
        "nnCreated": False,
        "nnUpdated": False,
    }

def save_supervisely_json(dst_json: Path, image_name: str, H: int, W: int, mask_prob, pts):
    objects = []

    if EXPORT_MASK:
        bin_mask = (np.nan_to_num(mask_prob, nan=0.0) >= 0.5).astype(np.uint8) * 255
        bmp = mask_to_supervisely_bitmap(bin_mask)

        objects.append({
            "classTitle": "mask",
            "description": "",
            "tags": [],
            "bitmap": bmp,
            "shape": "bitmap",
            "geometryType": "bitmap",
            "nnCreated": False,
            "nnUpdated": False,
        })

    for i, (x, y, c) in enumerate(pts):
        name = POINT_NAMES[i] if i < len(POINT_NAMES) else f"pt_{i}"
        objects.append(make_point_obj(name, x, y))

    ann = {
        "description": "",
        "name": image_name,  # важно для Supervisely
        "size": {"width": int(W), "height": int(H)},
        "tags": [],
        "objects": objects,
        "customBigData": {}
    }
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False)
    return ann

def write_meta_json():
    """Простейший meta.json для проекта"""
    meta_path = OUT_ROOT / "meta.json"
    if meta_path.exists():
        return

    classes = [
        {
            "title": "mask",
            "shape": "bitmap",
            "color": "#FF0000",
            "geometry_config": {},
            "id": 1,
            "hotkey": ""
        },
    ]

    for idx, title in enumerate(POINT_NAMES, start=2):
        classes.append({
            "title": title,
            "shape": "point",
            "color": "#00FF12",
            "geometry_config": {},
            "id": idx,
            "hotkey": ""
        })

    meta = {
        "classes": classes,
        "tags": [],
        "projectType": "images",
        "projectSettings": {}
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def main():
    if not CKPT_PATH.exists():
        print(f"[ERR] checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device.upper()} :: {CKPT_PATH}")

    model = LitBoundaryAwareSystem.load_from_checkpoint(CKPT_PATH, map_location=device)
    model.eval().to(device)

    # Какие подпапки брать
    if INCLUDE_FOLDERS:
        subdirs = [d for d in DATA_ROOT.iterdir()
                   if d.is_dir() and d.name in INCLUDE_FOLDERS]
    else:
        subdirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]

    images = []
    for sd in subdirs:
        img_dir = sd / IMAGES_SUBDIR
        if not img_dir.exists():
            continue
        images.extend([
            p for p in img_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ])

    if not images:
        print("[ERR] no images found")
        sys.exit(1)

    for p in images:
        # читаем RGB
        rgb_bgr = cv2.imread(str(p))
        if rgb_bgr is None:
            print(f"[WARN] cannot read image: {p}")
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        x, _ = preprocess_image(rgb)
        x = x.to(device)
        mask_prob, pts = forward_and_post(model, x, (H, W))

        # dataset name = подпапка (например, "005")
        dataset_name = p.parents[1].name  # DATA_ROOT / <subdir> / img / file -> parents[1] = <subdir>
        dataset_root = OUT_ROOT / dataset_name

        out_img_dir = dataset_root / "img"
        out_ann_dir = dataset_root / "ann"
        out_info_dir = dataset_root / "img_info"
        out_pred_dir = OUT_PRED / dataset_name

        # out_img_dir.mkdir(parents=True, exist_ok=True)
        out_ann_dir.mkdir(parents=True, exist_ok=True)
        out_info_dir.mkdir(parents=True, exist_ok=True)
        out_pred_dir.mkdir(parents=True, exist_ok=True)

        # Имя файла картинки
        img_name = p.name  # foo.png

        # img: копируем исходник как <dataset>/img/foo.png
        # out_img_path = out_img_dir / img_name
        # if not out_img_path.exists():
        #     shutil.copy2(p, out_img_path)

        # ann: <dataset>/ann/foo.png.json
        out_ann_path = out_ann_dir / f"{img_name}.json"
        save_supervisely_json(out_ann_path, img_name, H, W, mask_prob, pts)

        # img_info: <dataset>/img_info/foo.png.json
        out_info_path = out_info_dir / f"{img_name}.json"
        with open(out_info_path, "w", encoding="utf-8") as f:
            json.dump({
                "file_name": img_name,
                "size": {"h": H, "w": W},
                "points": pts
            }, f, ensure_ascii=False, indent=2)

        # пред-визуализация
        vis = draw_overlay(rgb, mask_prob, pts, alpha=0.25)
        vis_path = out_pred_dir / f"{p.stem}_vis.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        rel = p.relative_to(DATA_ROOT)
        print(f"[OK] {rel} → {dataset_name}/ann/{img_name}.json")

    # meta.json для проекта
    write_meta_json()

    print(f"\n[INFO] Done. Project saved to: {OUT_ROOT.resolve()}")
    print(f"[INFO] Previews: {OUT_PRED.resolve()}")

if __name__ == "__main__":
    main()
