from pathlib import Path

import cv2
import pytesseract
from PIL import Image

# =========================
# 路径配置
# =========================
IMG_DIR = Path("snapshoot")
OUT_DIR = Path("ocr_output")
OUT_DIR.mkdir(exist_ok=True)

# 如果你的 tesseract 不在 PATH 里，就取消注释并改成你的安装路径
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CROP_BOX = (0, 75, 3800, 2025)

TESS_CONFIG = r'--oem 3 --psm 6'

def load_and_crop_image(img_path: Path):
    """
    读取图片并按 CROP_BOX 裁剪，返回:
    - 原图(BGR)
    - 裁剪图(BGR)
    - 实际使用的裁剪框 (l, t, r, b)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")

    h, w = img.shape[:2]
    l, t, r, b = CROP_BOX

    # 防止裁剪框越界，便于手动调参时快速试错
    l = max(0, min(l, w))
    r = max(0, min(r, w))
    t = max(0, min(t, h))
    b = max(0, min(b, h))

    if r <= l or b <= t:
        raise ValueError(f"裁剪框无效: {(l, t, r, b)}，原图尺寸: {w}x{h}")

    cropped = img[t:b, l:r]
    return img, cropped, (l, t, r, b)


def preprocess_image(img_path: Path):
    """
    读取图片 -> 裁剪 -> 灰度 -> 二值化
    返回适合 OCR 的 PIL Image
    """
    _, img, _ = load_and_crop_image(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 简单二值化，截图文字通常会更清楚
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 放大一点，OCR 往往更稳
    th = cv2.resize(th, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(th)


def normalize_text(s: str) -> str:
    """
    OCR 常见修正
    """
    s = s.replace("O", "0")
    s = s.replace("o", "0")
    s = s.replace("I", "1")
    s = s.replace("l", "1")
    s = s.replace("|", "1")
    s = s.replace("‘", '"').replace("’", '"')
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")
    return s


def ocr_image(img_path: Path) -> str:
    img = preprocess_image(img_path)
    return pytesseract.image_to_string(img, config=TESS_CONFIG)


def main():
    img_files = sorted(IMG_DIR.glob("*.png"))
    if not img_files:
        print("snapshoot 目录下没有 png 图片。")
        return

    selected_files = img_files
    print(f"将处理全部 {len(selected_files)} 张截图。")
    print(f"CROP_BOX: {CROP_BOX}")

    generated_txt_paths = []
    for idx, img_path in enumerate(selected_files, start=1):
        print(f"[{idx}/{len(selected_files)}] OCR: {img_path.name}")

        full_img, cropped_img, crop_box = load_and_crop_image(img_path)
        l, t, r, b = crop_box
        print(f"  裁剪区域: left={l}, top={t}, right={r}, bottom={b}, size={r-l}x{b-t}")

        crop_preview_path = OUT_DIR / f"{img_path.stem}_crop_preview.png"
        cv2.imwrite(str(crop_preview_path), cropped_img)

        marked = full_img.copy()
        cv2.rectangle(marked, (l, t), (r, b), (0, 255, 0), 2)
        marked_path = OUT_DIR / f"{img_path.stem}_crop_marked.png"
        cv2.imwrite(str(marked_path), marked)

        text = ocr_image(img_path)
        out_txt_path = OUT_DIR / f"{img_path.stem}.txt"
        out_txt_path.write_text(text, encoding="utf-8")
        generated_txt_paths.append(out_txt_path)

    # 汇总所有 OCR 原样输出，并去掉空行
    merged_lines = []
    for txt_path in generated_txt_paths:
        raw = txt_path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            if line.strip():
                merged_lines.append(line)

    merged_path = OUT_DIR / "all_ocr_no_blank_lines.txt"
    merged_path.write_text("\n".join(merged_lines), encoding="utf-8")

    print("完成。输出目录：", OUT_DIR.resolve())
    print(f"每张截图输出: 原样 txt + crop_preview + crop_marked，共 {len(selected_files)} 组。")
    print("汇总文件（去空行）：", merged_path.resolve())


if __name__ == "__main__":
    main()