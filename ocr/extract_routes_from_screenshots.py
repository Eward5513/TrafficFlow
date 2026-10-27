import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pytesseract
from PIL import Image

from _paths import OCR_DATA

# =========================
# 路径配置
# =========================
IMG_DIR = OCR_DATA / "snapshoot_all"
OUT_DIR = OCR_DATA / "ocr_output_all"

# 如果你的 tesseract 不在 PATH 里，就取消注释并改成你的安装路径
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CROP_BOX = (0, 75, 3800, 2025)

TESS_CONFIG = r'--oem 3 --psm 6'

# 并发线程数：CPU 核心数的两倍
WORKERS = (os.cpu_count() or 1) * 2


def load_and_crop_image(img_path: Path):
    """
    读取图片并按 CROP_BOX 裁剪，返回:
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
    return cropped, (l, t, r, b)


def preprocess_image(cropped_img):
    """
    裁剪图 -> 灰度 -> 二值化
    返回适合 OCR 的 PIL Image
    """
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

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


def ocr_cropped_image(cropped_img) -> str:
    img = preprocess_image(cropped_img)
    return pytesseract.image_to_string(img, config=TESS_CONFIG)


def process_one_image(img_path: Path) -> Path:
    """
    单张截图的完整处理：读取裁剪 -> 保存裁剪预览 -> OCR -> 保存 txt。
    返回生成的 txt 路径。
    """
    cropped_img, _ = load_and_crop_image(img_path)

    crop_preview_path = OUT_DIR / f"{img_path.stem}_crop_preview.png"
    cv2.imwrite(str(crop_preview_path), cropped_img)

    text = ocr_cropped_image(cropped_img)
    out_txt_path = OUT_DIR / f"{img_path.stem}.txt"
    out_txt_path.write_text(text, encoding="utf-8")
    return out_txt_path


def main():
    workers = max(1, WORKERS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_files = sorted(IMG_DIR.glob("*.png"))
    if not img_files:
        print(f"{IMG_DIR} 目录下没有 png 图片。")
        return

    selected_files = img_files
    total = len(selected_files)
    print(f"将处理全部 {total} 张截图，并发数: {workers}。")
    print(f"CROP_BOX: {CROP_BOX}")

    done_count = 0
    failures: list[tuple[Path, Exception]] = []
    progress_lock = threading.Lock()

    def worker(img_path: Path) -> Path | None:
        nonlocal done_count
        try:
            result = process_one_image(img_path)
            error = None
        except Exception as exc:  # noqa: BLE001 - 记录失败图片，继续处理其余
            result = None
            error = exc
        with progress_lock:
            done_count += 1
            if error is None:
                print(f"[{done_count}/{total}] OCR 完成: {img_path.name}")
            else:
                failures.append((img_path, error))
                print(f"[{done_count}/{total}] OCR 失败: {img_path.name} ({error})")
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # map 保持与输入相同的顺序，汇总文件按文件名顺序拼接
        generated_txt_paths = list(executor.map(worker, selected_files))

    # 汇总所有 OCR 原样输出，并去掉空行
    merged_lines = []
    for txt_path in generated_txt_paths:
        if txt_path is None:
            continue
        raw = txt_path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            if line.strip():
                merged_lines.append(line)

    merged_path = OUT_DIR / "all_ocr_no_blank_lines.txt"
    merged_path.write_text("\n".join(merged_lines), encoding="utf-8")

    print("完成。输出目录：", OUT_DIR.resolve())
    print(f"每张截图输出: 原样 txt + crop_preview，成功 {total - len(failures)}/{total} 组。")
    if failures:
        print(f"失败 {len(failures)} 张：")
        for img_path, exc in failures:
            print(f"  {img_path.name}: {exc}")
    print("汇总文件（去空行）：", merged_path.resolve())


if __name__ == "__main__":
    main()