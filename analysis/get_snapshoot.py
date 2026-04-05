import time
from pathlib import Path

import pyautogui


# =========================
# 配置项
# =========================
OUTPUT_DIR = Path.cwd() / "snapshoot_all"
WAIT_BEFORE_START = 5.0
WAIT_AFTER_PAGEDOWN = 1.5
MAX_SHOTS = 8446
FILE_FMT = "{:06d}.png"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_next_index() -> int:
    existing = sorted(OUTPUT_DIR.glob("*.png"))
    if not existing:
        return 1

    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem))
        except ValueError:
            pass

    return max(nums) + 1 if nums else 1


def capture_fullscreen(save_path: Path) -> None:
    img = pyautogui.screenshot()
    img.save(save_path)


def main() -> None:
    pyautogui.FAILSAFE = True

    ensure_output_dir()
    idx = get_next_index()

    print(f"请在 {WAIT_BEFORE_START} 秒内手动完成这些操作：")
    print("1. 把远程桌面切到全屏")
    print("2. 把目标 txt 窗口放到最前")
    print("3. 滚动到起始位置")
    print("开始后脚本会：截图 -> PageDown -> 等 1 秒 -> 继续")
    print("停止方法：Ctrl+C，或把鼠标移到屏幕左上角")
    time.sleep(WAIT_BEFORE_START)

    shot_count = 8288
    try:
        while shot_count < MAX_SHOTS:
            save_path = OUTPUT_DIR / FILE_FMT.format(idx)
            capture_fullscreen(save_path)
            print(f"已保存: {save_path.name}")

            shot_count += 1
            idx += 1

            pyautogui.press("pagedown")
            time.sleep(WAIT_AFTER_PAGEDOWN)

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，脚本已停止。")
    except pyautogui.FailSafeException:
        print("\n触发安全停止，脚本已停止。")

    print(f"共保存 {shot_count} 张截图到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()