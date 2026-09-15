"""
Task 1: 调相机 / 读图像并显示 (5 分)

PDF 要求:"Call the camera to capture images and display them."
当前实现: 先用 sky.jpg 作为输入文件 (Task 4 替换为 cv2.VideoCapture(0))。
"""
import cv2
import sys
import os

# 默认输入: 与本脚本同目录的 sky.jpg (相对 cwd 无关)
INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sky.jpg")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT
    img = cv2.imread(path)                       # BGR, ndarray
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    print(f"input = {path}  shape = {img.shape}  dtype = {img.dtype}")
    cv2.imshow("Task 1", img)                    # 弹窗显示
    cv2.waitKey(0)                               # 0 = 等任意键
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
