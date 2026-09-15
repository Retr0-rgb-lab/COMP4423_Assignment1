# Task 1: 调相机 / 读图像并显示 (5 分)

## 目标(摘自作业 PDF)
> Task 1: Call the camera to capture images and display them. (5 marks)

## 关键决策

- **当前用文件输入替代相机**: `code/sky.jpg` 作默认输入。Task 4 再替换为 `cv2.VideoCapture(0)`。
- **OpenCV** 做图像 IO 与显示: `cv2.imread` (BGR)、`cv2.imshow`、`cv2.waitKey(0)` 按任意键关闭。
- **默认路径用 `__file__` 解析**,不依赖当前工作目录;命令行第一个参数可覆盖。
- **异常处理**: `imread` 返回 `None` 时抛 `FileNotFoundError`,避免后续 `imshow` 静默失败。

## Prompt 草稿

(本次未用 GenAI;Task 2 起开始记录 prompt 草稿)

## 代码改动

- 新增 `code/capture.py`(约 22 行): 读图 → 校验 → 打印 shape → 弹窗 → 等待键退出。
- `code/sky.jpg`: Task 1 的输入文件(测试用图)。

## 问题与解决方案

- **WSL 环境无 GUI**: `cv2.imshow` 在当前 shell (WSL) 无法弹窗,必须在 Windows 终端跑。
- **opencv 可能未装**: `../venv` 基础包只含 torch / d2l / numpy,OpenCV 需额外装(见 §验证)。

## 验证(在 Windows 终端跑)

```bash
cd "D:\Program Files\learn_torch\python-cv\Assignment1"
..\venv\Scripts\activate            # PowerShell
python code/capture.py              # 用默认 sky.jpg
python code/capture.py code/sky.jpg # 显式传参
```

预期:
- 弹出窗口显示 sky.jpg。
- 终端打印 `input = ... shape = (H, W, 3) dtype = uint8`。
- 按任意键窗口关闭、程序退出。

如果 `ModuleNotFoundError: No module named 'cv2'`,执行:

```bash
pip install opencv-python
```

## 已知局限 / 待办

- 仍未接真实相机(留给 Task 4)。
- 未做 resize / 灰度 / 颜色转换(T1 不要求)。
- 单张图,不循环(T4 再加 while 循环)。
