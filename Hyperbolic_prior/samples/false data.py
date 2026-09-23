import os
import cv2
import random
import numpy as np
import glob

# 1. 设置你刚刚规划好的路径 (根据你的实际路径进行修改)
NORMAL_IMG_DIR = '/home/cgz/lzc/FPC_Dataset/FPC/false_data/normal'
DEFORMED_IMG_DIR = '/home/cgz/lzc/FPC_Dataset/FPC/false_data/deformed'
OUTPUT_DIR = '/home/cgz/lzc/FPC_Dataset/FPC/false_data/new_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 获取图片列表 (安全模式)
normal_img_paths = glob.glob(os.path.join(NORMAL_IMG_DIR, '*.jpg')) + \
                   glob.glob(os.path.join(NORMAL_IMG_DIR, '*.png'))
deformed_img_paths = glob.glob(os.path.join(DEFORMED_IMG_DIR, '*.jpg')) + \
                     glob.glob(os.path.join(DEFORMED_IMG_DIR, '*.png'))

print(f"找到 {len(normal_img_paths)} 张正常图片，{len(deformed_img_paths)} 张形变图片。")
if len(normal_img_paths) == 0 or len(deformed_img_paths) == 0:
    print("❌ 错误：没有找到图片，请检查路径！")
    exit()


# ---------------------------------------------------------
# 【核心黑科技】：纯 Numpy 手写版 FDA 傅里叶光影迁移
# ---------------------------------------------------------
def fda_transfer(src_img, ref_img, beta=0.05):
    """
    将 ref_img (形变图) 的低频光影(高光/阴影) 转移到 src_img (正常图) 上。
    beta: 控制低频替换的大小，建议 0.01 ~ 0.09，值越大阴影/高光越重。
    """
    # 确保两张图尺寸一致才能做频域替换
    h, w, c = src_img.shape
    ref_img = cv2.resize(ref_img, (w, h))

    # 1. 对两张图进行 2D 傅里叶变换
    src_fft = np.fft.fft2(src_img, axes=(0, 1))
    src_fft_shift = np.fft.fftshift(src_fft, axes=(0, 1))

    ref_fft = np.fft.fft2(ref_img, axes=(0, 1))
    ref_fft_shift = np.fft.fftshift(ref_fft, axes=(0, 1))

    # 2. 计算需要替换的低频中心区域的大小
    b_h, b_w = int(h * beta), int(w * beta)
    c_h, c_w = h // 2, w // 2

    # 3. 把正常图的低频中心区域，替换成形变图的低频中心区域
    src_fft_shift[c_h - b_h: c_h + b_h, c_w - b_w: c_w + b_w, :] = \
        ref_fft_shift[c_h - b_h: c_h + b_h, c_w - b_w: c_w + b_w, :]

    # 4. 逆傅里叶变换，还原回图像
    src_fft_ishift = np.fft.ifftshift(src_fft_shift, axes=(0, 1))
    src_ifft = np.fft.ifft2(src_fft_ishift, axes=(0, 1))

    # 5. 取实部，限制在 0-255 像素范围内，并转回 uint8
    result = np.real(src_ifft)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------

# 3. 批量处理
for normal_path in normal_img_paths:
    img_normal = cv2.imread(normal_path)
    if img_normal is None: continue

    # 随机抽一张形变图
    img_ref = None
    while img_ref is None:
        ref_path = random.choice(deformed_img_paths)
        img_ref = cv2.imread(ref_path)

    # 调用我们手写的 FDA 函数 (这里我默认设 beta=0.04，你可以微调)
    synthetic_img_bgr = fda_transfer(img_normal, img_ref, beta=0.04)

    # 保存图片
    filename = os.path.basename(normal_path)
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, synthetic_img_bgr)

print("🎉 纯 Numpy 版合成完成！没有依赖任何新库，请去输出文件夹查看结果。")