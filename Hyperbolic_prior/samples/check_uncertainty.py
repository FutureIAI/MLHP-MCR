import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ================= 配置区域 =================
# 请修改为你的实际文件路径
npy_path = "/home/cgz/lzc/FPC_Dataset/img0002_ring_uncertainty.npy"
image_path = "/home/cgz/lzc/FPC_Dataset/img0002_ring.jpg"


# ===========================================

def visualize_uncertainty(npy_file, img_file=None):
    if not os.path.exists(npy_file):
        print(f"错误: 找不到文件 {npy_file}")
        return

    # 1. 加载数据
    uncertainty = np.load(npy_file)

    # 如果有 batch 维度，去掉它
    if len(uncertainty.shape) == 3:
        uncertainty = uncertainty[0]

    # 打印一些统计信息，帮你确认数据是否正常
    print(f"--- 不确定性度量统计 ---")
    print(f"维度: {uncertainty.shape}")
    print(f"最小值 (最确定): {uncertainty.min():.4f}")
    print(f"最大值 (最不确定): {uncertainty.max():.4f}")
    print(f"平均值: {uncertainty.mean():.4f}")

    # 2. 绘图
    plt.figure(figsize=(12, 6))

    # 如果提供了原图，做对比显示
    if img_file and os.path.exists(img_file):
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        # 使用 'magma' 或 'viridis'，这类色表对不确定性表现力最强
        #im = plt.imshow(uncertainty, cmap='magma', vmin=0.2380, vmax=0.2400)
        im = plt.imshow(uncertainty, cmap='magma')  # 删掉 vmin 和 vmax
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Uncertainty Heatmap (Higher is less certain)")
        plt.axis('off')
    else:
        # 只显示不确定性图
        im = plt.imshow(uncertainty, cmap='magma')
        plt.colorbar(im)
        plt.title("Uncertainty Heatmap")
        plt.axis('off')

    plt.tight_layout()

    # 保存结果
    save_path = npy_file.replace('.npy', '_heatmap.png')
    plt.savefig(save_path, dpi=200)
    print(f"可视化结果已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    visualize_uncertainty(npy_path, image_path)