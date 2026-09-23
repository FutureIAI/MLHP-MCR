import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def generate_uncertainty_heatmap(npy_path, original_img_path=None, c=0.1, output_path="final_uncertainty_heatmap.png"):
    """
    读取提取的特征 .npy 文件，计算并绘制真实的不确定性热力图。

    参数:
        npy_path: 你的 .npy 特征文件路径 (比如 original_euclidean.npy)
        original_img_path: (可选) 原图路径。如果提供，热力图会半透明叠加在原图上！
        c: 双曲空间的曲率，默认 0.1 (必须与你训练时保持一致)
        output_path: 生成的热力图保存路径
    """
    print(f"📂 正在加载特征文件: {npy_path}")
    if not os.path.exists(npy_path):
        print("❌ 找不到 .npy 文件，请检查路径！")
        return

    # 1. 加载特征
    features = np.load(npy_path)

    # 兼容性处理：把可能的 (1, H, W, 256) 变成 (H, W, 256)
    if len(features.shape) == 4:
        features = features[0]

    H, W, D = features.shape
    print(f"✅ 成功读取特征，尺寸为: {H}x{W}，特征维度: {D}")

    # 2. 计算“自信心” (特征在空间里的距离/范数)
    confidence = np.linalg.norm(features, axis=-1)

    # 3. 【核心逻辑】：将“自信心”翻转为“不确定性”
    # 庞加莱球的最大半径 R = 1 / sqrt(c)
    max_radius = 1.0 / np.sqrt(c)

    # 不确定性 = 最大理论半径 - 当前自信心
    uncertainty = max_radius - confidence

    # 防止因为极小浮点误差出现负数
    uncertainty = np.clip(uncertainty, 0, max_radius)

    # 4. 【视觉魔法】：使用百分位数过滤极端噪点，拉满对比度
    # 丢掉最低 1% 和最高 1% 的极值，防止几个坏死像素拉低了整张图的亮度
    vmin = np.percentile(uncertainty, 1)
    vmax = np.percentile(uncertainty, 99)
    print(f"📊 不确定性数值范围: 最小约 {vmin:.4f}, 最大约 {vmax:.4f}")

    # 5. 开始画图
    plt.figure(figsize=(12, 8))

    if original_img_path and os.path.exists(original_img_path):
        # 如果提供了原图，做半透明叠加叠图
        img = cv2.imread(original_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))  # 确保尺寸匹配

        plt.imshow(img)
        # alpha=0.6 表示热力图有 60% 的不透明度，cmap='magma' 是黑紫到明黄的渐变
        heatmap = plt.imshow(uncertainty, cmap='magma', alpha=0.6, vmin=vmin, vmax=vmax)
        plt.title("Uncertainty Heatmap (Overlay on Original Image)", fontsize=16)
    else:
        # 只画纯粹的热力图
        heatmap = plt.imshow(uncertainty, cmap='magma', vmin=vmin, vmax=vmax)
        plt.title("Pure Uncertainty Heatmap", fontsize=16)

    # 添加颜色条
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Uncertainty Level (Higher = Anomaly)", fontsize=12)
    plt.axis('off')

    # 保存并关闭
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 搞定！高亮热力图已保存至: {output_path}")


# ==================== 使用方法 ====================
if __name__ == "__main__":
    # 请在这里填入你刚刚生成的 .npy 文件的实际路径
    # (比如上一轮保存在 enhanced_features 文件夹里的 original_euclidean.npy)
    MY_NPY_FILE = "/home/cgz/lzc/HyperbolicImageSegmentation-main/samples/enhanced_features/img0022_ring_original_euclidean.npy"
    # (可选) 填入这张图对应的原图路径，不填(设为None)就只画纯热力图
    MY_IMAGE_FILE = "你的原图路径.jpg"  # 或者填 None

    generate_uncertainty_heatmap(
        npy_path=MY_NPY_FILE,
        original_img_path=MY_IMAGE_FILE,
        c=0.1,  # 曲率保持你训练时的 0.1
        output_path="perfect_heatmap.jpg"
    )

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def generate_texture_break_heatmap(npy_path, original_img_path=None, output_path="defect_gradient_heatmap.jpg"):
    print(f"📂 正在加载特征文件: {npy_path}")
    if not os.path.exists(npy_path):
        print("❌ 找不到 .npy 文件！")
        return

    # 1. 加载特征
    features = np.load(npy_path)
    if len(features.shape) == 4:
        features = features[0]
    H, W, D = features.shape

    # 2. 【核心修复一】：切除 CNN 边缘伪影 (Padding Artifacts)
    # 裁掉四周 15 个像素，防止边缘的异常值破坏对比度
    crop = 15
    cropped_features = features[crop:-crop, crop:-crop, :]

    if original_img_path and os.path.exists(original_img_path):
        img = cv2.imread(original_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        cropped_img = img[crop:-crop, crop:-crop, :]
    else:
        cropped_img = None

    print(f"✂️ 边缘伪影已切除，有效计算区域: {cropped_features.shape[:2]}")

    # 3. 【核心算法转换】：计算高维特征的“局部突变程度（空间梯度）”
    # 如果纹理正常，相邻像素特征差异极小；如果有缺陷/形变，特征差异会瞬间飙升！

    # 计算 X 方向和 Y 方向的特征差异
    diff_y = np.diff(cropped_features, axis=0)  # 垂直方向特征差异
    diff_x = np.diff(cropped_features, axis=1)  # 水平方向特征差异

    # 补齐维度以匹配图像大小 (复制最后一行/列)
    diff_y = np.vstack([diff_y, diff_y[-1:]])
    diff_x = np.hstack([diff_x, diff_x[:, -1:]])

    # 计算 256 维空间中的欧氏距离变化量作为“异常突变度”
    gradient_magnitude = np.sqrt(np.sum(diff_y ** 2 + diff_x ** 2, axis=-1))

    # 4. 再次使用百分位数过滤极值拉满对比度
    vmin = np.percentile(gradient_magnitude, 2)
    vmax = np.percentile(gradient_magnitude, 98)  # 压低 vmax，让微小缺陷也能发光

    print(f"📊 特征突变强度范围: 最小 {vmin:.4f}, 最大 {vmax:.4f}")

    # 5. 画图
    plt.figure(figsize=(12, 8))

    if cropped_img is not None:
        plt.imshow(cropped_img)
        heatmap = plt.imshow(gradient_magnitude, cmap='magma', alpha=0.6, vmin=vmin, vmax=vmax)
        plt.title("Texture Break Heatmap (Overlay)", fontsize=16)
    else:
        heatmap = plt.imshow(gradient_magnitude, cmap='magma', vmin=vmin, vmax=vmax)
        plt.title("Pure Texture Break Heatmap", fontsize=16)

    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Feature Gradient (Higher = Texture Break / Defect)", fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 搞定！特征突变热力图已保存至: {output_path}")


if __name__ == "__main__":
    # 填入你的文件路径
    MY_NPY_FILE = "/home/cgz/lzc/HyperbolicImageSegmentation-main/samples/enhanced_features/img0002_ring_enhanced_3d.npy"
    #MY_IMAGE_FILE = "/home/cgz/lzc/FPC_Dataset/img0002_ring.jpg"  # 或者填 None
    MY_IMAGE_FILE = None  # 或者填 None

    generate_texture_break_heatmap(
        npy_path=MY_NPY_FILE,
        original_img_path=MY_IMAGE_FILE,
        output_path="feature_gradient_heatmap.jpg"
    )