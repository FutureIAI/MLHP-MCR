import numpy as np
import matplotlib.pyplot as plt

# 读取你生成的 3D 特征文件
feat_3d = np.load("/home/cgz/lzc/HyperbolicImageSegmentation-main/samples/enhanced_features/img0022_bar_enhanced_3d.npy")

# 如果多了一个 batch 维度 (1, H, W, 3)，去掉它
if len(feat_3d.shape) == 4:
    feat_3d = feat_3d[0]

# 将数值归一化到 0-1 之间，以便作为 RGB 图像显示
f_min = feat_3d.min(axis=(0, 1), keepdims=True)
f_max = feat_3d.max(axis=(0, 1), keepdims=True)
feat_rgb = (feat_3d - f_min) / (f_max - f_min + 1e-8)

# 直接当成彩色图片画出来！
plt.figure(figsize=(12, 8))
plt.imshow(feat_rgb)
plt.title("Direct Visualization of 3D PCA Features (Fed to Transformer)", fontsize=16)
plt.axis('off')
plt.show()