import numpy as np
import matplotlib.pyplot as plt

# 1. 加载你生成的 gradient_uncertainty.npy 数据
# 请替换为你的实际文件路径
data = np.load("/home/cgz/lzc/HyperbolicImageSegmentation-main/samples/enhanced_features/img0022_ring_gradient_uncertainty.npy")

# 2. 【核心动作】：切除边缘伪影
# 注意：这只是为了可视化和找阈值，并没有破坏你的原数据！
crop = 20
cropped_data = data[crop:-crop, crop:-crop]

print("================ 数值诊断报告 ================")
print(f"内部真实最小值: {cropped_data.min():.6f}")
print(f"内部真实最大值: {cropped_data.max():.6f}")
print(f"内部平均值:   {cropped_data.mean():.6f}")

# 3. 动态寻找三个候选阈值
# 因为你的形变阴影是“变黑”的（低梯度），所以我们要找的是分布最底部的数值
t_strict = np.percentile(cropped_data, 2)  # 最严格阈值 (仅捕捉最严重的阴影，约最低的2%)
t_mid = np.percentile(cropped_data, 10)    # 中等阈值 (捕捉最低的 10%)
t_loose = np.percentile(cropped_data, 25)  # 宽松阈值 (捕捉最低的 25%)

print(f"\n💡 推荐候选阈值:")
print(f"T1 (严苛 - 2%):  {t_strict:.6f}")
print(f"T2 (适中 - 10%): {t_mid:.6f}")
print(f"T3 (宽松 - 25%): {t_loose:.6f}")
print("==============================================")

# 4. 画图大阅兵
plt.figure(figsize=(20, 5))

# 第一张图：重见天日的内部热力图
plt.subplot(1, 4, 1)
# 动态拉满内部对比度，这样你就能看到走线了！
vmin, vmax = np.percentile(cropped_data, 2), np.percentile(cropped_data, 98)
plt.imshow(cropped_data, cmap='magma', vmin=vmin, vmax=vmax)
plt.title("Cropped Heatmap (Internal Contrast)", fontsize=14)
plt.axis('off')

# 第二张图：严苛 Mask
plt.subplot(1, 4, 2)
plt.imshow(cropped_data < t_strict, cmap='gray')
plt.title(f"Mask T1: < {t_strict:.5f}", fontsize=14)
plt.axis('off')

# 第三张图：中等 Mask
plt.subplot(1, 4, 3)
plt.imshow(cropped_data < t_mid, cmap='gray')
plt.title(f"Mask T2: < {t_mid:.5f}", fontsize=14)
plt.axis('off')

# 第四张图：宽松 Mask
plt.subplot(1, 4, 4)
plt.imshow(cropped_data < t_loose, cmap='gray')
plt.title(f"Mask T3: < {t_loose:.5f}", fontsize=14)
plt.axis('off')

plt.tight_layout()
# 建议保存下来放大看
plt.savefig("threshold_diagnosis.jpg", dpi=200, bbox_inches='tight')
print("🎉 诊断图已生成，快去看看 threshold_diagnosis.jpg 吧！")