import os
import sys
import argparse
import logging
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from hesp.config.config import Config
from hesp.models.model import ModelFactory
from hesp.util.hyperbolic_nn import tf_exp_map_zero, tf_log_map_zero_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def np_exp_map_zero(v, c):
    """指数映射：欧氏 -> 双曲 (NumPy版)"""
    norm_v = np.linalg.norm(v, axis=-1, keepdims=True)
    sqrt_c = np.sqrt(c)
    eps = 1e-10
    # Formula: tanh(sqrt(c) * |v|) / (sqrt(c) * |v|) * v
    coeff = np.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v + eps)
    return coeff * v

def np_log_map_zero(y, c):
    """对数映射：双曲 -> 欧氏 (NumPy版)"""
    norm_y = np.linalg.norm(y, axis=-1, keepdims=True)
    sqrt_c = np.sqrt(c)
    eps = 1e-10
    # Formula: arctanh(sqrt(c) * |y|) / (sqrt(c) * |y|) * y
    dist = np.clip(sqrt_c * norm_y, -1 + eps, 1 - eps)
    coeff = np.arctanh(dist) / (sqrt_c * norm_y + eps)
    return coeff * y

class FeatureEnhancer:
    """特征增强器：提取特征 -> 双曲空间增强 -> 返回欧氏空间"""

    def __init__(self, checkpoint_dir, config_params):
        """
        初始化特征增强器

        Args:
            checkpoint_dir: 预训练模型的检查点目录
            config_params: 配置参数字典
        """
        self.checkpoint_dir = checkpoint_dir
        self.config_params = config_params
        self.model = None
        self.session = None
        self.graph = tf.Graph()

    def setup_model(self):
        """设置模型和计算图"""
        with self.graph.as_default():
            # 创建配置对象
            config = Config(
                dataset=self.config_params['dataset'],
                base_save_dir="",
                gpu_idx=self.config_params.get('gpu_idx', 0),
                mode='segmenter'
            )

            # 设置嵌入空间参数
            config.embedding_space._GEOMETRY = self.config_params.get('geometry', 'hyperbolic')
            config.embedding_space._DIM = self.config_params.get('dim', 256)
            config.embedding_space._INIT_CURVATURE = self.config_params.get('c', 1.0)
            config.embedding_space._HIERARCHICAL = self.config_params.get('hierarchical', True)

            # 设置分割器参数
            config.segmenter._PRETRAINED_MODEL = self.config_params.get('backbone_init', '')
            config.segmenter._OUTPUT_STRIDE = self.config_params.get('output_stride', 16)
            config.segmenter._BACKBONE = self.config_params.get('backbone', 'resnet_v2_101')
            config.segmenter._BATCH_SIZE = 1  # 推理时batch_size=1
            config.segmenter._FREEZE_BACKBONE = False
            config.segmenter._FREEZE_BN = True
            config.segmenter._EFN_OUT_DIM = config.embedding_space._DIM

            # 创建模型
            self.model = ModelFactory.create(mode='segmenter', config=config)
            self.model._init_predict()

            # 加载检查点
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            if not latest_checkpoint:
                raise ValueError(f"No checkpoint found in {self.checkpoint_dir}")

            saver = tf.train.Saver()
            self.session = tf.Session(graph=self.graph)
            saver.restore(self.session, latest_checkpoint)

            logger.info(f"Model loaded from {latest_checkpoint}")

    def extract_features(self, image):
        """提取图像特征"""
        with self.graph.as_default():
            from hesp.util.data_helpers import preprocess_image

            # === 还原：使用最稳定的常量编译，配合外层的50轮销毁，绝不溢出 ===
            def preprocess_fn(img):
                img_tensor = tf.constant(img, dtype=tf.float32)
                label_tensor = tf.zeros([tf.shape(img_tensor)[0], tf.shape(img_tensor)[1], 1], dtype=tf.int32)
                processed_img, _ = preprocess_image(
                    img_tensor, label_tensor, is_training=False, config=self.model.config
                )
                return processed_img

            processed_image = self.session.run(preprocess_fn(image))

            if len(processed_image.shape) == 3:
                processed_image = processed_image[None, ...]  # 变为 (1, H, W, 3)

            dummy_label = np.zeros((processed_image.shape[0], processed_image.shape[1], processed_image.shape[2], 1))
            results = self.model.predict(processed_image, dummy_label)

            # test_fn() 返回的 results["embeddings"] 实际对应
            # projected_embeddings，模型内部已经完成一次欧氏 -> 双曲指数映射。
            # 因此这里直接使用该双曲特征，不能再次调用 np_exp_map_zero()。
            hyperbolic_features = results['embeddings']

            probabilities = results['probabilities']
            return hyperbolic_features, probabilities

    def enhance_in_hyperbolic_space(self, hyperbolic_features, enhancement_type='mean', **kwargs):
        """
        在双曲空间中增强特征

        Args:
            hyperbolic_features: 双曲空间特征 (H, W, D)
            enhancement_type: 增强类型 ['mean', 'interpolate', 'scale']
            **kwargs: 增强参数

        Returns:
            enhanced_hyperbolic: 增强后的双曲特征
        """
        if enhancement_type == 'mean':
            # 双曲平均：对局部区域进行双曲平均
            return self.hyperbolic_mean(hyperbolic_features, **kwargs)
        elif enhancement_type == 'interpolate':
            # 双曲插值：在两个特征之间进行双曲插值
            return self.hyperbolic_interpolation(hyperbolic_features, **kwargs)
        elif enhancement_type == 'scale':
            # 双曲缩放：沿测地线缩放特征
            return self.hyperbolic_scaling(hyperbolic_features, **kwargs)
        else:
            raise ValueError(f"Unknown enhancement type: {enhancement_type}")

    def hyperbolic_mean(self, hyperbolic_features, kernel_size=3):
        """双曲平均：使用 NumPy 实现切空间平均"""
        # 获取曲率
        c = self.config_params.get('c', 0.1)

        # 移除 batch 维度 (如果是 4D)
        if len(hyperbolic_features.shape) == 4:
            hyperbolic_features = hyperbolic_features[0]

        H, W, D = hyperbolic_features.shape
        pad = kernel_size // 2
        padded = np.pad(hyperbolic_features, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

        enhanced = np.zeros_like(hyperbolic_features)

        for i in range(H):
            for j in range(W):
                # 获取局部区域 (k*k, D)
                patch = padded[i:i + kernel_size, j:j + kernel_size, :]
                patch_flat = patch.reshape(-1, D)

                # --- 关键修正：使用 NumPy 版函数代替 tf_ 版 ---
                # 1. 对数映射到切空间
                tangent_vectors = np_log_map_zero(patch_flat, c)

                # 2. 在切空间计算算术平均 (现在这里是真正的 NumPy 数组了)
                tangent_mean = np.mean(tangent_vectors, axis=0, keepdims=True)

                # 3. 指数映射回双曲空间
                h_mean = np_exp_map_zero(tangent_mean, c)

                enhanced[i, j, :] = h_mean[0]

        return enhanced

    def hyperbolic_interpolation(self, hyperbolic_features, alpha=0.5, reference_feature=None):
        """双曲插值：NumPy 实现"""
        c = self.config_params.get('c', 0.1)

        if len(hyperbolic_features.shape) == 4:
            hyperbolic_features = hyperbolic_features[0]

        H, W, D = hyperbolic_features.shape
        enhanced = np.zeros_like(hyperbolic_features)

        for i in range(H):
            for j in range(W):
                current_feature = hyperbolic_features[i, j, :]

                # 1. 对数映射
                log_map = np_log_map_zero(current_feature[None, :], c)[0]

                # 2. 线性插值
                interpolated_tangent = alpha * log_map

                # 3. 指数映射
                enhanced_point = np_exp_map_zero(interpolated_tangent[None, :], c)[0]

                enhanced[i, j, :] = enhanced_point

        return enhanced

    def hyperbolic_to_euclidean(self, hyperbolic_features):
        """将双曲空间特征转换回欧氏空间"""
        c_val = self.config_params['c']

        # 直接使用刚才定义的 NumPy 版对数映射
        # 不再需要 self.session.run，也不再需要 tf_log_map_zero_batch
        euclidean_features = np_log_map_zero(hyperbolic_features, c_val)

        return euclidean_features

    def process_image(self, image_path, enhancement_type='mean', save_output=True):
        """
        处理单张图像：提取特征 -> 双曲增强 -> 返回欧氏特征

        Args:
            image_path: 输入图像路径
            enhancement_type: 增强类型
            save_output: 是否保存输出

        Returns:
            enhanced_3d: PCA降维后的三通道增强特征 (H, W, 3)
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.info(f"Processing image: {image_path}")
        logger.info(f"Image shape: {image.shape}")

        # 提取模型内部已经完成指数映射的双曲特征
        hyperbolic_features, probabilities = self.extract_features(image)

        # 双曲空间增强
        enhanced_hyperbolic = self.enhance_in_hyperbolic_space(
            hyperbolic_features,
            enhancement_type=enhancement_type
        )

        # 转换回欧氏空间
        enhanced_euclidean = self.hyperbolic_to_euclidean(enhanced_hyperbolic)
        # ==================== 【在此处插入：PCA 256维 -> 3维】 ====================
        enh_squeeze = enhanced_euclidean[0] if len(enhanced_euclidean.shape) == 4 else enhanced_euclidean
        H, W, D = enh_squeeze.shape

        # 展平以便进行 PCA
        enh_flat = enh_squeeze.reshape(-1, D)

        logger.info("正在执行专属 PCA 降维 (Enhanced 256D -> 3D)...")
        pca_transformer = PCA(n_components=3)
        pca_transformer.fit(enh_flat)

        # 转换并变回图像的宽高
        enhanced_3d = pca_transformer.transform(enh_flat).reshape(H, W, 3)
        # ===========================================================================
        if save_output:
            # 定义目标文件夹路径
            base_dir = os.path.dirname(image_path)
            enhanced_3d_dir = os.path.join(base_dir, "2026FPC_Selected_332_enhanced_3d")

            # 确保文件夹存在
            os.makedirs(enhanced_3d_dir, exist_ok=True)

            basename = os.path.basename(image_path).split('.')[0]

            # 保存三通道特征增强文件
            np.save(os.path.join(enhanced_3d_dir, f"{basename}_enhanced_3d.npy"), enhanced_3d)

            logger.info(f"Saved 3-channel enhanced .npy file for {basename}")

            # 移除了可视化图表的生成调用
            # self.visualize_results(image, enhanced_euclidean, enhanced_euclidean, basename)

        return enhanced_3d


    def visualize_results(self, image, original_features, enhanced_features, basename):
        """可视化结果"""
        output_dir = "enhanced_features"

        # 计算特征范数（置信度）
        original_norm = np.linalg.norm(original_features, axis=-1)
        enhanced_norm = np.linalg.norm(enhanced_features, axis=-1)

        # 绘制结果
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # 原始特征范数
        im1 = axes[0, 1].imshow(original_norm, cmap='hot')
        axes[0, 1].set_title('Original Feature Norm')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])

        # 增强后特征范数
        im2 = axes[0, 2].imshow(enhanced_norm, cmap='hot')
        axes[0, 2].set_title('Enhanced Feature Norm')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2])

        # 特征差异
        diff_norm = enhanced_norm - original_norm
        im3 = axes[1, 0].imshow(diff_norm, cmap='RdBu_r', vmin=-np.max(np.abs(diff_norm)),
                                vmax=np.max(np.abs(diff_norm)))
        axes[1, 0].set_title('Norm Difference (Enhanced - Original)')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])

        # PCA可视化（前3个主成分）
        from sklearn.decomposition import PCA

        # 重塑特征
        H, W, D = original_features.shape
        original_flat = original_features.reshape(-1, D)
        enhanced_flat = enhanced_features.reshape(-1, D)

        # 合并进行PCA
        combined = np.vstack([original_flat, enhanced_flat])
        pca = PCA(n_components=3)
        pca.fit(combined)

        original_pca = pca.transform(original_flat).reshape(H, W, 3)
        enhanced_pca = pca.transform(enhanced_flat).reshape(H, W, 3)

        # 归一化显示
        def normalize_for_display(pca_features):
            min_val = pca_features.min(axis=(0, 1), keepdims=True)
            max_val = pca_features.max(axis=(0, 1), keepdims=True)
            return (pca_features - min_val) / (max_val - min_val + 1e-8)

        axes[1, 1].imshow(normalize_for_display(original_pca))
        axes[1, 1].set_title('Original Features (PCA)')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(normalize_for_display(enhanced_pca))
        axes[1, 2].set_title('Enhanced Features (PCA)')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{basename}_visualization.png"), dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Results saved to {output_dir}/{basename}_*")


def main():
    parser = argparse.ArgumentParser(description="Extract and enhance features in hyperbolic space (Batch Mode)")

    # 必需参数
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to pretrained model checkpoint directory")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images to process")

    # 模型配置参数（应与训练时一致）
    parser.add_argument("--dataset", type=str, default="fpc",
                        choices=["coco", "pascal", "ade", "toy", "fpc"],
                        help="Dataset used for training")
    parser.add_argument("--geometry", type=str, default="hyperbolic",
                        choices=["euclidean", "hyperbolic"],
                        help="Geometry type")
    parser.add_argument("--dim", type=int, default=256,
                        help="Embedding space dimension")
    parser.add_argument("--c", type=float, default=0.1,
                        help="Curvature of hyperbolic space")

    # 增强参数
    parser.add_argument("--enhancement_type", type=str, default="mean",
                        choices=["mean", "interpolate", "scale"],
                        help="Type of enhancement in hyperbolic space")
    parser.add_argument("--gpu_idx", type=int, default=0,
                        help="GPU index to use")

    args = parser.parse_args()

    # 配置参数
    config_params = {
        'dataset': args.dataset,
        'geometry': args.geometry,
        'dim': args.dim,
        'c': args.c,
        'hierarchical': True,
        'gpu_idx': args.gpu_idx,
        'backbone': 'resnet_v2_101',
        'output_stride': 16,
    }

    try:
        # 获取目录下的所有文件并过滤出 .jpg 文件
        all_files = os.listdir(args.input_dir)
        jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]
        jpg_files.sort()

        total_files = len(jpg_files)
        logger.info(f"Found {total_files} PNG files to process.")
        print(f"[FeatureEnhancer] Found {total_files} PNG files in: {args.input_dir}", flush=True)
        if total_files == 0:
            raise ValueError(f"No PNG images found in input_dir: {args.input_dir}")

        # ================== 【终极绝招：设置安全处理批次】 ==================
        CHUNK_SIZE = 50  # 每处理50张，强制清理一次底层显存（绝对跑不到溢出的上限）

        for chunk_start in range(0, total_files, CHUNK_SIZE):
            chunk_files = jpg_files[chunk_start: chunk_start + CHUNK_SIZE]
            logger.info(
                f"\n========== ♻️ 正在启动新的计算图环境 (处理第 {chunk_start + 1} 到 {min(chunk_start + CHUNK_SIZE, total_files)} 张) ==========")

            # 1. 强行清空 TensorFlow 底层所有的残留幽灵节点
            tf.reset_default_graph()

            # 2. 重新初始化增强器（重建干净的模型蓝图）
            print(f"[FeatureEnhancer] Initializing model for chunk starting at {chunk_start + 1}...", flush=True)
            enhancer = FeatureEnhancer(args.checkpoint_dir, config_params)
            enhancer.setup_model()
            print("[FeatureEnhancer] Model initialized successfully.", flush=True)

            # 3. 处理这 50 张图片
            for i, filename in enumerate(chunk_files, 1):
                image_path = os.path.join(args.input_dir, filename)
                global_idx = chunk_start + i
                logger.info(f"[{global_idx}/{total_files}] Processing {filename}...")

                enhanced_3d = enhancer.process_image(
                    image_path,
                    enhancement_type=args.enhancement_type,
                    save_output=True
                )

            # 4. 极其关键：处理完 50 张后，强制关闭会话，销毁对象，归还 GPU 显存！
            enhancer.session.close()
            del enhancer

            # 呼叫 Python 垃圾回收车，彻底扫清内存
            import gc
            gc.collect()
            # ====================================================================

        logger.info("\n🎉 Batch feature extraction and enhancement completed successfully!")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()