"""FPC 多光源数据集与反光掩膜生成；按数据流职责组织代码，不改变数据接口。"""

import os
import random
import sys
from types import ModuleType

import cv2
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize


# 基础注册组件：在未安装完整 BasicSR 注册模块时提供兼容实现
class Registry:
    def __init__(self, n):
        self.name = n  # 规范化存储传入的名称参数，防止形参闲置
        self._d = {}

    def register(self, o=None):
        if o is None:
            return lambda x: self.register(x)
        self._d[o.__name__] = o
        return o

    def get(self, n, s='basicsr'):
        r = self._d.get(n) or self._d.get(f"{n}_{s}")
        if r is None:
            raise KeyError(n)
        return r

    def __contains__(self, n):
        return n in self._d


# BasicSR 注册表兼容注入
if 'basicsr.utils.registry' not in sys.modules:
    m = ModuleType('basicsr.utils.registry')
    m.Registry = Registry
    m.DATASET_REGISTRY, m.ARCH_REGISTRY, m.MODEL_REGISTRY, m.LOSS_REGISTRY, m.METRIC_REGISTRY = [
        Registry(i) for i in ['ds', 'arch', 'mod', 'loss', 'met']
    ]
    sys.modules['basicsr.utils.registry'] = m

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Dataset_FPC_18Channel(data.Dataset):
    """自定义24通道FPC板数据集：9通道三光源图像 + 9通道.npy特征增强先验 + 3通道全局无损阴影偏差 + 3通道高光区域掩膜"""

    # 1. 初始化与文件读取：配置三光源、先验目录并兼容图片扩展名
    def __init__(self, opt):
        super(Dataset_FPC_18Channel, self).__init__()
        # 禁用 OpenCV 进程内多线程与 OpenCL 内存缓存，防止多进程数据加载时触发 Segmentation fault 死锁
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_coax_folder = opt['dataroot_lq_coaxial']
        self.lq_bar_folder = opt['dataroot_lq_bar']
        self.lq_ring_folder = opt['dataroot_lq_ring']
        self.prior_folder = opt['dataroot_prior']
        self.paths = []

        if os.path.exists(self.gt_folder):
            for filename in sorted(os.listdir(self.gt_folder)):
                if filename.endswith('_coax.png') or filename.endswith('_coax.jpg'):
                    ext = '.png' if filename.endswith('.png') else '.jpg'
                    self.paths.append({'id': filename[:-9], 'ext': ext})

    def _read_img(self, folder, name):
        """ 作用：自动修复 .jp 截断并自适应匹配各种图片后缀 """
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            if path.endswith('.jp'):
                path += 'g'
            for e in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                if os.path.exists(os.path.splitext(path)[0] + e):
                    path = os.path.splitext(path)[0] + e
                    break

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _read_light_triplet(self, folders, img_id, ext):
        """按 coax、bar、ring 顺序读取三路光源图像。"""
        return tuple(self._read_img(folder, f'{img_id}_{light}{ext}')
                     for folder, light in zip(folders, ('coax', 'bar', 'ring')))

    def _load_prior_triplet(self, img_id):
        """按三路光源顺序读取增强先验特征。"""
        return tuple(np.load(os.path.join(
            self.prior_folder, f'{img_id}_{light}_enhanced_3d.npy'
        )).astype(np.float32) for light in ('coax', 'bar', 'ring'))

    # 2. 板面与阴影处理：提取有效板面、背景区域和全局阴影偏差
    def _fill_external_contours(self, mask_u8, minimum_area):
        """ 作用：填充有效板面外轮廓，同时剔除与板面无关的极小孤立区域 """
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(mask_u8)
        valid = [
            contour for contour in contours
            if cv2.contourArea(contour) >= minimum_area
        ]
        if valid:
            cv2.drawContours(filled, valid, -1, 255, thickness=cv2.FILLED)
        return filled

    def _refine_board_mask(self, board_mask, edge_shrink_px=3):
        """ 作用：填充板内孔洞并收缩板面边缘，避免背景边界进入反光检测 """
        mask_u8 = board_mask.astype(np.uint8) * 255
        mask_u8 = self._fill_external_contours(
            mask_u8, mask_u8.shape[0] * mask_u8.shape[1] * 0.001
        )
        if edge_shrink_px > 0 and cv2.countNonZero(mask_u8) > 0:
            ksize = 2 * edge_shrink_px + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (ksize, ksize)
            )
            mask_u8 = cv2.erode(mask_u8, kernel)
        return mask_u8 > 0

    def _compute_perfect_mask(self, lq_coax, lq_bar, lq_ring):
        """ 作用：采用OpenCV轮廓连通域孔洞填充法，消除FPC板内部由于白色丝印或镜面过曝产生的虚假背景洞 """
        lum_coax, lum_bar, lum_ring = np.mean(lq_coax, axis=2), np.mean(lq_bar, axis=2), np.mean(lq_ring, axis=2)
        bg_mask = ((lum_coax == 1.0) & (lum_bar == 1.0) & (lum_ring == 1.0)).astype(np.uint8)
        contours, _ = cv2.findContours((1 - bg_mask).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        global_board_mask = np.zeros_like(bg_mask, dtype=np.uint8)
        cv2.drawContours(global_board_mask, contours, -1, 1.0, thickness=-1)
        global_board_mask = global_board_mask.astype(np.float32)
        return global_board_mask, (1.0 - global_board_mask).astype(np.uint8)

    def _fill_shadow_holes(self, diff, kernel_size):
        """ 作用：利用形态学膨胀填充阴影区域内白色字体与细小金属线条导致的局部零值孔洞 """
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(diff, kernel)

    def _compute_global_drop(self, lq_coax, lq_bar, lq_ring):
        """ 作用：在大图全分辨率下进行背景光估计、前景提取与物理阴影偏差提取（步骤①②③） """
        h, w = lq_coax.shape[:2]
        k_size = int(min(h, w) * 0.65)
        if k_size % 2 == 0:
            k_size += 1

        lum_coax = np.mean(lq_coax, axis=2)
        lum_bar = np.mean(lq_bar, axis=2)
        lum_ring = np.mean(lq_ring, axis=2)

        global_board_mask, _ = self._compute_perfect_mask(lq_coax, lq_bar, lq_ring)

        img_drop_channels = []
        if self.opt['phase'] == 'train':
            current_smooth_k = random.choice([55, 65, 75])
        else:
            current_smooth_k = 65

        for lum in [lum_coax, lum_bar, lum_ring]:
            bg_stream = cv2.blur(lum, (k_size, k_size), borderType=cv2.BORDER_REFLECT)
            smooth_stream = cv2.blur(lum, (current_smooth_k, current_smooth_k), borderType=cv2.BORDER_REFLECT)
            diff = np.maximum(bg_stream - smooth_stream, 0.0)
            diff_dilated = self._fill_shadow_holes(diff, kernel_size=current_smooth_k)
            img_drop_channels.append(diff_dilated)

        return np.stack(img_drop_channels, axis=2) * global_board_mask[:, :, np.newaxis]

    # 3. 局部统计工具：平滑、局部极值、掩膜高斯滤波与细节提取
    def _smoothstep01(self, value):
        """ 作用：把连续反光证据平滑限制到0到1，避免额外的二值硬开关 """
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _local_max(self, value, radius):
        """ 作用：在给定像素容差内取局部最大值，使跨光源证据允许小范围位置偏差 """
        if radius <= 0:
            return value.astype(np.float32)
        size = 2 * int(radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        return cv2.dilate(value.astype(np.float32), kernel)

    def _local_mean(self, value, radius):
        """ 作用：计算局部均值，形成不受单个亮纹支配的参考亮度基线 """
        if radius <= 0:
            return value.astype(np.float32)
        size = 2 * int(radius) + 1
        return cv2.boxFilter(
            value.astype(np.float32), cv2.CV_32F, (size, size),
            normalize=True, borderType=cv2.BORDER_REFLECT101
        )

    def _build_masked_gaussian_cache(self, mask, sigmas):
        """ 作用：一次性缓存三光源共用的掩膜高斯分母，避免每个特征重复计算 """
        mask_f = mask.astype(np.float32)
        cache = {'mask': mask_f}
        for sigma in sigmas:
            cache[float(sigma)] = cv2.GaussianBlur(
                mask_f, (0, 0), sigma,
                borderType=cv2.BORDER_REFLECT101
            )
        return cache

    def _masked_gaussian(self, value, mask, sigma, cache=None):
        """ 作用：只在板面有效像素内归一化高斯平滑，防止白色背景污染低频 """
        mask_f = cache['mask'] if cache is not None else mask.astype(np.float32)
        valid = mask_f[..., None] if value.ndim == 3 else mask_f
        numerator = cv2.GaussianBlur(
            value * valid, (0, 0), sigma,
            borderType=cv2.BORDER_REFLECT101
        )
        denominator = cache[float(sigma)] if cache is not None else cv2.GaussianBlur(
            mask_f, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101
        )
        if value.ndim == 3:
            denominator = denominator[..., None]
        return numerator / np.maximum(denominator, 1e-6)

    def _build_fine_bright_detail(self, luminance, board_mask, blur_cache):
        """ 作用：在1到5像素尺度提取正向白亮细节，保留绿墨条纹及金属细小反光点 """
        fine = []
        for sigma in (0.75, 1.35, 2.40):
            local_low = self._masked_gaussian(
                luminance, board_mask, sigma, blur_cache
            )
            fine.append(np.maximum(luminance - local_low, 0.0))
        return np.maximum.reduce(fine) * board_mask.astype(np.float32)

    def _compute_highlight_board_mask(self, images_rgb):
        """ 作用：为反光检测单独构造板面掩膜，不改变阴影与先验使用的原有板面掩膜 """
        common_white = np.ones(images_rgb[0].shape[:2], dtype=bool)
        for image in images_rgb:
            rgb = np.clip(image.astype(np.float32), 0.0, 1.0)
            channel_min = np.min(rgb, axis=2)
            channel_range = np.max(rgb, axis=2) - channel_min
            common_white &= (
                self._smoothstep01((channel_min - 0.94) / 0.06)
                * (1.0 - self._smoothstep01(channel_range / 0.05)) > 0.80
            )
        seed = (~common_white).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
        board = self._fill_external_contours(
            seed, seed.shape[0] * seed.shape[1] * 0.001
        ) > 0
        return self._refine_board_mask(board)

    # 4. 反光特征：亮度、色度、纹理、白化和局部异常证据
    def _extract_highlight_features(
        self, image_rgb, board_mask, blur_cache,
        low_frequency_sigma=9.0, local_context_sigma=18.0
    ):
        """ 作用：提取亮度、色度、纹理、白色平台和本光源局部异常等连续证据 """
        rgb = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        luminance = lab[..., 0] / 100.0
        chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2) / 128.0
        low = self._masked_gaussian(
            luminance, board_mask, low_frequency_sigma, blur_cache
        )
        high = luminance - low
        grad_x = cv2.Sobel(
            luminance, cv2.CV_32F, 1, 0, ksize=3
        ) / 4.0
        grad_y = cv2.Sobel(
            luminance, cv2.CV_32F, 0, 1, ksize=3
        ) / 4.0
        gradient = cv2.magnitude(grad_x, grad_y)
        texture = self._masked_gaussian(
            np.abs(high) + 0.35 * gradient, board_mask,
            max(1.0, low_frequency_sigma / 3.0), blur_cache
        )

        local_mean = self._masked_gaussian(
            luminance, board_mask, local_context_sigma, blur_cache
        )
        local_square_mean = self._masked_gaussian(
            luminance * luminance, board_mask, local_context_sigma, blur_cache
        )
        local_std = np.sqrt(np.maximum(
            local_square_mean - local_mean * local_mean, 0.0
        ))
        local_inconsistency = self._smoothstep01(
            np.maximum(luminance - local_mean, 0.0)
            / (0.035 + 2.0 * local_std)
        )

        fine_bright_detail = self._build_fine_bright_detail(
            luminance, board_mask, blur_cache
        )
        fine_mean = self._masked_gaussian(
            fine_bright_detail, board_mask, 4.0, blur_cache
        )
        fine_square_mean = self._masked_gaussian(
            fine_bright_detail * fine_bright_detail,
            board_mask, 4.0, blur_cache
        )
        fine_std = np.sqrt(np.maximum(
            fine_square_mean - fine_mean * fine_mean, 0.0
        ))
        fine_detail_strength = self._smoothstep01(
            np.maximum(
                fine_bright_detail - 0.45 * fine_mean - 0.002, 0.0
            ) / (0.018 + 0.65 * fine_std)
        )
        local_chroma = self._masked_gaussian(
            chroma, board_mask, 3.0, blur_cache
        )
        local_chroma_drop = self._smoothstep01(
            np.maximum(local_chroma - chroma, 0.0)
            / (local_chroma + 0.018)
        )

        minimum_channel = np.min(rgb, axis=2)
        absolute_white = self._smoothstep01(
            (minimum_channel - 0.68) / 0.30
        )
        channel_clipping = np.mean(
            self._smoothstep01((rgb - 0.94) / 0.06), axis=2
        )
        neutral_support = 1.0 - self._smoothstep01(
            (chroma - 0.025) / 0.14
        )
        white_surface = self._smoothstep01(
            (luminance - 0.48) / 0.46
        ) * neutral_support
        valid = board_mask.astype(np.float32)
        return {
            'luminance': luminance * valid,
            'texture': texture * valid,
            'chroma': chroma * valid,
            'absolute_white': absolute_white * valid,
            'white_surface': white_surface * valid,
            'channel_clipping': channel_clipping * valid,
            'local_inconsistency': local_inconsistency * valid,
            'fine_detail_strength': fine_detail_strength * valid,
            'local_chroma_drop': local_chroma_drop * valid
        }

    # 5. 跨光源反光证据：错位容忍、双参考比较与单光源退化评分
    def _build_shift_tolerant_reference_envelope(
        self, first_reference, second_reference, tolerance_px, cache
    ):
        """ 作用：缓存并组合双参考容差包络，防止0到5像素错位和重复滤波 """
        cache = {} if cache is None else cache
        fine_radius = max(1, min(2, int(tolerance_px)))
        for reference in (first_reference, second_reference):
            key = id(reference)
            if key not in cache:
                cache[key] = {
                    'luminance_upper': self._local_max(
                        reference['luminance'], tolerance_px
                    ),
                    'texture_upper': self._local_max(
                        reference['texture'], tolerance_px
                    ),
                    'chroma_upper': self._local_max(
                        reference['chroma'], tolerance_px
                    ),
                    'white_upper': self._local_max(
                        reference['white_surface'], tolerance_px
                    ),
                    'absolute_white_upper': self._local_max(
                        reference['absolute_white'], tolerance_px
                    ),
                    'clipping_upper': self._local_max(
                        reference['channel_clipping'], tolerance_px
                    ),
                    'chroma_drop_upper': self._local_max(
                        reference['local_chroma_drop'], tolerance_px
                    ),
                    'fine_near': self._local_max(
                        reference['fine_detail_strength'], fine_radius
                    ),
                    'fine_wide': self._local_max(
                        reference['fine_detail_strength'], tolerance_px
                    ),
                    'luminance_context': self._local_mean(
                        reference['luminance'], tolerance_px
                    )
                }
        first, second = cache[id(first_reference)], cache[id(second_reference)]
        return {
            'luminance_upper': np.maximum(
                first['luminance_upper'], second['luminance_upper']
            ),
            'texture_consensus': np.minimum(
                first['texture_upper'], second['texture_upper']
            ),
            'texture_best': np.maximum(
                first['texture_upper'], second['texture_upper']
            ),
            'chroma_consensus': np.minimum(
                first['chroma_upper'], second['chroma_upper']
            ),
            'shared_white': np.minimum(
                first['white_upper'], second['white_upper']
            ),
            'absolute_white_upper': np.maximum(
                first['absolute_white_upper'], second['absolute_white_upper']
            ),
            'clipping_upper': np.maximum(
                first['clipping_upper'], second['clipping_upper']
            ),
            'chroma_drop_upper': np.maximum(
                first['chroma_drop_upper'], second['chroma_drop_upper']
            ),
            'fine_shared_near': np.sqrt(
                first['fine_near'] * second['fine_near']
            ),
            'fine_shared_wide': np.sqrt(
                first['fine_wide'] * second['fine_wide']
            ),
            'luminance_context': np.maximum(
                first['luminance_context'], second['luminance_context']
            )
        }

    def _build_single_light_degradation(
        self, current, first_reference, second_reference,
        board_mask, tolerance_px, reference_cache
    ):
        """ 作用：只生成当前一路相对双正常参考发生白化、局部突变和信息压缩的证据 """
        reference = self._build_shift_tolerant_reference_envelope(
            first_reference, second_reference, tolerance_px, reference_cache
        )
        luminance_gap = np.maximum(
            current['luminance'] - reference['luminance_upper'], 0.0
        )
        brightness_excess = self._smoothstep01(luminance_gap / 0.18)
        fine_context_gap = np.maximum(
            current['luminance'] - reference['luminance_context'], 0.0
        )
        fine_brightness_excess = self._smoothstep01(
            np.maximum(fine_context_gap - 0.004, 0.0) / 0.095
        )

        reference_chroma = reference['chroma_consensus']
        chroma_loss = self._smoothstep01(
            np.maximum(reference_chroma - current['chroma'], 0.0)
            / (reference_chroma + 0.025)
        )
        colored_material = self._smoothstep01(
            (reference_chroma - 0.035) / 0.13
        )
        absolute_white_excess = self._smoothstep01(
            np.maximum(
                current['absolute_white']
                - reference['absolute_white_upper'] - 0.015, 0.0
            ) / 0.36
        )
        clipping_excess = self._smoothstep01(
            np.maximum(
                current['channel_clipping']
                - reference['clipping_upper'] - 0.015, 0.0
            ) / 0.48
        )
        chroma_drop_excess = self._smoothstep01(
            np.maximum(
                current['local_chroma_drop']
                - reference['chroma_drop_upper'] - 0.008, 0.0
            ) / 0.40
        )
        colored_direction = np.maximum(chroma_loss, chroma_drop_excess)
        neutral_direction = np.maximum(
            absolute_white_excess, clipping_excess
        )
        directional_whitening = (
            colored_material * colored_direction
            + (1.0 - colored_material) * neutral_direction
        )
        neutral_fine_direction = np.maximum(
            clipping_excess,
            absolute_white_excess * self._smoothstep01(
                np.maximum(current['absolute_white'] - 0.16, 0.0) / 0.48
            )
        )
        fine_directional_whitening = (
            colored_material * colored_direction
            + (1.0 - colored_material) * neutral_fine_direction
        )

        reference_texture = reference['texture_consensus']
        texture_loss = self._smoothstep01(
            np.maximum(reference_texture - current['texture'], 0.0)
            / (reference_texture + 0.018)
        )
        preserved_texture = self._smoothstep01(
            current['texture'] / (reference['texture_best'] + 0.025)
        )
        flat_clipped_plateau = (
            current['channel_clipping']
            * (0.25 + 0.75 * current['local_inconsistency'])
            * (1.0 - preserved_texture)
        )
        neutral_damage = np.maximum(texture_loss, flat_clipped_plateau)
        colored_damage = np.maximum(chroma_loss, texture_loss)
        information_loss = (
            colored_material * colored_damage
            + (1.0 - colored_material) * neutral_damage
        )
        whitening = brightness_excess * directional_whitening
        common_white = np.clip(
            current['white_surface'] * reference['shared_white'], 0.0, 1.0
        )
        common_white_hard = common_white >= 0.52

        broad_confidence = (
            whitening
            * (0.12 + 0.88 * information_loss)
            * (0.25 + 0.75 * current['local_inconsistency'])
            * np.square(1.0 - common_white)
        )
        fine_cross_light_novelty = np.maximum(
            1.0 - reference['fine_shared_wide'],
            (1.0 - reference['fine_shared_near'])
            * fine_directional_whitening
        )
        fine_source_direction = np.maximum.reduce((
            fine_brightness_excess * fine_directional_whitening,
            colored_material * chroma_loss * current['local_chroma_drop'],
            (1.0 - colored_material) * clipping_excess
        ))
        fine_local_support = np.maximum(
            current['fine_detail_strength'],
            0.70 * current['local_inconsistency']
        )
        fine_highlight = (
            current['fine_detail_strength']
            * fine_cross_light_novelty
            * (0.30 + 0.70 * fine_brightness_excess)
            * (0.18 + 0.82 * fine_directional_whitening)
            * (0.35 + 0.65 * fine_local_support)
            * np.sqrt(np.clip(fine_source_direction, 0.0, 1.0))
            * np.square(1.0 - common_white)
        )

        broad_confidence = cv2.GaussianBlur(
            broad_confidence.astype(np.float32), (0, 0), 0.65
        )
        fine_highlight = cv2.GaussianBlur(
            fine_highlight.astype(np.float32), (0, 0), 0.22
        )
        broad_confidence[common_white_hard] = 0.0
        fine_highlight[common_white_hard] = 0.0
        valid = board_mask.astype(np.float32)
        return (
            np.clip(broad_confidence, 0.0, 1.0) * valid,
            np.clip(fine_highlight, 0.0, 1.0) * valid
        )

    # 6. 反光掩膜仲裁：自适应边界、生长连通域与独占光源归属
    def _adaptive_highlight_boundaries(
        self, score, board_mask, sensitivity, fine_branch=False
    ):
        """ 作用：分别为宽反光与细碎反光生成高置信核心和低置信生长边界 """
        values = score[board_mask]
        if values.size == 0:
            return np.inf, np.inf
        median = float(np.median(values))
        absolute_deviation = np.abs(values - median)
        mad_scale = float(np.median(absolute_deviation)) * 1.4826
        q25, q75, q90 = np.percentile(values, (25.0, 75.0, 90.0))
        robust_scale = max(
            mad_scale, (q75 - q25) / 1.349,
            (q90 - median) / 1.282, 1e-6
        )
        sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        strictness = (
            2.75 - 1.75 * sensitivity
            if fine_branch else 3.2 - 2.1 * sensitivity
        )
        evidence_floor = (
            0.060 - 0.045 * sensitivity
            if fine_branch else 0.135 - 0.090 * sensitivity
        )
        core_boundary = max(
            evidence_floor, median + strictness * robust_scale
        )
        minimum_growth = 0.006 if fine_branch else 0.015
        growth_ratio = (
            0.42 - 0.18 * sensitivity
            if fine_branch else 0.52 - 0.20 * sensitivity
        )
        growth_boundary = max(
            minimum_growth, core_boundary * growth_ratio
        )
        return growth_boundary, core_boundary

    def _grow_highlight_components(
        self, score, board_mask, boundaries
    ):
        """ 作用：只保留含高置信核心的低置信连通区域，覆盖反光内部并抑制孤立弱纹理 """
        growth_boundary, core_boundary = boundaries
        support = (score >= growth_boundary) & board_mask
        seeds = (score >= core_boundary) & support
        if not np.any(seeds):
            return np.zeros_like(board_mask, dtype=bool)
        count, labels = cv2.connectedComponents(
            support.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return np.zeros_like(board_mask, dtype=bool)
        seed_labels = np.unique(labels[seeds])
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size == 0:
            return np.zeros_like(board_mask, dtype=bool)
        return np.isin(labels, seed_labels)

    def _resolve_cross_light_ownership(
        self, masks, scores, tolerance_px
    ):
        """ 作用：冲突区域归属证据明显占优的光源，仅删除强度接近的共同反光 """
        masks = masks.astype(bool)
        scores = np.clip(scores.astype(np.float32), 0.0, 1.0)
        output = np.zeros_like(masks, dtype=bool)
        nearby_scores = np.stack([
            self._local_max(score, tolerance_px) for score in scores
        ])
        for index in range(3):
            other = np.any(
                np.delete(masks, index, axis=0), axis=0
            ).astype(np.float32)
            other_nearby = self._local_max(other, tolerance_px) > 0
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                masks[index].astype(np.uint8), connectivity=8
            )
            if count <= 1:
                continue
            for label_index in range(1, count):
                x, y, width, height, _ = stats[label_index]
                region = np.s_[y:y + height, x:x + width]
                component = labels[region] == label_index
                conflict_ratio = float(np.mean(
                    other_nearby[region][component]
                ))
                if conflict_ratio < 0.35:
                    output_region = output[index][region]
                    output_region[component] = True
                    continue

                current_values = scores[index][region][component]
                current_strength = (
                    0.55 * float(np.max(current_values))
                    + 0.45 * float(np.mean(current_values))
                )
                other_strength = 0.0
                for other_index in range(3):
                    if other_index == index:
                        continue
                    other_values = nearby_scores[other_index][region][component]
                    other_strength = max(
                        other_strength,
                        0.55 * float(np.max(other_values))
                        + 0.45 * float(np.mean(other_values))
                    )
                if current_strength >= 1.18 * other_strength + 0.006:
                    output_region = output[index][region]
                    output_region[component] = True

        collisions = np.sum(output, axis=0) > 1
        if np.any(collisions):
            order = np.argsort(scores, axis=0)
            winner = order[-1]
            best = np.take_along_axis(
                scores, winner[None, ...], axis=0
            )[0]
            second = np.take_along_axis(
                scores, order[-2][None, ...], axis=0
            )[0]
            decisive = collisions & (best >= 1.18 * second + 0.006)
            output[:, collisions] = False
            for index in range(3):
                output[index, decisive & (winner == index)] = True
        return output

    def _arbitrate_exclusive_highlight(
        self, broad_confidence, fine_confidence,
        board_mask, sensitivity, tolerance_px
    ):
        """ 作用：宽反光与细碎反光分别生长，再完成三光源独占来源仲裁 """
        broad_boundaries = [
            self._adaptive_highlight_boundaries(
                score, board_mask, sensitivity
            ) for score in broad_confidence
        ]
        fine_boundaries = [
            self._adaptive_highlight_boundaries(
                score, board_mask, sensitivity, True
            ) for score in fine_confidence
        ]
        broad_masks = np.stack([
            self._grow_highlight_components(score, board_mask, boundary)
            for score, boundary in zip(
                broad_confidence, broad_boundaries
            )
        ])
        fine_masks = np.stack([
            self._grow_highlight_components(score, board_mask, boundary)
            for score, boundary in zip(fine_confidence, fine_boundaries)
        ])
        candidates = broad_masks | fine_masks
        combined_confidence = np.maximum(
            broad_confidence, fine_confidence
        )
        return self._resolve_cross_light_ownership(
            candidates, combined_confidence, tolerance_px
        ) & board_mask[None, ...]

    # 7. 反光掩膜生成入口：整图推理和训练裁剪图块处理
    def _compute_global_highlight(self, lq_coax, lq_bar, lq_ring):
        """ 作用：在大图全分辨率下计算允许0到5像素错位的三光源独占反光掩膜 """
        images = (lq_coax, lq_bar, lq_ring)
        board_mask = self._compute_highlight_board_mask(images)
        if not np.any(board_mask):
            return np.zeros((*lq_coax.shape[:2], 3), dtype=np.float32)

        cache_small_map = board_mask.size <= 512 * 512
        blur_cache = self._build_masked_gaussian_cache(
            board_mask, (0.75, 1.35, 2.40, 3.0, 4.0, 9.0, 18.0)
        ) if cache_small_map else None
        features = [
            self._extract_highlight_features(image, board_mask, blur_cache)
            for image in images
        ]
        reference_cache = {} if cache_small_map else None
        broad_confidence, fine_confidence = [], []
        for current_index in range(3):
            reference_indices = [
                index for index in range(3) if index != current_index
            ]
            broad, fine = self._build_single_light_degradation(
                features[current_index],
                features[reference_indices[0]],
                features[reference_indices[1]],
                board_mask,
                tolerance_px=5,
                reference_cache=reference_cache
            )
            broad_confidence.append(broad)
            fine_confidence.append(fine)

        exclusive = self._arbitrate_exclusive_highlight(
            np.stack(broad_confidence),
            np.stack(fine_confidence),
            board_mask,
            sensitivity=0.70,
            tolerance_px=5
        )
        return np.moveaxis(exclusive.astype(np.float32), 0, -1)

    def _compute_cropped_training_highlight(self, cropped_lq):
        """ 作用：训练阶段只在实际使用的裁剪图块上计算反光，避免整图计算后被裁掉 """
        cropped_lq = np.ascontiguousarray(cropped_lq[:, :, :9])
        return self._compute_global_highlight(
            cropped_lq[:, :, :3], cropped_lq[:, :, 3:6],
            cropped_lq[:, :, 6:9]
        )

    def __getitem__(self, index):
        sample = self.paths[index % len(self.paths)]
        img_id, ext = sample['id'], sample['ext']

        gt_coax, gt_bar, gt_ring = self._read_light_triplet(
            (self.gt_folder,) * 3, img_id, ext
        )
        img_gt = np.concatenate([gt_coax, gt_bar, gt_ring], axis=2)

        lq_coax, lq_bar, lq_ring = self._read_light_triplet(
            (self.lq_coax_folder, self.lq_bar_folder, self.lq_ring_folder),
            img_id, ext
        )
        img_lq_imgs = np.concatenate([lq_coax, lq_bar, lq_ring], axis=2)

        img_drop = self._compute_global_drop(lq_coax, lq_bar, lq_ring)

        p_coax, p_bar, p_ring = self._load_prior_triplet(img_id)
        img_prior = np.concatenate([p_coax, p_bar, p_ring], axis=2)

        global_board_mask, _ = self._compute_perfect_mask(lq_coax, lq_bar, lq_ring)
        img_prior = img_prior * global_board_mask[:, :, np.newaxis]

        img_lq_all = np.concatenate([img_lq_imgs, img_prior, img_drop], axis=2)

        if self.opt['phase'] == 'train':
            gt_size, scale = self.opt['gt_size'], self.opt['scale']
            img_gt, img_lq_all = padding(img_gt, img_lq_all, gt_size)
            img_gt, img_lq_all = paired_random_crop(img_gt, img_lq_all, gt_size, scale, img_id)
            if self.opt.get('geometric_augs', True):
                img_gt, img_lq_all = random_augmentation(img_gt, img_lq_all)
            img_highlight = self._compute_cropped_training_highlight(img_lq_all)
        else:
            img_highlight = self._compute_global_highlight(
                lq_coax, lq_bar, lq_ring
            )
        img_lq_all = np.concatenate([img_lq_all, img_highlight], axis=2)

        img_gt, img_lq_all = img2tensor([img_gt, img_lq_all], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq_all,
            'gt': img_gt,
            'lq_path': os.path.join(self.lq_coax_folder, f"{img_id}_coax{ext}"),
            'gt_path': os.path.join(self.gt_folder, f"{img_id}_coax{ext}")
        }

    def __len__(self):
        return len(self.paths)


# 8. 通用配对数据集：BasicSR 标准 LQ/GT 读取、裁剪、增强和归一化
class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration."""

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception(f"gt path {gt_path} not working")

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception(f"lq path {lq_path} not working")

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


# 9. 高斯去噪数据集：读取 GT、生成训练噪声或固定测试噪声
class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration."""

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paired_paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [os.path.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(os.listdir(self.gt_folder)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        gt_path = self.paths[index] if isinstance(self.paths[index], str) else self.paths[index].get('gt_path', '')
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except Exception:
                raise Exception(f"gt path {gt_path} not working")
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except Exception:
                raise Exception(f"gt path {gt_path} not working")

        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value]) / 255.0
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)
        else:
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test / 255.0, img_lq.shape)
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': gt_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
