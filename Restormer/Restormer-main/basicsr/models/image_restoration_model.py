"""三光源阴影与反光联合修复模型；按职责组织代码，不改变算法逻辑。"""

import importlib
import os
import random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger

loss_module = importlib.import_module('basicsr.models.losses')

# 基础组件：反光伪目标异常、全分辨率平滑与 Mixup 数据增强
class _HighlightPseudoTargetError(RuntimeError):
    """反光跨光源RGB伪目标因候选与外围种子均缺失或出现非有限数值而无法生成。"""

def _conv_macro_smooth51(x, mode='reflect'):
    """
    作用：利用 1D 拆分平滑实现 100% 全分辨率 51x51 平滑。
    无任何下采样，直接调用 PyTorch 原生 C++ 算子，彻底避免动态显存申请与 GPU 寻址瓶颈。
    """
    x_h = F.pad(x, (25, 25, 0, 0), mode=mode)
    x_h = F.avg_pool2d(x_h, kernel_size=(1, 51), stride=1)
    x_v = F.pad(x_h, (0, 0, 25, 25), mode=mode)
    return F.avg_pool2d(x_v, kernel_size=(51, 1), stride=1)

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        """ 作用：初始化Mixup增强类，配置Beta分布与运算设备 """
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        """ 作用：执行Mixup算法，对当前Batch数据进行随机打乱与线性融合 """
        file_lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = file_lam * target + (1 - file_lam) * target[r_index, :]
        input_ = file_lam * input_ + (1 - file_lam) * input_[r_index, :]
        return target, input_

    def __call__(self, target, input_):
        """ 作用：使实例可调用，根据配置项 decide 应用Mixup增强或恒等映射 """
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """三光源阴影与反光联合修复模型。"""

    # 1. 模型生命周期：创建网络、加载权重并配置训练组件
    def __init__(self, opt):
        """ 作用：初始化模型类，构建网络、加载权重并常驻拉普拉斯核显存Buffer以极致提速 """
        super(ImageCleanModel, self).__init__(opt)

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(
                self.net_g, load_path, self.opt['path'].get('strict_load_g', True),
                param_key=self.opt['path'].get('param_key', 'params')
            )

        if self.is_train:
            self.init_training_settings()

        self.loss_history = []
        self.shadow_loss_history = []
        self.highlight_loss_history = []
        self.pix_loss_history = []
        self.iter_history = []

        self.raw_prior = None
        self.drop = None
        self.highlight_mask = None

        self.total_loss_accum = 0.0
        self.shadow_loss_accum = 0.0
        self.highlight_loss_accum = 0.0
        self.pix_loss_accum = 0.0
        self.period_step_count = 0

        self.epoch_count = 0

    def init_training_settings(self):
        """ 作用：初始化训练 environment 专属参数，配置基础像素损失(L1 Loss)及EMA影子网络 """
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """ 作用：筛选过滤网络中需训练的有效参数，创建并初始化指定的优化器(如AdamW) """
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        self.optimizers.append(self.optimizer_g)

    # 2. 数据输入：拆分三路 RGB、先验、阴影变暗量与反光掩膜
    def _split_input(self, data):
        """拆分输入张量并屏蔽反光区域的先验。"""
        lq_all = data['lq'].to(self.device)
        img, prior, drop, highlight_mask = (
            lq_all[:, :9], lq_all[:, 9:18], lq_all[:, 18:21], lq_all[:, 21:24]
        )
        prior = prior * (1.0 - torch.repeat_interleave(highlight_mask, 3, dim=1))
        return img, prior, drop, highlight_mask

    def feed_train_data(self, data):
        """ 作用：切片18通道训练输入，对9通道先验赋(0-1)随机权重并加入领域扰动后组装喂给网络 """
        img, prior, self.drop, self.highlight_mask = self._split_input(data)

        shift_h = random.randint(-8, 8)
        shift_w = random.randint(-8, 8)
        prior = torch.roll(prior, shifts=(shift_h, shift_w), dims=(2, 3))

        self.raw_prior = prior.clone()
        self.prior_weight = random.uniform(0.0, 1.0)

        if random.random() < 0.3:
            self.prior_weight = 0.0

        scale_factor = random.uniform(0.85, 1.15)
        prior = prior * scale_factor

        noise = torch.randn_like(prior) * 0.02
        prior = prior + noise

        scaled_prior = prior * self.prior_weight
        self.lq = torch.cat([img, scaled_prior], dim=1)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        """ 作用：接收验证/测试 data，支持通过外部传入固定权重参数(如0.1、0.8)人工调节先验控图强度 """
        img, prior, self.drop, self.highlight_mask = self._split_input(data)

        self.prior_weight = data.get('prior_weight', self.opt.get('prior_weight', 1.0))
        self.raw_prior = prior.clone()
        scaled_prior = prior * self.prior_weight

        self.lq = torch.cat([img, scaled_prior], dim=1)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    # 3. 训练监控：记录并绘制总损失、阴影损失、反光损失和像素损失
    def plot_loss_curve(self, current_iter):
        """ 作用：在指定迭代步后台无GUI渲染并保存训练Loss曲线折线图至picture文件夹 """
        save_dir = '/home/cgz/lzc/Restormer/Restormer-main/picture'
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        curves = [(self.loss_history, 'skyblue', 'Total Loss', 'Total Training Loss'),
                  (self.shadow_loss_history, 'orange', 'Shadow Loss', 'Overall Shadow Loss'),
                  (self.highlight_loss_history, 'green', 'Highlight Loss', 'Decoupled Highlight Loss'),
                  (self.pix_loss_history, 'purple', 'Global Pixel Loss', 'Global Pixel Loss')]
        for ax, (values, color, label, title) in zip(axes, curves):
            ax.plot(self.iter_history, values, color=color, linestyle='-', label=label)
            ax.set_title(title)
            ax.grid(True)

        for ax in axes:
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Loss Value')
            ax.legend()

        plt.suptitle(f'Synchronized Training Loss Curves (Iter {current_iter})', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'loss_curve_{current_iter}.png'), dpi=150)
        plt.close()

    # 4. 阴影处理：提取阴影区域并构造亮度、纹理、结构和颜色监督
    def _apply_prior_gating(self, prior_activated, drop):
        """ 作用：执行图像驱动先场定向门控，利用物理变变暗信号drop剔除高光反光区的先场干扰 """
        shadow_gate = (drop > 0.0).float()
        return prior_activated * shadow_gate

    def _extract_image_driven_shadow(self, prior):
        """ 作用：结合大窗口均值滤波背景落差与.npy空間区域先场，自适应提取全图深浅阴影 """
        masks_s = []
        priors_activated = []

        for i in range(3):
            drop = self.drop[:, i:i + 1, :, :]
            p_prior = prior[:, i * 3:(i + 1) * 3, :, :].mean(dim=1, keepdim=True)
            prior_activated = torch.tanh(torch.abs(p_prior) * 1)
            prior_activated = self._apply_prior_gating(prior_activated, drop)

            base_mask = torch.tanh(drop * 18.0) + prior_activated * self.prior_weight
            base_mask = F.pad(base_mask, (1, 1, 1, 1), mode='reflect')
            base_mask = F.avg_pool2d(base_mask, kernel_size=3, stride=1, padding=0)

            masks_s.append(torch.clamp(base_mask, 0.0, 1.0))
            priors_activated.append(prior_activated)

        return masks_s, priors_activated

    def _compute_relative_shadow_severity(self):
        """ 作用：提取物理变暗信号并经局部平均池化与绝对指数映射，自适应计算三光源独立绝对阴影严重度权重图 """
        smoothed_drop = F.avg_pool2d(self.drop, kernel_size=(1, 31), stride=1, padding=(0, 15))
        smoothed_drop = F.avg_pool2d(smoothed_drop, kernel_size=(31, 1), stride=1, padding=(15, 0))
        return torch.exp(smoothed_drop * 5.0)

    def _compute_soft_fallback_mask(self, neighbor_weight_sum):
        """ 作用：利用 Sigmoid 软平滑结合 3x3 空间滤波计算连续回退掩码，消除 0/1 阶跃坑洞与局域震荡 """
        soft_mask = torch.sigmoid((5e-4 - neighbor_weight_sum) * 10000.0)
        return F.avg_pool2d(soft_mask, kernel_size=3, stride=1, padding=1)

    def _project_cross_light_std_increment(self, unified_std, self_std, unified_mean):
        """ 作用：保留本光源标准差基线，使合法跨光源增量直通，并仅在容量边界附近进行自适应平滑限制 """
        capacity_mean = torch.clamp(unified_mean, 0.0, 1.0)
        sigma_max = torch.sqrt(torch.clamp(capacity_mean * (1.0 - capacity_mean), min=0.0))
        borrowed_increment = F.relu(unified_std - self_std)
        remaining_capacity = F.relu(sigma_max - self_std)
        transition_width = remaining_capacity / 16.0
        transition_start = remaining_capacity - transition_width
        transition_end = remaining_capacity + transition_width
        safe_width = torch.where(transition_width > 0.0, transition_width, torch.ones_like(transition_width))
        smooth_increment = borrowed_increment - (borrowed_increment - transition_start) ** 2 / (4.0 * safe_width)
        projected_increment = torch.where(
            borrowed_increment <= transition_start,
            borrowed_increment,
            torch.where(borrowed_increment >= transition_end, remaining_capacity, smooth_increment)
        )
        return self_std + torch.minimum(F.relu(projected_increment), remaining_capacity)

    def _cal_shadow_structure_polarity_loss(self, pred_sub, gt_sub, mask_s):
        """ 作用：在阴影内以本光源干净GT为目标，对高对比结构施加带符号一阶梯度极性约束 """
        pred_grad_x = pred_sub[:, :, :, 1:] - pred_sub[:, :, :, :-1]
        pred_grad_y = pred_sub[:, :, 1:, :] - pred_sub[:, :, :-1, :]
        gt_grad_x = gt_sub[:, :, :, 1:] - gt_sub[:, :, :, :-1]
        gt_grad_y = gt_sub[:, :, 1:, :] - gt_sub[:, :, :-1, :]

        strength_x = gt_grad_x.detach().abs().mean(dim=1, keepdim=True)
        strength_y = gt_grad_y.detach().abs().mean(dim=1, keepdim=True)
        weight_x = (strength_x / (strength_x + 0.05)).square()
        weight_y = (strength_y / (strength_y + 0.05)).square()
        mask_x = torch.minimum(mask_s[:, :, :, 1:], mask_s[:, :, :, :-1]) * weight_x
        mask_y = torch.minimum(mask_s[:, :, 1:, :], mask_s[:, :, :-1, :]) * weight_y

        loss_x = torch.sum(torch.abs(pred_grad_x - gt_grad_x) * mask_x) / \
                 (torch.sum(mask_x) * pred_sub.size(1) + 1e-5)
        loss_y = torch.sum(torch.abs(pred_grad_y - gt_grad_y) * mask_y) / \
                 (torch.sum(mask_y) * pred_sub.size(1) + 1e-5)
        return 0.5 * (loss_x + loss_y)

    def _prepare_shadow_gt_statistics(self, clean_gt, highlight_block_masks):
        """ 作用：在优化前预先计算三光源干净GT的11x11低频、局部标准差与有效支撑掩膜 """
        highlight_blocks = [(mask > 0.5).to(clean_gt.dtype) for mask in highlight_block_masks]
        gt_low_all, gt_std_all, source_support_all = [], [], []
        for k in range(3):
            gt_sub_k = clean_gt[:, k * 3:(k + 1) * 3, :, :]
            finite = torch.isfinite(gt_sub_k).all(1, keepdim=True)
            valid = (1.0 - highlight_blocks[k]) * finite.to(gt_sub_k.dtype)
            safe_gt = torch.where(torch.isfinite(gt_sub_k), gt_sub_k, torch.zeros_like(gt_sub_k))
            valid_pad = F.pad(valid, (5, 5, 5, 5), mode='reflect')
            weight = F.avg_pool2d(valid_pad, kernel_size=11, stride=1)
            moments = F.avg_pool2d(
                F.pad(torch.cat([safe_gt, safe_gt.square()], dim=1) * valid,
                      (5, 5, 5, 5), mode='reflect'), kernel_size=11, stride=1
            ) / weight.clamp_min(1e-6)
            low, square_low = moments.chunk(2, dim=1)
            gt_low_all.append(low)
            gt_std_all.append(torch.sqrt(F.relu(square_low - low.square()) + 1e-5))
            source_support_all.append((weight > 1e-6).to(clean_gt.dtype))
        return gt_low_all, gt_std_all, source_support_all, highlight_blocks

    def _cal_single_target_shadow_loss(self, p_sub, clean_gt, masks_s, priors_activated,
                                       i, gt_stats, severity_weights):
        """ 作用：计算单个目标光源在阴影区域的统计量对齐、极性、SSIM与色彩一致性损失 """
        gt_low_all, gt_std_all, source_support_all, highlight_blocks = gt_stats
        mask_s = masks_s[i]
        health_i = (1.0 - mask_s) * (1.0 - highlight_blocks[i]) * source_support_all[i]
        gt_sub = clean_gt[:, i * 3:(i + 1) * 3, :, :]
        severity_weight = severity_weights[:, i:i + 1, :, :]
        attn_multiplier = 1.0 + self.prior_weight * priors_activated[i]

        gt_low_i = gt_low_all[i]
        gt_std_i = gt_std_all[i]

        neighbor_std_sum = 0.0
        neighbor_mean_sum = 0.0
        neighbor_weight_sum = 0.0

        for j in range(3):
            if j != i:
                source_available_j = (1.0 - highlight_blocks[j]) * source_support_all[j]
                health_j = (1.0 - masks_s[j]) * source_available_j
                relative_health_j = F.relu(health_j - health_i)

                gt_low_j = gt_low_all[j]
                gt_std_j = gt_std_all[j]

                joint_healthy_mask = health_i * health_j
                joint_denom = torch.sum(joint_healthy_mask, dim=[2, 3], keepdim=True) + 1e-5

                mean_std_i = torch.sum(gt_std_i * joint_healthy_mask, dim=[2, 3], keepdim=True) / joint_denom
                mean_std_j = torch.sum(gt_std_j * joint_healthy_mask, dim=[2, 3], keepdim=True) / joint_denom
                ratio_contrast = (mean_std_i + 1e-5) / (mean_std_j + 1e-5)

                mean_low_i = torch.sum(gt_low_i * joint_healthy_mask, dim=[2, 3], keepdim=True) / joint_denom
                mean_low_j = torch.sum(gt_low_j * joint_healthy_mask, dim=[2, 3], keepdim=True) / joint_denom
                ratio_tone = (mean_low_i + 1e-5) / (mean_low_j + 1e-5)

                ratio_contrast = torch.clamp(ratio_contrast, 0.2, 2.5)
                ratio_tone = torch.clamp(ratio_tone, 0.2, 2.5)

                h_w_total = joint_healthy_mask.shape[2] * joint_healthy_mask.shape[3]
                area_ratio = torch.sum(joint_healthy_mask, dim=[2, 3], keepdim=True) / h_w_total
                gate_weight = torch.clamp(area_ratio / 0.05, 0.0, 1.0)

                ratio_tone = ratio_tone * gate_weight + 1.0 * (1.0 - gate_weight)
                ratio_contrast = ratio_contrast * gate_weight + 1.0 * (1.0 - gate_weight)

                effective_weight_j = relative_health_j * (health_j * health_j)
                neighbor_std_sum += (gt_std_j * ratio_contrast) * effective_weight_j
                neighbor_mean_sum += (gt_low_j * ratio_tone) * effective_weight_j
                neighbor_weight_sum += effective_weight_j

        unified_gt_std = neighbor_std_sum / (neighbor_weight_sum + 1e-5)
        unified_gt_mean = neighbor_mean_sum / (neighbor_weight_sum + 1e-5)

        unified_gt_mean = torch.max(unified_gt_mean, gt_low_i)
        unified_gt_std = torch.max(unified_gt_std, gt_std_i)

        fallback_mask = self._compute_soft_fallback_mask(neighbor_weight_sum)
        unified_gt_std = unified_gt_std * (1.0 - fallback_mask) + gt_std_i * fallback_mask
        unified_gt_mean = unified_gt_mean * (1.0 - fallback_mask) + gt_low_i * fallback_mask

        unified_gt_std = self._project_cross_light_std_increment(
            unified_gt_std, gt_std_i, unified_gt_mean
        )

        p_pad11 = F.pad(p_sub, (5, 5, 5, 5), mode='reflect')
        p_mean11 = F.avg_pool2d(p_pad11, kernel_size=11, stride=1)
        p_sq11 = F.avg_pool2d(p_pad11 ** 2, kernel_size=11, stride=1)
        p_std11 = torch.sqrt(F.relu(p_sq11 - p_mean11 ** 2) + 1e-5)

        p_mean51 = _conv_macro_smooth51(p_mean11, mode='reflect')
        p_std51 = _conv_macro_smooth51(p_std11, mode='reflect')

        gt_mean51 = _conv_macro_smooth51(unified_gt_mean, mode='reflect')
        gt_std51 = _conv_macro_smooth51(unified_gt_std, mode='reflect')

        mask_s_sum = torch.sum(mask_s, dim=[2, 3], keepdim=True) + 1e-5
        p_macro = torch.sum(p_sub * mask_s, dim=[2, 3], keepdim=True) / mask_s_sum
        gt_macro = torch.sum(unified_gt_mean * mask_s, dim=[2, 3], keepdim=True) / mask_s_sum

        brightness_error = (0.5 * torch.abs(p_mean11 - unified_gt_mean) +
                            0.3 * torch.abs(p_mean51 - gt_mean51) +
                            0.2 * torch.abs(p_macro - gt_macro)) * mask_s * attn_multiplier * severity_weight

        texture_error = (0.6 * torch.abs(p_std11 - unified_gt_std) +
                         0.4 * torch.abs(p_std51 - gt_std51)) * mask_s * attn_multiplier * severity_weight

        global_mask_sum = torch.sum(mask_s) + 1e-5
        loss_b = torch.sum(brightness_error) / (global_mask_sum * 3.0)
        loss_t = torch.sum(texture_error) / (global_mask_sum * 3.0)
        loss_polarity = self._cal_shadow_structure_polarity_loss(p_sub, gt_sub, mask_s)

        c1, c2 = 0.0001, 0.0009
        mu_p = F.avg_pool2d(p_sub, kernel_size=11, stride=1, padding=5)
        mu_g = F.avg_pool2d(gt_sub, kernel_size=11, stride=1, padding=5)
        sigma_p2 = F.relu(F.avg_pool2d(p_sub ** 2, kernel_size=11, stride=1, padding=5) - mu_p ** 2)
        sigma_g2 = F.relu(F.avg_pool2d(gt_sub ** 2, kernel_size=11, stride=1, padding=5) - mu_g ** 2)
        sigma_pg = F.avg_pool2d(p_sub * gt_sub, kernel_size=11, stride=1, padding=5) - mu_p * mu_g

        ssim_idx = ((2 * mu_p * mu_g + c1) * (2 * sigma_pg + c2)) / ((mu_p ** 2 + mu_g ** 2 + c1) * (sigma_p2 + sigma_g2 + c2))
        loss_ssim = torch.clamp(1.0 - ssim_idx, min=0.0, max=2.0)
        loss_ssim_val = torch.sum(loss_ssim * mask_s) / (global_mask_sum * 3.0)

        dot_prod = torch.sum(p_sub * gt_sub, dim=1, keepdim=True)
        p_norm = torch.sqrt(torch.sum(p_sub ** 2, dim=1, keepdim=True) + 1e-8)
        gt_norm = torch.sqrt(torch.sum(gt_sub ** 2, dim=1, keepdim=True) + 1e-8)

        loss_color = 1.0 - ((dot_prod + 1e-8) / (p_norm * gt_norm))
        loss_color = torch.clamp(loss_color, min=0.0)
        loss_color_val = torch.sum(loss_color * mask_s) / global_mask_sum

        shadow_total = (loss_b * 2.5 + loss_t * 4.5 + loss_polarity + loss_color_val * 0.5 + loss_ssim_val * 0.2)

        shadow_sub_dict = {
            'l_shd_bright': loss_b,
            'l_shd_tex': loss_t,
            'l_shd_polarity': loss_polarity,
            'l_shd_color': loss_color_val,
            'l_shd_ssim': loss_ssim_val
        }
        return shadow_total, shadow_sub_dict

    def _cal_shadow_consistency_loss(self, pred, clean_gt, masks_s, priors_activated,
                                     highlight_block_masks):
        """ 作用：基于区域掩膜计算跨光源暗区损失，在统计前硬排除反光来源并保留原相对健康度与软回退 """
        gt_stats = self._prepare_shadow_gt_statistics(clean_gt, highlight_block_masks)
        severity_weights = self._compute_relative_shadow_severity()
        shadow_total = 0.0
        shadow_sub_dict = {'l_shd_bright': 0.0, 'l_shd_tex': 0.0, 'l_shd_polarity': 0.0, 'l_shd_color': 0.0, 'l_shd_ssim': 0.0}
        for i in range(3):
            p_sub = pred[:, i * 3:(i + 1) * 3, :, :]
            s_tot, s_dict = self._cal_single_target_shadow_loss(
                p_sub, clean_gt, masks_s, priors_activated, i, gt_stats, severity_weights
            )
            shadow_total = shadow_total + s_tot
            for k in shadow_sub_dict:
                shadow_sub_dict[k] = shadow_sub_dict[k] + s_dict[k]
        return shadow_total, shadow_sub_dict

    # 5. 反光处理（一）：独占反光仲裁、阻断掩膜与三路专用前向
    def _arbitrate_highlight_masks(self, highlight_mask=None):
        """ 作用：使用原始或填充对齐后的0/1反光掩膜，仲裁并返回三路单光源独占反光区域 """
        highlight_mask = self.highlight_mask if highlight_mask is None else highlight_mask
        binary_bool = [highlight_mask[:, i:i + 1] > 0.5 for i in range(3)]
        highlight_count = torch.stack(binary_bool, dim=0).sum(dim=0)
        mask_dtype = highlight_mask.dtype
        binary_masks = [mask.to(mask_dtype) for mask in binary_bool]
        exclusive_masks = [
            (mask & (highlight_count == 1)).to(mask_dtype) for mask in binary_bool
        ]
        return binary_masks, exclusive_masks

    def _synthesize_highlight_block_masks(self, binary_highlight_masks, planned_inpaint_masks):
        """ 作用：将原始反光核心与自适应计划修复区合并，合成三光源统一反光阻断掩膜 """
        return [torch.maximum(binary_highlight_masks[i], planned_inpaint_masks[i]) for i in range(3)]

    def _build_target_light_input(self, model_input, highlight_block_masks, target_index):
        """ 作用：为指定目标光源保留自身RGB，并在其他光源的反光阻断区用剩余非反光光源RGB替换 """
        light_rgbs = [model_input[:, i * 3:(i + 1) * 3] for i in range(3)]
        routed_rgbs = []
        for source_index in range(3):
            if source_index == target_index:
                routed_rgbs.append(light_rgbs[source_index])
                continue
            replacement_index = 3 - target_index - source_index
            replace_mask = (highlight_block_masks[source_index] > 0.5) & \
                (highlight_block_masks[replacement_index] < 0.5)
            routed_rgbs.append(torch.where(
                replace_mask, light_rgbs[replacement_index], light_rgbs[source_index]
            ))
        return torch.cat(routed_rgbs + [model_input[:, 9:]], dim=1)

    def _forward_target_light_outputs(self, network, model_input, highlight_block_masks):
        """ 作用：基于统一反光阻断区分别执行A/B/C目标专用前向，每次仅保留对应3通道并拼接输出 """
        if not torch.any(torch.cat(highlight_block_masks, dim=1) > 0.5).item():
            return network(model_input)
        target_outputs = [
            network(self._build_target_light_input(model_input, highlight_block_masks, i))
            for i in range(3)
        ]
        output_is_list = isinstance(target_outputs[0], list)
        output_lists = [output if isinstance(output, list) else [output] for output in target_outputs]
        merged_outputs = [torch.cat([
            output_lists[i][level][:, i * 3:(i + 1) * 3] for i in range(3)
        ], dim=1) for level in range(len(output_lists[0]))]
        return merged_outputs if output_is_list else merged_outputs[0]

    # 6. 反光处理（二）：连通块聚合、修复区扩展与参考区域规划

    def _find_nearby_highlight_component_edges(self, labels, group_distance):
        """ 作用：在指定欧氏距离内局部扫描不同反光连通块，返回块编号及最近连接端点 """
        group_distance = max(0.0, float(group_distance))
        radius = int(np.ceil(group_distance))
        foreground_y, foreground_x = np.nonzero(labels)
        if group_distance <= 0.0 or foreground_y.size == 0:
            return []

        image_height, image_width = labels.shape
        y0 = max(0, int(foreground_y.min()) - radius)
        y1 = min(image_height, int(foreground_y.max()) + radius + 1)
        x0 = max(0, int(foreground_x.min()) - radius)
        x1 = min(image_width, int(foreground_x.max()) + radius + 1)
        local_labels = labels[y0:y1, x0:x1]
        local_height, local_width = local_labels.shape
        component_count = int(labels.max())
        offsets = [
            (dy * dy + dx * dx, dy, dx)
            for dy in range(radius + 1)
            for dx in range(-radius, radius + 1)
            if (dy > 0 or dx > 0) and dy * dy + dx * dx <= group_distance ** 2
        ]
        offsets.sort()
        nearest_edges = {}

        for _, dy, dx in offsets:
            source_y0, source_y1 = max(0, -dy), min(local_height, local_height - dy)
            source_x0, source_x1 = max(0, -dx), min(local_width, local_width - dx)
            if source_y0 >= source_y1 or source_x0 >= source_x1:
                continue
            source = local_labels[source_y0:source_y1, source_x0:source_x1]
            target = local_labels[
                source_y0 + dy:source_y1 + dy,
                source_x0 + dx:source_x1 + dx
            ]
            valid = (source > 0) & (target > 0) & (source != target)
            match_y, match_x = np.nonzero(valid)
            if match_y.size == 0:
                continue

            source_ids = source[match_y, match_x] - 1
            target_ids = target[match_y, match_x] - 1
            first_ids = np.minimum(source_ids, target_ids)
            second_ids = np.maximum(source_ids, target_ids)
            pair_keys = first_ids.astype(np.int64) * component_count + second_ids
            _, first_matches = np.unique(pair_keys, return_index=True)
            for match_index in first_matches:
                first = int(first_ids[match_index])
                second = int(second_ids[match_index])
                pair = (first, second)
                if pair in nearest_edges:
                    continue
                source_point = np.asarray(
                    [source_y0 + match_y[match_index] + y0,
                     source_x0 + match_x[match_index] + x0], dtype=np.int32
                )
                target_point = source_point + np.asarray([dy, dx], dtype=np.int32)
                if int(source_ids[match_index]) == first:
                    start, end = source_point, target_point
                else:
                    start, end = target_point, source_point
                nearest_edges[pair] = (first, second, start, end)

        return list(nearest_edges.values())

    def _build_grouped_highlight_envelope(self, mask, group_distance):
        """ 作用：按指定最短距离连接并填孔反光碎片，返回紧聚合包络及其原始反光核心像素数 """
        grouped_envelopes = []
        mask_cpu = mask.detach().to(device='cpu', dtype=torch.uint8).numpy()
        for batch_index in range(mask.size(0)):
            labels, count = ndimage.label(
                mask_cpu[batch_index, 0], structure=np.ones((3, 3), dtype=np.uint8)
            )
            if count == 0:
                continue

            parents = list(range(count))

            def _find(index):
                """ 作用：查找并压缩当前原始反光连通块所属的聚合根节点 """
                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index

            edges = self._find_nearby_highlight_component_edges(labels, group_distance)
            for first, second, _, _ in edges:
                root_first, root_second = _find(first), _find(second)
                if root_first != root_second:
                    parents[root_second] = root_first

            groups = {}
            for index in range(count):
                groups.setdefault(_find(index), []).append(index)

            envelope_union = np.zeros_like(labels, dtype=bool)
            for members in groups.values():
                core_highlight_mask = np.isin(
                    labels, np.asarray(members, dtype=np.int32) + 1
                )
                tight_envelope = core_highlight_mask.copy()
                member_set = set(members)
                for first, second, start, end in edges:
                    if first not in member_set or second not in member_set:
                        continue
                    steps = int(max(abs(int(end[0]) - int(start[0])),
                                    abs(int(end[1]) - int(start[1]))))
                    ys = np.rint(np.linspace(start[0], end[0], steps + 1)).astype(np.int32)
                    xs = np.rint(np.linspace(start[1], end[1], steps + 1)).astype(np.int32)
                    tight_envelope[ys, xs] = True
                    for point in range(1, len(ys)):
                        if ys[point] != ys[point - 1] and xs[point] != xs[point - 1]:
                            tight_envelope[ys[point - 1], xs[point]] = True
                envelope_union |= ndimage.binary_fill_holes(tight_envelope)

            final_labels, final_count = ndimage.label(
                envelope_union, structure=np.ones((3, 3), dtype=np.uint8)
            )
            for final_index in range(1, final_count + 1):
                tight_envelope = ndimage.binary_fill_holes(final_labels == final_index)
                ys, xs = np.where(tight_envelope)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                local_envelope = torch.from_numpy(
                    tight_envelope[y0:y1, x0:x1]
                ).to(device=mask.device, dtype=mask.dtype).unsqueeze(0).unsqueeze(0)
                core_pixel_count = int(np.count_nonzero(
                    (mask_cpu[batch_index, 0] > 0) & tight_envelope
                ))
                grouped_envelopes.append(
                    (batch_index, (int(y0), int(y1), int(x0), int(x1)),
                     local_envelope, core_pixel_count)
                )
        return grouped_envelopes

    def _expand_highlight_roi(self, bounds, margin, image_height, image_width):
        """ 作用：将反光紧致边界框按指定上下文距离扩展，并裁剪到图像有效范围 """
        y0, y1, x0, x1 = bounds
        return (max(0, y0 - margin), min(image_height, y1 + margin),
                max(0, x0 - margin), min(image_width, x1 + margin))

    def _dilate_four_neighborhood(self, mask, radius):
        """ 作用：严格按上下左右四邻域逐层扩展掩码，不引入方形最大池化的对角捷径 """
        expanded = mask > 0.5
        for _ in range(radius):
            next_mask = expanded.clone()
            next_mask[:, :, 1:, :] |= expanded[:, :, :-1, :]
            next_mask[:, :, :-1, :] |= expanded[:, :, 1:, :]
            next_mask[:, :, :, 1:] |= expanded[:, :, :, :-1]
            next_mask[:, :, :, :-1] |= expanded[:, :, :, 1:]
            expanded = next_mask
        return expanded.to(mask.dtype)

    def _build_highlight_repair_and_context_masks(self, grouped_envelopes, template_mask,
                                                   raw_highlight_union,
                                                   multi_highlight_mask):
        """ 作用：按原始核心面积构造0或10像素自适应修复区，合并重叠任务并从紧包络外10至20像素取颜色参考 """
        batch_size, _, image_height, image_width = template_mask.shape
        full_adaptive_mask = template_mask.new_zeros(
            (batch_size, 1, image_height, image_width)
        )
        grouped_by_batch = [[] for _ in range(batch_size)]
        for batch_index, bounds, tight_envelope, core_pixel_count in grouped_envelopes:
            y0, y1, x0, x1 = bounds
            roi_y0, roi_y1, roi_x0, roi_x1 = self._expand_highlight_roi(
                bounds, 20, image_height, image_width
            )
            envelope_roi = template_mask.new_zeros((1, 1, roi_y1 - roi_y0, roi_x1 - roi_x0))
            envelope_roi[:, :, y0 - roi_y0:y1 - roi_y0, x0 - roi_x0:x1 - roi_x0] = \
                tight_envelope
            repair_roi = envelope_roi if core_pixel_count < 20 else \
                self._dilate_four_neighborhood(envelope_roi, 10)
            full_adaptive_mask[
                batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
            ] = torch.maximum(
                full_adaptive_mask[
                    batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
                ], repair_roi
            )
            grouped_by_batch[batch_index].append(
                (bounds, tight_envelope,
                 tight_envelope.detach().to(device='cpu', dtype=torch.uint8).numpy()[0, 0] > 0)
            )

        full_inpaint_mask = full_adaptive_mask * (1.0 - multi_highlight_mask)
        repair_tasks = []
        repair_cpu = full_adaptive_mask.detach().to(
            device='cpu', dtype=torch.uint8
        ).numpy()
        four_connected = np.asarray(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        )
        for batch_index in range(batch_size):
            labels, count = ndimage.label(repair_cpu[batch_index, 0], structure=four_connected)
            for component_index in range(1, count + 1):
                component = labels == component_index
                members = [
                    (bounds, tight_envelope) for bounds, tight_envelope, tight_cpu
                    in grouped_by_batch[batch_index]
                    if np.any(component[bounds[0]:bounds[1], bounds[2]:bounds[3]] & tight_cpu)
                ]
                if not members:
                    continue
                tight_bounds = (
                    min(bounds[0] for bounds, _ in members),
                    max(bounds[1] for bounds, _ in members),
                    min(bounds[2] for bounds, _ in members),
                    max(bounds[3] for bounds, _ in members)
                )
                roi_bounds = self._expand_highlight_roi(
                    tight_bounds, 20, image_height, image_width
                )
                roi_y0, roi_y1, roi_x0, roi_x1 = roi_bounds
                repair_mask = torch.from_numpy(
                    component[roi_y0:roi_y1, roi_x0:roi_x1]
                ).to(device=template_mask.device, dtype=template_mask.dtype).unsqueeze(0).unsqueeze(0) * \
                    (1.0 - multi_highlight_mask[
                        batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
                    ])
                if not torch.any(repair_mask > 0.5).item():
                    continue
                tight_component = torch.zeros_like(repair_mask)
                for (y0, y1, x0, x1), tight_envelope in members:
                    tight_component[:, :, y0 - roi_y0:y1 - roi_y0,
                                    x0 - roi_x0:x1 - roi_x0] = torch.maximum(
                        tight_component[:, :, y0 - roi_y0:y1 - roi_y0,
                                        x0 - roi_x0:x1 - roi_x0], tight_envelope
                    )
                highlight_core_mask = template_mask[
                    batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
                ] * repair_mask
                statistics_inner = self._dilate_four_neighborhood(tight_component, 10)
                statistics_outer = self._dilate_four_neighborhood(tight_component, 20)
                geometric_context = statistics_outer * (1.0 - statistics_inner)
                raw_highlights = raw_highlight_union[
                    batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
                ]
                same_light_repairs = full_inpaint_mask[
                    batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
                ]
                reference_context = geometric_context * (1.0 - raw_highlights) * \
                    (1.0 - same_light_repairs)
                repair_tasks.append(
                    (batch_index, roi_bounds, highlight_core_mask, repair_mask, reference_context)
                )

        return repair_tasks, full_inpaint_mask

    def _prepare_all_highlight_repair_tasks(self, binary_masks, exclusive_masks):
        """ 作用：以2.5像素欧氏距离聚合独占反光，并准备原始核心、0或10像素自适应修复区与10至20像素参考环 """
        group_distance = max(
            0.0, float(self.opt['train'].get('highlight_group_distance', 2.5))
        )
        highlight_count = torch.stack(
            [(mask > 0.5).to(mask.dtype) for mask in binary_masks], dim=0
        ).sum(dim=0)
        raw_highlight_union = (highlight_count > 0.0).to(exclusive_masks[0].dtype)
        multi_highlight_mask = (highlight_count >= 2.0).to(exclusive_masks[0].dtype)
        all_repair_tasks, inpaint_masks = [], []

        for exclusive_mask in exclusive_masks:
            grouped_envelopes = self._build_grouped_highlight_envelope(
                exclusive_mask, group_distance
            )
            repair_tasks, inpaint_mask = self._build_highlight_repair_and_context_masks(
                grouped_envelopes, exclusive_mask, raw_highlight_union,
                multi_highlight_mask
            )
            all_repair_tasks.append(repair_tasks)
            inpaint_masks.append(inpaint_mask)

        planned_union = torch.stack(inpaint_masks, dim=0).amax(dim=0)
        for light_index, repair_tasks in enumerate(all_repair_tasks):
            filtered_tasks = []
            for batch_index, roi_bounds, highlight_core_mask, repair_mask, reference_context in repair_tasks:
                y0, y1, x0, x1 = roi_bounds
                blocked = planned_union[batch_index:batch_index + 1, :, y0:y1, x0:x1]
                filtered_tasks.append(
                    (batch_index, roi_bounds, highlight_core_mask, repair_mask,
                     reference_context * (1.0 - blocked))
                )
            all_repair_tasks[light_index] = filtered_tasks

        return all_repair_tasks, inpaint_masks

    # 7. 反光处理（三）：候选排序、亮度校准、孔洞补全与伪目标生成
    def _masked_average_pool(self, image, mask, kernel_size):
        """ 作用：仅使用指定有效像素计算归一化局部均值，并同时返回每个位置是否具有统计支持 """
        padding = kernel_size // 2
        weight = F.avg_pool2d(mask.to(image.dtype), kernel_size, stride=1, padding=padding)
        average = F.avg_pool2d(image * mask.to(image.dtype), kernel_size,
                               stride=1, padding=padding) / weight.clamp_min(1e-6)
        return average, weight > 1e-6

    def _build_reference_ring_bounded_luminance_calibration(self, selected_rgb,
                                                             selected_valid,
                                                             current_roi,
                                                             repair_mask,
                                                             reference_context):
        """ 作用：以参考环学习有界低频亮度校准，保留加性高频并用乘性下界阻止缩暗时产生人为纯黑 """

        def _fit_huber_ridge(features, target):
            """ 作用：对坐标、候选低频亮度和常量截距执行单通道Huber稳健岭回归，退化时返回空值 """
            if features.size(0) < 20 or not torch.isfinite(features).all().item() or \
                    not torch.isfinite(target).all().item():
                return None
            ridge = features.new_tensor(max(float(features.size(0)) * 0.01, 0.1))
            regularizer = torch.eye(
                features.size(1), device=features.device, dtype=features.dtype
            )
            regularizer[-1, -1] = 0.0
            weights, coefficients = target.new_ones(target.size(0)), None
            for _ in range(4):
                weighted_features = features * weights.unsqueeze(1)
                matrix = features.transpose(0, 1).matmul(weighted_features) + \
                    ridge * regularizer
                vector = features.transpose(0, 1).matmul(target * weights.unsqueeze(1))
                try:
                    coefficients = torch.linalg.solve(matrix, vector)
                except RuntimeError:
                    return None
                if not torch.isfinite(coefficients).all().item():
                    return None
                residual = target - features.matmul(coefficients)
                residual_center = torch.median(residual, dim=0).values
                scale = 1.4826 * torch.median(torch.abs(residual - residual_center))
                if not torch.isfinite(scale).item() or scale.item() <= 1e-6:
                    break
                sample_error = torch.sqrt(torch.mean(residual.square(), dim=1))
                weights = torch.minimum(
                    torch.ones_like(sample_error),
                    1.345 * scale / sample_error.clamp_min(1e-6)
                )
            return coefficients

        safe_candidate = torch.where(
            torch.isfinite(selected_rgb), selected_rgb, torch.zeros_like(selected_rgb)
        )
        safe_current = torch.where(
            torch.isfinite(current_roi), current_roi, torch.zeros_like(current_roi)
        )
        candidate_valid = (selected_valid > 0.5) & \
            torch.isfinite(selected_rgb).all(1, keepdim=True)
        current_finite = torch.isfinite(current_roi).all(1, keepdim=True)
        reference_seed = (reference_context > 0.5) & candidate_valid & current_finite
        candidate_reconstruction_low, _ = self._masked_average_pool(
            safe_candidate, candidate_valid.to(selected_rgb.dtype), 7
        )
        candidate_reference_low, candidate_reference_support = self._masked_average_pool(
            safe_candidate, reference_seed.to(selected_rgb.dtype), 7
        )
        current_reference_low, current_reference_support = self._masked_average_pool(
            safe_current, reference_seed.to(current_roi.dtype), 7
        )
        reference = reference_seed & candidate_reference_support & current_reference_support
        reference_pixels = reference[0, 0]
        if torch.count_nonzero(reference_pixels).item() == 0:
            return safe_candidate

        candidate_reconstruction_luma = self._rgb_to_luminance(candidate_reconstruction_low)
        candidate_reference_luma = self._rgb_to_luminance(candidate_reference_low)
        current_reference_luma = self._rgb_to_luminance(current_reference_low)
        height, width = selected_rgb.shape[2:]
        x_line = torch.linspace(-1.0, 1.0, width, device=selected_rgb.device,
                                dtype=selected_rgb.dtype) if width > 1 else selected_rgb.new_zeros(1)
        y_line = torch.linspace(-1.0, 1.0, height, device=selected_rgb.device,
                                dtype=selected_rgb.dtype) if height > 1 else selected_rgb.new_zeros(1)
        x_coordinate = x_line.view(1, 1, 1, width).expand(1, 1, height, width)
        y_coordinate = y_line.view(1, 1, height, 1).expand(1, 1, height, width)
        feature_map = torch.cat((
            x_coordinate, y_coordinate, candidate_reconstruction_luma,
            torch.ones_like(x_coordinate)
        ), dim=1)
        reference_features = feature_map[0, :, reference_pixels].transpose(0, 1)
        reference_luminance_residual = (current_reference_luma - candidate_reference_luma)[
            0, :, reference_pixels
        ].transpose(0, 1)
        median_luminance_residual = torch.median(
            reference_luminance_residual, dim=0
        ).values
        coefficients = _fit_huber_ridge(reference_features, reference_luminance_residual)
        if coefficients is None:
            predicted_luminance_residual = median_luminance_residual.view(
                1, 1, 1, 1
            ).expand_as(candidate_reconstruction_luma)
        else:
            predicted_luminance_residual = feature_map.permute(0, 2, 3, 1).reshape(
                -1, feature_map.size(1)
            ).matmul(coefficients).view(1, height, width, 1).permute(0, 3, 1, 2)
        target_low_luma = candidate_reconstruction_luma + predicted_luminance_residual
        luminance_ratio = torch.clamp(
            target_low_luma / candidate_reconstruction_luma.clamp_min(1e-4), 0.75, 1.50
        )
        candidate_high = safe_candidate - candidate_reconstruction_low
        additive_target = candidate_reconstruction_low * luminance_ratio + candidate_high
        multiplicative_floor = safe_candidate * luminance_ratio
        calibrated = torch.clamp(
            torch.where(
                (luminance_ratio < 1.0).expand_as(additive_target),
                torch.maximum(additive_target, multiplicative_floor), additive_target
            ), 0.0, 1.0
        )
        return torch.where((repair_mask > 0.5).expand_as(calibrated), calibrated, safe_candidate)

    def _fill_highlight_target_holes(self, selected_rgb, selected_valid, current_roi,
                                     repair_mask, reference_context):
        """ 作用：以候选RGB和外围正常颜色为种子进行四邻域传播，并确定性补全修复组件中的少量无效位置 """
        repair = repair_mask > 0.5
        candidate_seed = repair & selected_valid & torch.isfinite(selected_rgb).all(1, keepdim=True)
        boundary_seed = (reference_context > 0.5) & torch.isfinite(current_roi).all(1, keepdim=True)
        if not torch.any(candidate_seed | boundary_seed).item():
            raise _HighlightPseudoTargetError('候选RGB与外围正常颜色均无法为反光组件提供有限种子。')
        domain = self._dilate_four_neighborhood(repair_mask, 6) > 0.5
        values = torch.where(candidate_seed.expand_as(selected_rgb), selected_rgb,
                             torch.zeros_like(selected_rgb))
        values = torch.where(boundary_seed.expand_as(values), current_roi, values)
        valid = (candidate_seed | boundary_seed) & domain
        for _ in range(12):
            valid_float = valid.to(values.dtype)
            neighbor_count, neighbor_sum = torch.zeros_like(valid_float), torch.zeros_like(values)
            neighbor_count[:, :, 1:] += valid_float[:, :, :-1]
            neighbor_count[:, :, :-1] += valid_float[:, :, 1:]
            neighbor_count[:, :, :, 1:] += valid_float[:, :, :, :-1]
            neighbor_count[:, :, :, :-1] += valid_float[:, :, :, 1:]
            neighbor_sum[:, :, 1:] += values[:, :, :-1] * valid_float[:, :, :-1]
            neighbor_sum[:, :, :-1] += values[:, :, 1:] * valid_float[:, :, 1:]
            neighbor_sum[:, :, :, 1:] += values[:, :, :, :-1] * valid_float[:, :, :, :-1]
            neighbor_sum[:, :, :, :-1] += values[:, :, :, 1:] * valid_float[:, :, :, 1:]
            new_valid = domain & ~valid & (neighbor_count > 0.0)
            values = torch.where(
                new_valid.expand_as(values), neighbor_sum / neighbor_count.clamp_min(1.0), values
            )
            valid |= new_valid
        remaining = repair & ~valid
        if torch.any(remaining).item():
            fallback_mask = boundary_seed if torch.any(boundary_seed).item() else candidate_seed
            fallback_rgb = torch.sum(values * fallback_mask.to(values.dtype), dim=(2, 3), keepdim=True) / \
                torch.sum(fallback_mask, dim=(2, 3), keepdim=True).to(values.dtype).clamp_min(1.0)
            values = torch.where(remaining.expand_as(values), fallback_rgb, values)
        if not torch.isfinite(torch.where(repair.expand_as(values), values,
                                          torch.zeros_like(values))).all().item():
            raise _HighlightPseudoTargetError('补全后的反光候选目标包含非有限数值。')
        return values

    def _build_two_pixel_low_frequency_transition(self, repaired_rgb, current_rgb,
                                                   repair_mask, highlight_core_mask,
                                                   reference_context):
        """ 作用：以单边暗度缺口校正大组件边缘和小组件，保留不暗于参考的候选纹理并在参考不足时保持原值 """
        repair = (repair_mask > 0.5).to(repaired_rgb.dtype)
        core = (highlight_core_mask > 0.5).to(repaired_rgb.dtype)
        eroded_once = repair * (1.0 - self._dilate_four_neighborhood(1.0 - repair, 1))
        eroded_twice = eroded_once * (
            1.0 - self._dilate_four_neighborhood(1.0 - eroded_once, 1)
        )
        outer_layer, inner_layer = repair - eroded_once, eroded_once - eroded_twice
        safe_current = torch.where(
            torch.isfinite(current_rgb), current_rgb, torch.zeros_like(current_rgb)
        )
        normal_mask = (reference_context > 0.5).to(repaired_rgb.dtype) * \
            torch.isfinite(current_rgb).all(1, keepdim=True).to(repaired_rgb.dtype)
        normal_low, normal_support = self._masked_average_pool(
            safe_current, normal_mask, 11
        )
        large_component = (torch.sum(core) >= 20.0).to(repaired_rgb.dtype)
        large_geometry_weight = (outer_layer * (2.0 / 3.0) + inner_layer * (1.0 / 3.0)) * \
            (1.0 - core) * normal_support.to(repaired_rgb.dtype) * \
            large_component
        reference_pixels = normal_mask[0, 0] > 0.5
        has_reference = torch.any(reference_pixels).item()
        reference_anchor = torch.median(
            safe_current[0, :, reference_pixels], dim=1
        ).values.view(1, 3, 1, 1) if has_reference else repaired_rgb
        small_geometry_weight = repair * (2.0 / 3.0) * (1.0 - large_component) \
            if has_reference else torch.zeros_like(repair)
        candidate_luma = self._rgb_to_luminance(repaired_rgb)
        normal_luma, anchor_luma = self._rgb_to_luminance(normal_low), \
            self._rgb_to_luminance(reference_anchor)
        large_weight = large_geometry_weight * torch.clamp(
            F.relu(normal_luma - candidate_luma) / normal_luma.clamp_min(1e-4), 0.0, 1.0
        )
        small_weight = small_geometry_weight * torch.clamp(
            F.relu(anchor_luma - candidate_luma) / anchor_luma.clamp_min(1e-4), 0.0, 1.0
        )
        blend_weight = large_weight + small_weight
        transitioned = torch.clamp(
            repaired_rgb * (1.0 - blend_weight) + normal_low * large_weight +
            reference_anchor * small_weight, 0.0, 1.0
        )
        return torch.where((blend_weight > 0.0).expand_as(repaired_rgb), transitioned, repaired_rgb)

    def _build_ranked_calibrated_candidate_rgb(self, candidate_rois,
                                                candidate_valid_masks,
                                                current_roi, repair_mask,
                                                reference_context):
        """ 作用：按原始清晰度排序候选，执行对数域稳健乘性标定、主次回退、参考环有界亮度低频校准及空洞补全 """

        def _fit_robust_origin_gain(source, target, prior=None):
            """ 作用：仅排除非有限和近零数值，以对数域Huber稳健中心估计公共或通道独立增益 """
            fallback = None if prior is None else prior
            stable = torch.isfinite(source) & torch.isfinite(target) & \
                (source > 1e-6) & (target > 1e-6)
            if torch.count_nonzero(stable).item() < 4:
                return fallback
            log_ratio = torch.log(target[stable].float()) - torch.log(source[stable].float())
            prior_value = None if prior is None else prior.float()
            if prior_value is not None and (
                    not torch.isfinite(prior_value).item() or prior_value.item() <= 0.0):
                return fallback
            log_offset = log_ratio if prior_value is None else log_ratio - torch.log(prior_value)
            center = torch.median(log_offset)
            regularizer = log_offset.new_zeros(()) if prior_value is None else \
                log_offset.new_tensor(log_offset.numel() * 0.01)
            for _ in range(4):
                residual = log_offset - center
                residual_center = torch.median(residual)
                scale = 1.4826 * torch.median(torch.abs(residual - residual_center))
                if not torch.isfinite(scale).item() or scale.item() <= 1e-6:
                    center = center * log_offset.numel() / (log_offset.numel() + regularizer)
                    break
                weights = torch.minimum(
                    torch.ones_like(residual),
                    1.345 * scale / torch.abs(residual).clamp_min(1e-6)
                )
                updated = torch.sum(weights * log_offset) / \
                    (torch.sum(weights) + regularizer).clamp_min(1e-10)
                if not torch.isfinite(updated).item():
                    return fallback
                center = updated
            gain = torch.exp(torch.clamp(center, -20.0, 20.0))
            gain = gain if prior_value is None else prior_value * gain
            return gain.to(source.dtype) if torch.isfinite(gain).item() and gain.item() > 0.0 \
                else fallback

        finite_candidates = torch.isfinite(candidate_rois).all(1, keepdim=True)
        current_finite = torch.isfinite(current_roi).all(1, keepdim=True)
        valid_candidates = (candidate_valid_masks > 0.5) & finite_candidates
        safe_candidates = torch.where(
            torch.isfinite(candidate_rois), candidate_rois, torch.zeros_like(candidate_rois)
        )
        safe_current = torch.where(
            torch.isfinite(current_roi), current_roi, torch.zeros_like(current_roi)
        )
        repair = repair_mask > 0.5
        candidate_luma = self._rgb_to_luminance(safe_candidates)
        normalized_luma, has_repair_pixels = [], []
        for candidate_index in range(candidate_rois.size(0)):
            valid_repair = valid_candidates[candidate_index:candidate_index + 1] & repair
            has_pixels = torch.any(valid_repair).item()
            normalizer = torch.median(
                candidate_luma[candidate_index, 0][valid_repair[0, 0]]
            ).clamp_min(0.03) if has_pixels else candidate_luma.new_ones(())
            normalized_luma.append(
                candidate_luma[candidate_index:candidate_index + 1] / normalizer
            )
            has_repair_pixels.append(has_pixels)
        normalized_luma = torch.cat(normalized_luma, dim=0)
        sharpness_valid = valid_candidates & repair
        grad_x, grad_y = self._compute_first_order_gradients(normalized_luma)
        valid_x = F.pad(
            (sharpness_valid[:, :, :, 1:] & sharpness_valid[:, :, :, :-1]).to(candidate_rois.dtype),
            (0, 1, 0, 0), mode='constant', value=0.0
        )
        valid_y = F.pad(
            (sharpness_valid[:, :, 1:, :] & sharpness_valid[:, :, :-1, :]).to(candidate_rois.dtype),
            (0, 0, 0, 1), mode='constant', value=0.0
        )
        numerator = torch.sum(
            torch.abs(grad_x) * valid_x + torch.abs(grad_y) * valid_y,
            dim=(1, 2, 3)
        )
        denominator = torch.sum(valid_x + valid_y, dim=(1, 2, 3))
        sharpness = torch.where(
            denominator > 0.0, numerator / denominator.clamp_min(1e-5),
            torch.zeros_like(numerator)
        )

        current_luma = self._rgb_to_luminance(safe_current)
        common_gains, relative_channel_gains = [], []
        for candidate_index in range(candidate_rois.size(0)):
            paired_reference = (reference_context > 0.5) & current_finite & \
                valid_candidates[candidate_index:candidate_index + 1]
            if torch.count_nonzero(paired_reference).item() < 4:
                common_gains.append(candidate_luma.new_ones(()))
                relative_channel_gains.append(candidate_luma.new_ones(3))
                continue
            pair = paired_reference[0, 0]
            common_gain = _fit_robust_origin_gain(
                candidate_luma[candidate_index, 0][pair],
                current_luma[0, 0][pair]
            )
            if common_gain is None:
                common_gains.append(candidate_luma.new_ones(()))
                relative_channel_gains.append(candidate_luma.new_ones(3))
                continue
            common_gain = torch.clamp(common_gain, 0.25, 4.0)
            relative_gains = []
            for channel in range(3):
                channel_gain = _fit_robust_origin_gain(
                    safe_candidates[candidate_index, channel][pair],
                    safe_current[0, channel][pair], common_gain
                )
                relative_gains.append(channel_gain / common_gain)
            common_gains.append(common_gain)
            relative_channel_gains.append(torch.stack(relative_gains))

        common_gains = torch.stack(common_gains)
        relative_channel_gains = torch.stack(relative_channel_gains)
        gain_maps = torch.clamp(
            common_gains.view(-1, 1, 1, 1) *
            relative_channel_gains.view(-1, 3, 1, 1),
            0.25, 4.0
        )
        calibrated_candidates = torch.clamp(safe_candidates * gain_maps, 0.0, 1.0)
        availability_vector = torch.tensor(
            has_repair_pixels, device=candidate_rois.device, dtype=torch.bool
        )
        availability = availability_vector.view(-1, 1, 1, 1)
        valid_candidates &= availability
        sharpness = torch.where(
            availability_vector, sharpness,
            torch.full_like(sharpness, float('-inf'))
        )
        if torch.any(availability_vector).item():
            primary_selector = F.one_hot(
                torch.argmax(sharpness), num_classes=candidate_rois.size(0)
            ).to(candidate_rois.dtype).view(-1, 1, 1, 1)
            secondary_selector = 1.0 - primary_selector
            primary_rgb = torch.sum(calibrated_candidates * primary_selector, dim=0, keepdim=True)
            secondary_rgb = torch.sum(calibrated_candidates * secondary_selector, dim=0, keepdim=True)
            valid_float = valid_candidates.to(candidate_rois.dtype)
            primary_valid = torch.sum(valid_float * primary_selector, dim=0, keepdim=True) > 0.5
            secondary_valid = torch.sum(valid_float * secondary_selector, dim=0, keepdim=True) > 0.5
            selected_valid = primary_valid | secondary_valid
            selected_rgb = torch.where(
                primary_valid.expand_as(primary_rgb), primary_rgb, secondary_rgb
            )
        else:
            selected_rgb, selected_valid = torch.zeros_like(current_roi), torch.zeros_like(repair)
        selected_rgb = self._build_reference_ring_bounded_luminance_calibration(
            selected_rgb, selected_valid, safe_current, repair_mask, reference_context
        )
        selected_rgb = self._fill_highlight_target_holes(
            selected_rgb, selected_valid, current_roi, repair_mask, reference_context
        )
        return selected_rgb

    def _rgb_to_luminance(self, image):
        """ 作用：使用固定感知权重将RGB图像转换为单通道亮度，统一候选清晰度和反光损失的亮度定义 """
        weights = image.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
        return torch.sum(image * weights, dim=1, keepdim=True)

    def _build_exclusive_highlight_pseudo_targets(self, clean_gt, repair_tasks,
                                                   binary_highlight_masks,
                                                   light_index):
        """ 作用：生成含大组件边缘与小组件单边暗度缺口校正的自适应RGB伪目标，并记录完整成功掩膜 """
        calibrated_regions = []
        successful_inpaint_mask = clean_gt.new_zeros(
            (clean_gt.size(0), 1, clean_gt.size(2), clean_gt.size(3))
        )
        if not repair_tasks:
            return calibrated_regions, successful_inpaint_mask

        gt_current = clean_gt[:, light_index * 3:(light_index + 1) * 3]
        source_indices = [index for index in range(3) if index != light_index]
        for repair_task in repair_tasks:
            batch_index, roi_bounds, highlight_core_mask, repair_mask, reference_context = repair_task
            roi_y0, roi_y1, roi_x0, roi_x1 = roi_bounds
            candidate_rois = torch.cat([
                clean_gt[
                    batch_index:batch_index + 1,
                    source_index * 3:(source_index + 1) * 3,
                    roi_y0:roi_y1, roi_x0:roi_x1
                ]
                for source_index in source_indices
            ], dim=0)
            candidate_highlights = torch.cat([
                binary_highlight_masks[source_index][
                    batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
                ]
                for source_index in source_indices
            ], dim=0)
            candidate_valid = (candidate_highlights < 0.5) & \
                torch.isfinite(candidate_rois).all(1, keepdim=True)
            current_roi = gt_current[
                batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
            ]
            try:
                selected_rgb = self._build_ranked_calibrated_candidate_rgb(
                    candidate_rois, candidate_valid, current_roi,
                    repair_mask, reference_context
                )
                selected_rgb = self._build_two_pixel_low_frequency_transition(
                    selected_rgb, current_roi, repair_mask,
                    highlight_core_mask, reference_context
                )
                target = torch.where(repair_mask > 0.5, selected_rgb, current_roi)
                if not torch.isfinite(target).all().item():
                    raise _HighlightPseudoTargetError('反光直接RGB伪目标包含非有限数值。')
                calibrated_regions.append(
                    (batch_index, roi_bounds, highlight_core_mask, repair_mask, target)
                )
                successful_inpaint_mask[
                    batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
                ] = torch.maximum(
                    successful_inpaint_mask[
                        batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1
                    ],
                    repair_mask
                )
            except _HighlightPseudoTargetError:
                continue

        return calibrated_regions, successful_inpaint_mask

    # 8. 反光损失：局部对比度、梯度、色度、困难像素与光源隔离约束
    def _compute_local_std(self, x, kernel_size=21):
        """ 作用：计算输入图像在 21x21 滑动窗口下的局部标准差(局部对比度)图 """
        pad = kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        x_mean = F.avg_pool2d(x_pad, kernel_size=kernel_size, stride=1)
        x_sq_mean = F.avg_pool2d(x_pad ** 2, kernel_size=kernel_size, stride=1)
        return torch.sqrt(F.relu(x_sq_mean - x_mean ** 2) + 1e-5)

    def _cal_contrast_std_loss(self, p_std, std_target, mask):
        """ 作用：仅在亮度域计算高光预测区域与校准目标的局部标准差匹配损失 """
        diff_std = torch.abs(p_std - std_target) * mask
        return torch.sum(diff_std) / (torch.sum(mask) + 1e-5)

    def _compute_first_order_gradients(self, x):
        """ 作用：统一计算水平与垂直一阶保号前向梯度，并在图像右侧和下侧使用零梯度填充 """
        grad_x = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), mode='constant', value=0)
        grad_y = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), mode='constant', value=0)
        return grad_x, grad_y

    def _count_total_isolation_active_pairs(self, binary_highlight_masks, planned_inpaint_masks, batch_size):
        """ 作用：统计Batch内所有光源间产生反光污染隔离保护的有效配对总数 """
        total_count = 0.0
        for hl_idx in range(3):
            affected = torch.maximum(binary_highlight_masks[hl_idx], planned_inpaint_masks[hl_idx])
            for prot_idx in range(3):
                if prot_idx == hl_idx:
                    continue
                protect = affected * (1.0 - binary_highlight_masks[prot_idx])
                for b in range(batch_size):
                    if torch.any(protect[b:b + 1] > 0.5).item():
                        total_count += 1.0
        return max(total_count, 1.0)

    def _cal_single_target_isolation_loss(self, p_sub, gt_sub, binary_highlight_masks,
                                          planned_inpaint_masks, target_index, total_active_count):
        """ 作用：计算指定目标光源在其他光源反光遮罩下的光源隔离保护损失 """
        loss_sum = p_sub.sum() * 0.0
        for hl_idx in range(3):
            if hl_idx == target_index:
                continue
            affected_mask = torch.maximum(binary_highlight_masks[hl_idx], planned_inpaint_masks[hl_idx])
            protect_mask = affected_mask * (1.0 - binary_highlight_masks[target_index])
            for b in range(p_sub.size(0)):
                batch_mask = protect_mask[b:b + 1]
                if not torch.any(batch_mask > 0.5).item():
                    continue
                batch_pred = p_sub[b:b + 1]
                batch_gt = gt_sub[b:b + 1]
                rgb_error = torch.mean(torch.abs(batch_pred - batch_gt), dim=1, keepdim=True)
                luma_excess = F.relu(self._rgb_to_luminance(batch_pred) - self._rgb_to_luminance(batch_gt))
                pollution_strength = luma_excess + 4.0 * luma_excess.square()
                polluted_pixels = (batch_mask > 0.5) & (luma_excess > 0.0)
                full_loss = torch.sum(rgb_error * batch_mask) / torch.sum(batch_mask)
                hard_loss = full_loss * 0.0
                if torch.any(polluted_pixels).item():
                    polluted_strength = pollution_strength[polluted_pixels]
                    hard_count = polluted_strength.numel() if polluted_strength.numel() < 20 else max(1, (polluted_strength.numel() + 9) // 10)
                    hard_indices = torch.topk(polluted_strength, hard_count, largest=True, sorted=False).indices
                    hard_loss = (rgb_error + pollution_strength)[polluted_pixels][hard_indices].mean()
                loss_sum = loss_sum + full_loss + hard_loss
        return loss_sum / total_active_count

    def _cal_single_target_highlight_loss(self, p_sub, clean_gt, calibrated_region_i,
                                          binary_highlight_masks, planned_inpaint_masks,
                                          light_index, total_calibrated_count, total_active_count):
        """ 作用：仅针对当前目标光源的3通道预测，计算反光伪目标监督与非反光光源隔离保护损失 """
        def _build_interior_structure_mask(repair_mask):
            repair = (repair_mask > 0.5).to(repair_mask.dtype)
            invalid = F.pad(1.0 - repair, (3, 3, 3, 3), mode='constant', value=1.0)
            return repair * (1.0 - F.max_pool2d(invalid, kernel_size=7, stride=1))

        def _cal_sign_preserving_grad_loss(pred_grad_x, pred_grad_y, target_grad_x, target_grad_y, mask):
            mask_x = F.pad(torch.minimum(mask[:, :, :, 1:], mask[:, :, :, :-1]), (0, 1, 0, 0), mode='constant', value=0.0)
            mask_y = F.pad(torch.minimum(mask[:, :, 1:, :], mask[:, :, :-1, :]), (0, 0, 0, 1), mode='constant', value=0.0)
            loss_x = torch.abs(pred_grad_x - target_grad_x) * mask_x
            loss_y = torch.abs(pred_grad_y - target_grad_y) * mask_y
            edge_count = torch.sum(mask_x) + torch.sum(mask_y)
            return torch.sum(loss_x + loss_y) / (edge_count * 3.0 + 1e-5)

        def _cal_nonlinear_dc_tone_loss(p_region, tone_target, mask):
            diff = torch.abs(p_region - tone_target)
            loss_local = torch.sum((diff + 2.0 * (diff ** 2)) * mask) / (torch.sum(mask) * 3.0 + 1e-5)
            mask_denom = torch.sum(mask, dim=[2, 3], keepdim=True) + 1e-5
            valid_hl_count = torch.sum((mask_denom > 1e-3).float()) + 1e-5
            p_macro = torch.sum(p_region * mask, dim=[2, 3], keepdim=True) / mask_denom
            target_macro = torch.sum(tone_target * mask, dim=[2, 3], keepdim=True) / mask_denom
            return loss_local + torch.sum((p_macro - target_macro) ** 2) / (valid_hl_count * 3.0) * 1.5

        def _cal_texture_or_appearance_fallback_loss(p_region, target, p_std, target_std,
                                                      pred_grad_x, pred_grad_y,
                                                      target_grad_x, target_grad_y,
                                                      repair_mask, structure_mask):
            gradient_energy = torch.mean(torch.abs(target_grad_x) + torch.abs(target_grad_y), 1, keepdim=True)
            local_gradient_energy = F.avg_pool2d(gradient_energy, kernel_size=7, stride=1, padding=3)
            gradient_confidence = 1.0 - torch.exp(-local_gradient_energy / 0.02)
            contrast_confidence = 1.0 - torch.exp(-target_std / 0.02)
            local_reliability = torch.clamp(torch.sqrt(gradient_confidence * contrast_confidence) * structure_mask, 0.0, 1.0)
            appearance_mask = repair_mask * (1.0 - local_reliability)
            texture_loss = _cal_sign_preserving_grad_loss(
                pred_grad_x, pred_grad_y, target_grad_x, target_grad_y, local_reliability
            ) + self._cal_contrast_std_loss(p_std, target_std, local_reliability)
            appearance_loss = _cal_nonlinear_dc_tone_loss(p_region, target, appearance_mask) + \
                self._cal_contrast_std_loss(p_std, target_std, appearance_mask)
            texture_coverage = torch.sum(local_reliability) / torch.sum(repair_mask).clamp_min(1.0)
            return texture_coverage * texture_loss + (1.0 - texture_coverage) * appearance_loss

        def _cal_highlight_hard_pixel_removal_loss(p_region, target, core_mask, repair_mask):
            def _cal_region_hard_loss(region_mask, pixel_error):
                values = pixel_error[region_mask]
                if values.numel() == 0:
                    return None
                hard_count = values.numel() if values.numel() < 20 else max(1, (values.numel() + 4) // 5)
                return torch.topk(values, hard_count, largest=True, sorted=False).values.mean()

            repair = repair_mask > 0.5
            core = (core_mask > 0.5) & repair
            outer = repair & ~core
            rgb_absolute_error = torch.abs(p_region - target)
            rgb_error = torch.mean(rgb_absolute_error, dim=1, keepdim=True)
            rgb_squared_error = torch.mean(rgb_absolute_error.square(), dim=1, keepdim=True)
            luma_error = torch.abs(self._rgb_to_luminance(p_region) - self._rgb_to_luminance(target))
            pixel_error = rgb_error + 3.0 * rgb_squared_error + luma_error + 4.0 * luma_error.square()
            part_losses = [l for l in (_cal_region_hard_loss(core, pixel_error), _cal_region_hard_loss(outer, pixel_error)) if l is not None]
            return torch.stack(part_losses).mean() if part_losses else p_region.sum() * 0.0

        def _cal_chromaticity_loss(p_region, color_target, mask):
            pred_positive = torch.clamp(p_region, min=0.0)
            target_positive = torch.clamp(color_target, min=0.0)
            target_luma = self._rgb_to_luminance(color_target)
            pred_chroma = pred_positive / pred_positive.sum(1, keepdim=True).clamp_min(0.03)
            target_chroma = target_positive / target_positive.sum(1, keepdim=True).clamp_min(0.03)
            color_weight = target_luma / (target_luma + 0.05)
            weighted_mask = mask * color_weight
            return torch.sum(torch.abs(pred_chroma - target_chroma) * weighted_mask) / (torch.sum(mask) * 3.0 + 1e-5)

        tone_losses, tex_app_losses, chroma_losses, hard_losses = [], [], [], []
        p_luma = self._rgb_to_luminance(p_sub)
        p_std = self._compute_local_std(p_luma, kernel_size=7)
        pred_grad_x, pred_grad_y = self._compute_first_order_gradients(p_sub)

        for batch_index, roi_bounds, target_core, target_component, calibrated_target in calibrated_region_i:
            roi_y0, roi_y1, roi_x0, roi_x1 = roi_bounds
            structure_mask = _build_interior_structure_mask(target_component)
            target_luma = self._rgb_to_luminance(calibrated_target)
            target_std = self._compute_local_std(target_luma, kernel_size=7)
            target_grad_x, target_grad_y = self._compute_first_order_gradients(calibrated_target)
            pred_region = p_sub[batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1]
            pred_std_region = p_std[batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1]
            pred_grad_x_region = pred_grad_x[batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1]
            pred_grad_y_region = pred_grad_y[batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1]

            tone_losses.append(_cal_nonlinear_dc_tone_loss(pred_region, calibrated_target, target_component))
            tex_app_losses.append(_cal_texture_or_appearance_fallback_loss(
                pred_region, calibrated_target, pred_std_region, target_std,
                pred_grad_x_region, pred_grad_y_region, target_grad_x,
                target_grad_y, target_component, structure_mask
            ))
            chroma_losses.append(_cal_chromaticity_loss(pred_region, calibrated_target, target_component))
            hard_losses.append(_cal_highlight_hard_pixel_removal_loss(pred_region, calibrated_target, target_core, target_component))

        zero_loss = p_sub.sum() * 0.0
        denom = max(float(total_calibrated_count), 1.0)
        loss_tone = torch.stack(tone_losses).sum() / denom if tone_losses else zero_loss
        loss_tex_app = torch.stack(tex_app_losses).sum() / denom if tex_app_losses else zero_loss
        loss_chroma = torch.stack(chroma_losses).sum() / denom if chroma_losses else zero_loss
        loss_hard = torch.stack(hard_losses).sum() / denom if hard_losses else zero_loss

        gt_sub = clean_gt[:, light_index * 3:(light_index + 1) * 3]
        loss_isolation = self._cal_single_target_isolation_loss(
            p_sub, gt_sub, binary_highlight_masks, planned_inpaint_masks, light_index, total_active_count
        )

        l_hl_i = (loss_tone * 4.0 + loss_tex_app * 4.0 + loss_chroma * 2.0 + loss_hard * 4.0 + loss_isolation * 4.0)
        sub_dict_i = {
            'l_hl_tone': loss_tone,
            'l_hl_tex_or_app': loss_tex_app,
            'l_hl_chroma': loss_chroma,
            'l_hl_hard': loss_hard,
            'l_hl_isolation': loss_isolation
        }
        return l_hl_i, sub_dict_i

    def _cal_decoupled_highlight_loss(self, pred, calibrated_regions, clean_gt,
                                      planned_inpaint_masks, binary_highlight_masks):
        """ 作用：以固定总权重监督反光修复并隔离反光覆盖区内其他非反光光源，纹理不可靠时连续回退到候选外观监督 """
        total_calibrated_count = sum(len(r) for r in calibrated_regions)
        total_active_count = self._count_total_isolation_active_pairs(
            binary_highlight_masks, planned_inpaint_masks, clean_gt.size(0)
        )
        l_highlight = 0.0
        highlight_sub_dict = {'l_hl_tone': 0.0, 'l_hl_tex_or_app': 0.0, 'l_hl_chroma': 0.0, 'l_hl_hard': 0.0, 'l_hl_isolation': 0.0}
        for i in range(3):
            p_sub = pred[:, i * 3:(i + 1) * 3, :, :]
            l_hl_i, hl_dict_i = self._cal_single_target_highlight_loss(
                p_sub, clean_gt, calibrated_regions[i], binary_highlight_masks,
                planned_inpaint_masks, i, total_calibrated_count, total_active_count
            )
            l_highlight = l_highlight + l_hl_i
            for k in highlight_sub_dict:
                highlight_sub_dict[k] = highlight_sub_dict[k] + hl_dict_i[k]
        return l_highlight, highlight_sub_dict

    # 9. 基础像素损失与单步优化：汇总各项损失、反向传播并更新参数
    def _cal_single_target_pixel_loss(self, pred_sub, successful_inpaint_mask, light_index):
        """ 作用：计算单目标光源3通道切片在健康区域的基础像素损失并按3通道等权归一化 """
        healthy_mask = 1.0 - torch.repeat_interleave(successful_inpaint_mask, 3, dim=1)
        gt_sub = self.gt[:, light_index * 3:(light_index + 1) * 3]
        preds_list = pred_sub if isinstance(pred_sub, list) else [pred_sub]
        return sum(self.cri_pix(p * healthy_mask, gt_sub * healthy_mask) / 3.0 for p in preds_list)

    def _cal_masked_pixel_loss(self, preds, successful_inpaint_masks):
        """ 作用：基础像素损失仅移除成功获得伪目标监督的自适应修复区，失败区域保留原始GT监督 """
        highlight_mask_9ch = torch.repeat_interleave(
            torch.cat(successful_inpaint_masks, dim=1), 3, dim=1
        )
        healthy_mask = 1.0 - highlight_mask_9ch

        l_pix = 0.0
        preds_list = preds if isinstance(preds, list) else [preds]
        for pred in preds_list:
            l_pix += self.cri_pix(pred * healthy_mask, self.gt * healthy_mask)
        return l_pix

    def optimize_parameters(self, current_iter):
        """ 作用：执行单步反向优化(提前生成阻断掩膜、分路前向即时反向传播释放显存、计算损失并更新权重) """
        self.optimizer_g.zero_grad()

        with torch.no_grad():
            binary_highlight_masks, exclusive_highlight_masks = self._arbitrate_highlight_masks()
            repair_tasks_by_light, planned_inpaint_masks = \
                self._prepare_all_highlight_repair_tasks(
                    binary_highlight_masks, exclusive_highlight_masks
                )
            highlight_block_masks = self._synthesize_highlight_block_masks(
                binary_highlight_masks, planned_inpaint_masks
            )
            masks_s, priors_activated = self._extract_image_driven_shadow(self.raw_prior)
            calibrated_regions = []
            successful_inpaint_masks = []
            for i in range(3):
                light_regions, successful_mask = \
                    self._build_exclusive_highlight_pseudo_targets(
                        self.gt, repair_tasks_by_light[i],
                        binary_highlight_masks, i
                    )
                calibrated_regions.append(light_regions)
                successful_inpaint_masks.append(successful_mask)
            effective_shadow_masks = [
                masks_s[i] * (1.0 - planned_inpaint_masks[i]) for i in range(3)
            ]
            gt_stats = self._prepare_shadow_gt_statistics(self.gt, highlight_block_masks)
            severity_weights = self._compute_relative_shadow_severity()
            total_calibrated_count = sum(len(r) for r in calibrated_regions)
            total_active_count = self._count_total_isolation_active_pairs(
                binary_highlight_masks, planned_inpaint_masks, self.gt.size(0)
            )
            has_highlights = torch.any(torch.cat(highlight_block_masks, dim=1) > 0.5).item()

        loss_dict = OrderedDict()

        if not has_highlights:
            preds = self.net_g(self.lq)
            preds_list = preds if isinstance(preds, list) else [preds]
            self.output = preds_list[-1]

            l_pix = self._cal_masked_pixel_loss(preds_list, successful_inpaint_masks)
            shadow_total, shadow_sub_dict = self._cal_shadow_consistency_loss(
                self.output, self.gt, effective_shadow_masks, priors_activated, highlight_block_masks
            )
            l_highlight, highlight_sub_dict = self._cal_decoupled_highlight_loss(
                self.output, calibrated_regions, self.gt, planned_inpaint_masks, binary_highlight_masks
            )
            total_loss = l_pix + shadow_total * 0.5 + l_highlight * 0.15
            total_loss.backward()
        else:
            total_loss_val = 0.0
            l_pix_val = 0.0
            shadow_total_val = 0.0
            l_hl_val = 0.0
            shadow_sub_dict = {'l_shd_bright': 0.0, 'l_shd_tex': 0.0, 'l_shd_polarity': 0.0, 'l_shd_color': 0.0, 'l_shd_ssim': 0.0}
            highlight_sub_dict = {'l_hl_tone': 0.0, 'l_hl_tex_or_app': 0.0, 'l_hl_chroma': 0.0, 'l_hl_hard': 0.0, 'l_hl_isolation': 0.0}
            saved_preds_sub = []

            for i in range(3):
                target_input = self._build_target_light_input(self.lq, highlight_block_masks, i)
                pred_i = self.net_g(target_input)
                pred_i_list = pred_i if isinstance(pred_i, list) else [pred_i]
                pred_i_sub_list = [p[:, i * 3:(i + 1) * 3] for p in pred_i_list]
                p_sub_final = pred_i_sub_list[-1]
                saved_preds_sub.append(p_sub_final.detach())

                l_pix_i = self._cal_single_target_pixel_loss(pred_i_sub_list, successful_inpaint_masks[i], i)
                shadow_i, s_dict_i = self._cal_single_target_shadow_loss(
                    p_sub_final, self.gt, effective_shadow_masks, priors_activated, i, gt_stats, severity_weights
                )
                l_hl_i, hl_dict_i = self._cal_single_target_highlight_loss(
                    p_sub_final, self.gt, calibrated_regions[i], binary_highlight_masks,
                    planned_inpaint_masks, i, total_calibrated_count, total_active_count
                )
                loss_i = l_pix_i + shadow_i * 0.5 + l_hl_i * 0.15
                loss_i.backward()

                total_loss_val = total_loss_val + loss_i.detach()
                l_pix_val = l_pix_val + l_pix_i.detach()
                shadow_total_val = shadow_total_val + shadow_i.detach()
                l_hl_val = l_hl_val + l_hl_i.detach()
                for k in shadow_sub_dict:
                    shadow_sub_dict[k] = shadow_sub_dict[k] + s_dict_i[k].detach()
                for k in highlight_sub_dict:
                    highlight_sub_dict[k] = highlight_sub_dict[k] + hl_dict_i[k].detach()

            self.output = torch.cat(saved_preds_sub, dim=1)
            total_loss = total_loss_val
            l_pix = l_pix_val
            shadow_total = shadow_total_val
            l_highlight = l_hl_val

        loss_dict['total_loss'] = total_loss
        loss_dict['l_shadow'] = shadow_total
        loss_dict['l_highlight'] = l_highlight
        loss_dict['l_pix'] = l_pix
        loss_dict.update(shadow_sub_dict)
        loss_dict.update(highlight_sub_dict)

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 1.0)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        self.total_loss_accum += total_loss.item()
        self.shadow_loss_accum += shadow_total.item()
        self.highlight_loss_accum += l_highlight.item()
        self.pix_loss_accum += l_pix.item()
        self.period_step_count += 1

        if current_iter % 79 == 0 or current_iter == self.opt['train']['total_iter']:
            if self.period_step_count > 0:
                self.loss_history.append(self.total_loss_accum / self.period_step_count)
                self.shadow_loss_history.append(self.shadow_loss_accum / self.period_step_count)
                self.highlight_loss_history.append(self.highlight_loss_accum / self.period_step_count)
                self.pix_loss_history.append(self.pix_loss_accum / self.period_step_count)
                self.iter_history.append(current_iter)

                self.epoch_count += 1
                if self.epoch_count % 25 == 0 or current_iter == self.opt['train']['total_iter']:
                    self.plot_loss_curve(current_iter)

            self.total_loss_accum = 0.0
            self.shadow_loss_accum = 0.0
            self.highlight_loss_accum = 0.0
            self.pix_loss_accum = 0.0
            self.period_step_count = 0

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # 10. 推理、验证与保存：尺寸填充、EMA 推理、结果导出和断点保存
    def pad_test(self, window_size):
        """ 作用：测试时对高宽非Transformer整倍数边界进行自适应镜像填充包裹，前向完还原逆裁剪 """
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()

        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size

        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        highlight_mask = F.pad(
            self.highlight_mask, (0, mod_pad_w, 0, mod_pad_h), 'reflect'
        )
        self.nonpad_test(img, highlight_mask)

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None, highlight_mask=None):
        """ 作用：按统一反光阻断规则执行无填充或已填充推理，有训练状态时优先使用EMA影子网络 """
        if img is None:
            img = self.lq
        with torch.no_grad():
            binary_highlight_masks, exclusive_highlight_masks = self._arbitrate_highlight_masks(highlight_mask)
            _, planned_inpaint_masks = self._prepare_all_highlight_repair_tasks(
                binary_highlight_masks, exclusive_highlight_masks
            )
            highlight_block_masks = self._synthesize_highlight_block_masks(
                binary_highlight_masks, planned_inpaint_masks
            )

        active_net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        active_net.eval()
        with torch.no_grad():
            pred = self._forward_target_light_outputs(
                active_net, img, highlight_block_masks
            )
        if isinstance(pred, list):
            pred = pred[-1]
        self.output = pred

        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, *args, **kwargs):
        """ 作用：多卡分布式 environment 下的验证 logic 安全中转，限制仅在 0 号主进程上触发推理评估 """
        if os.environ.get('LOCAL_RANK', '0') == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, *args, **kwargs)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, *args, **kwargs):
        """ 作用：单卡非分布式全量指标验证核心存存根，原生框架预留位置，当前返回默认初始分值 0.0 """
        return 0.

    def get_current_visuals(self):
        """ 作用：收集捕获并打包传出当前Batch处于张量状态下的输入图、结果图与目标GT图像对 """
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        """ 作用：持久化存盘，导出网络 and 优化器当前的状态权重文件(.pth)，用以支撑任意时间点断点续训 """
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
