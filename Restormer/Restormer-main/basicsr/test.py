import os
import glob
import importlib.util
import re
import sys
from types import ModuleType
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage

# ==========================================
# 🛠️ 用户核心参数配置区域
# ==========================================
MODEL_PATH = '/home/cgz/lzc/Restormer/Restormer-main/experiments/FPC_18Channel_Restormer_Project/models/net_g_20000.pth'
IMAGE_DIR = '/home/cgz/lzc/FPC_Dataset/test'
NPY_DIR = '/home/cgz/lzc/FPC_Dataset/2026FPC_Selected_420_enhanced_3d'
OUTPUT_DIR = '/home/cgz/lzc/FPC_Dataset/restored_true_0.75prior'

# 特征增强 .npy 的动态调权滑块系数（1.0代表全量注入，0.0代表关闭先验）
PRIOR_SCALE = 0.75
HIGHLIGHT_GROUP_DISTANCE = 2.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 📐 导入网络拓扑结构 (Restormer 18ch)
# ==========================================
def _load_restormer_without_model_registry():
    """ 作用：绕过basicsr.models自动扫描，仅加载推理所需Restormer结构，避免其他模型文件的语法错误阻断test """
    import basicsr

    basicsr_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(basicsr_dir, 'models')
    arch_file = os.path.join(models_dir, 'archs', 'restormer_arch.py')
    if not os.path.isfile(arch_file):
        raise FileNotFoundError(f"找不到Restormer网络结构文件: {arch_file}")

    models_package = ModuleType('basicsr.models')
    models_package.__file__ = os.path.join(models_dir, '__init__.py')
    models_package.__path__ = [models_dir]
    models_package.__package__ = 'basicsr.models'
    sys.modules['basicsr.models'] = models_package
    basicsr.models = models_package

    archs_package = ModuleType('basicsr.models.archs')
    archs_package.__path__ = [os.path.join(models_dir, 'archs')]
    archs_package.__package__ = 'basicsr.models.archs'
    sys.modules['basicsr.models.archs'] = archs_package
    models_package.archs = archs_package

    module_name = 'basicsr.models.archs.restormer_arch'
    spec = importlib.util.spec_from_file_location(module_name, arch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法建立Restormer网络结构加载器: {arch_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Restormer


Restormer = _load_restormer_without_model_registry()


def _compute_perfect_mask(lq_coax, lq_bar, lq_ring):
    """ 作用：提取高鲁棒性前景板区掩膜，配合OpenCV轮廓填充填平板内假孔洞 """
    srcs = [img.mean(axis=2) for img in [lq_coax, lq_bar, lq_ring]]
    all_white = (srcs[0] == 1.0) & (srcs[1] == 1.0) & (srcs[2] == 1.0)
    bg_mask = (all_white * 255).astype(np.uint8)
    fg = 255 - bg_mask

    contours, _ = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    perfect_fg = np.zeros_like(fg)
    cv2.drawContours(perfect_fg, contours, -1, 255, thickness=-1)
    return perfect_fg


def _load_training_highlight_detector():
    """ 作用：直接加载训练数据集的反光检测器，使导出阶段与训练阶段始终共用同一套掩膜逻辑 """
    import basicsr

    basicsr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(basicsr_dir, 'data')
    dataset_file = os.path.join(data_dir, 'paired_image_dataset.py')
    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"找不到训练数据处理文件: {dataset_file}")

    if 'basicsr.data' not in sys.modules:
        data_package = ModuleType('basicsr.data')
        data_package.__file__ = os.path.join(data_dir, '__init__.py')
        data_package.__path__ = [data_dir]
        data_package.__package__ = 'basicsr.data'
        sys.modules['basicsr.data'] = data_package
        basicsr.data = data_package

    module_name = 'basicsr.data.paired_image_dataset'
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, dataset_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法建立训练数据处理文件加载器: {dataset_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    dataset_class = getattr(module, 'Dataset_FPC_18Channel', None)
    if dataset_class is None or not hasattr(dataset_class, '_compute_global_highlight'):
        raise AttributeError('训练数据处理文件中缺少Dataset_FPC_18Channel._compute_global_highlight。')
    return dataset_class.__new__(dataset_class)


HIGHLIGHT_DETECTOR = _load_training_highlight_detector()


def convert_color_space(img, to_rgb=True):
    """ 作用：封装图像色彩空间转换（BGR与RGB互转），保证推断期与训练数据集通道顺序绝对一致 """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if to_rgb else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def load_fpc_model(model_path):
    """ 作用：构建基础拓扑并挂载注入对接函数 """
    model = Restormer(
        inp_channels=18, out_channels=9,
        dim=48, num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias'
    )

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('params_ema', checkpoint.get('state_dict', checkpoint.get('params', checkpoint)))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model


def build_fpc_input_tensor(imgs_combined, priors_combined, highlight_mask, board_mask, prior_scale):
    """ 作用：全局先验预注入与解耦封装。在送入切块推理引擎前，全局拦截高光区域与板区背景先验并注入调权系数，完成完美的 18 通道张量拼装 """
    imgs = imgs_combined.unsqueeze(0)  # (1, 9, H, W)
    priors = priors_combined.unsqueeze(0)  # (1, 9, H, W)

    # 将 3 通道的高光掩膜扩展为 9 通道遮罩
    high_mask_9ch = torch.repeat_interleave(highlight_mask, 3, dim=1)

    # 提前完成板区背景与高光区域先验特征的双重归零拦截，并注入全局权重系数
    priors_processed = priors * board_mask * (1.0 - high_mask_9ch) * prior_scale

    # 组装完整的 18 通道输入矩阵
    return torch.cat([imgs, priors_processed], dim=1)


def _arbitrate_highlight_masks(highlight_mask):
    """ 作用：仲裁三路二值反光掩膜，返回原始反光与单光源独占反光列表 """
    binary_bool = [highlight_mask[:, i:i + 1] > 0.5 for i in range(3)]
    highlight_count = torch.stack(binary_bool, dim=0).sum(dim=0)
    binary_masks = [mask.to(highlight_mask.dtype) for mask in binary_bool]
    return binary_masks, [
        (mask & (highlight_count == 1)).to(highlight_mask.dtype) for mask in binary_bool
    ]


def _find_nearby_highlight_component_edges(labels, group_distance):
    """ 作用：复现训练端局部扫描，返回指定欧氏距离内反光连通块的最近连接端点 """
    group_distance = max(0.0, float(group_distance))
    radius = int(np.ceil(group_distance))
    foreground_y, foreground_x = np.nonzero(labels)
    if group_distance <= 0.0 or foreground_y.size == 0:
        return []
    image_height, image_width = labels.shape
    y0, y1 = max(0, int(foreground_y.min()) - radius), min(
        image_height, int(foreground_y.max()) + radius + 1
    )
    x0, x1 = max(0, int(foreground_x.min()) - radius), min(
        image_width, int(foreground_x.max()) + radius + 1
    )
    local_labels = labels[y0:y1, x0:x1]
    local_height, local_width = local_labels.shape
    component_count = int(labels.max())
    offsets = sorted([
        (dy * dy + dx * dx, dy, dx) for dy in range(radius + 1)
        for dx in range(-radius, radius + 1)
        if (dy > 0 or dx > 0) and dy * dy + dx * dx <= group_distance ** 2
    ])
    nearest_edges = {}
    for _, dy, dx in offsets:
        source_y0, source_y1 = max(0, -dy), min(local_height, local_height - dy)
        source_x0, source_x1 = max(0, -dx), min(local_width, local_width - dx)
        if source_y0 >= source_y1 or source_x0 >= source_x1:
            continue
        source = local_labels[source_y0:source_y1, source_x0:source_x1]
        target = local_labels[source_y0 + dy:source_y1 + dy, source_x0 + dx:source_x1 + dx]
        match_y, match_x = np.nonzero((source > 0) & (target > 0) & (source != target))
        if match_y.size == 0:
            continue
        source_ids, target_ids = source[match_y, match_x] - 1, target[match_y, match_x] - 1
        first_ids, second_ids = np.minimum(source_ids, target_ids), np.maximum(source_ids, target_ids)
        _, first_matches = np.unique(
            first_ids.astype(np.int64) * component_count + second_ids, return_index=True
        )
        for match_index in first_matches:
            first, second = int(first_ids[match_index]), int(second_ids[match_index])
            if (first, second) in nearest_edges:
                continue
            source_point = np.asarray([
                source_y0 + match_y[match_index] + y0,
                source_x0 + match_x[match_index] + x0
            ], dtype=np.int32)
            target_point = source_point + np.asarray([dy, dx], dtype=np.int32)
            start, end = (source_point, target_point) if int(source_ids[match_index]) == first \
                else (target_point, source_point)
            nearest_edges[(first, second)] = (first, second, start, end)
    return list(nearest_edges.values())


def _build_grouped_highlight_envelopes(mask, group_distance):
    """ 作用：全图标记反光块后按聚合组局部边界框连线并填孔，避免每组重复扫描整图 """
    grouped_envelopes = []
    mask_cpu = mask.detach().to(device='cpu', dtype=torch.uint8).numpy()
    for batch_index in range(mask.size(0)):
        labels, count = ndimage.label(
            mask_cpu[batch_index, 0], structure=np.ones((3, 3), dtype=np.uint8)
        )
        if count == 0:
            continue
        component_slices = ndimage.find_objects(labels)
        parents = list(range(count))

        def _find(index):
            """ 作用：查找并压缩当前反光连通块所属的聚合根节点 """
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        edges = _find_nearby_highlight_component_edges(labels, group_distance)
        for first, second, _, _ in edges:
            root_first, root_second = _find(first), _find(second)
            if root_first != root_second:
                parents[root_second] = root_first
        groups = {}
        for index in range(count):
            groups.setdefault(_find(index), []).append(index)
        for members in groups.values():
            member_slices = [component_slices[index] for index in members]
            y0 = min(component_slice[0].start for component_slice in member_slices)
            y1 = max(component_slice[0].stop for component_slice in member_slices)
            x0 = min(component_slice[1].start for component_slice in member_slices)
            x1 = max(component_slice[1].stop for component_slice in member_slices)
            local_labels = labels[y0:y1, x0:x1]
            tight_envelope = np.isin(
                local_labels, np.asarray(members, dtype=np.int32) + 1
            )
            member_set = set(members)
            for first, second, start, end in edges:
                if first not in member_set or second not in member_set:
                    continue
                steps = int(max(abs(int(end[0]) - int(start[0])),
                                abs(int(end[1]) - int(start[1]))))
                ys = np.rint(np.linspace(start[0], end[0], steps + 1)).astype(np.int32) - y0
                xs = np.rint(np.linspace(start[1], end[1], steps + 1)).astype(np.int32) - x0
                tight_envelope[ys, xs] = True
                for point in range(1, len(ys)):
                    if ys[point] != ys[point - 1] and xs[point] != xs[point - 1]:
                        tight_envelope[ys[point - 1], xs[point]] = True
            tight_envelope = ndimage.binary_fill_holes(tight_envelope)
            local_envelope = torch.from_numpy(tight_envelope).to(
                device=mask.device, dtype=mask.dtype
            ).unsqueeze(0).unsqueeze(0)
            core_pixel_count = int(np.count_nonzero(
                (mask_cpu[batch_index, 0, y0:y1, x0:x1] > 0) & tight_envelope
            ))
            grouped_envelopes.append((
                batch_index, (int(y0), int(y1), int(x0), int(x1)),
                local_envelope, core_pixel_count
            ))
    return grouped_envelopes


def _dilate_four_neighborhood(mask, radius):
    """ 作用：严格按上下左右四邻域扩展计划修复区，不引入对角线捷径 """
    expanded = mask > 0.5
    for _ in range(radius):
        next_mask = expanded.clone()
        next_mask[:, :, 1:, :] |= expanded[:, :, :-1, :]
        next_mask[:, :, :-1, :] |= expanded[:, :, 1:, :]
        next_mask[:, :, :, 1:] |= expanded[:, :, :, :-1]
        next_mask[:, :, :, :-1] |= expanded[:, :, :, 1:]
        expanded = next_mask
    return expanded.to(mask.dtype)


def _prepare_all_highlight_repair_tasks(binary_masks, exclusive_masks, group_distance):
    """ 作用：复现训练端反光任务几何，推理时仅返回目标专用前向所需的三路计划修复掩膜 """
    _, _, image_height, image_width = exclusive_masks[0].shape
    highlight_count = torch.stack(
        [(mask > 0.5).to(mask.dtype) for mask in binary_masks], dim=0
    ).sum(dim=0)
    multi_highlight_mask = (highlight_count >= 2.0).to(exclusive_masks[0].dtype)
    planned_masks = []
    for exclusive_mask in exclusive_masks:
        planned_mask = torch.zeros_like(exclusive_mask)
        for batch_index, (y0, y1, x0, x1), envelope, core_count in \
                _build_grouped_highlight_envelopes(exclusive_mask, group_distance):
            roi_y0, roi_y1 = max(0, y0 - 20), min(image_height, y1 + 20)
            roi_x0, roi_x1 = max(0, x0 - 20), min(image_width, x1 + 20)
            envelope_roi = exclusive_mask.new_zeros(
                (1, 1, roi_y1 - roi_y0, roi_x1 - roi_x0)
            )
            envelope_roi[:, :, y0 - roi_y0:y1 - roi_y0, x0 - roi_x0:x1 - roi_x0] = envelope
            repair_roi = envelope_roi if core_count < 20 else \
                _dilate_four_neighborhood(envelope_roi, 10)
            planned_mask[batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1] = \
                torch.maximum(
                    planned_mask[batch_index:batch_index + 1, :, roi_y0:roi_y1, roi_x0:roi_x1],
                    repair_roi
                )
        planned_masks.append(planned_mask * (1.0 - multi_highlight_mask))
    return planned_masks


def _synthesize_highlight_block_masks(highlight_mask, group_distance=HIGHLIGHT_GROUP_DISTANCE):
    """ 作用：在完整大图上合成与训练一致的“反光核心＋计划修复区”三路统一阻断掩膜 """
    binary_masks, exclusive_masks = _arbitrate_highlight_masks(highlight_mask)
    planned_masks = _prepare_all_highlight_repair_tasks(
        binary_masks, exclusive_masks, group_distance
    )
    return torch.cat([
        torch.maximum(binary_masks[i], planned_masks[i]) for i in range(3)
    ], dim=1)


def _build_target_light_input(model_input, highlight_block_masks, target_index):
    """ 作用：保留目标光源自身RGB，并在完整阻断区内用剩余非反光光源替换外来反光RGB """
    light_rgbs = [model_input[:, i * 3:(i + 1) * 3] for i in range(3)]
    routed_rgbs = []
    for source_index in range(3):
        if source_index == target_index:
            routed_rgbs.append(light_rgbs[source_index])
            continue
        replacement_index = 3 - target_index - source_index
        replace_mask = (highlight_block_masks[:, source_index:source_index + 1] > 0.5) & \
            (highlight_block_masks[:, replacement_index:replacement_index + 1] < 0.5)
        routed_rgbs.append(torch.where(
            replace_mask, light_rgbs[replacement_index], light_rgbs[source_index]
        ))
    return torch.cat(routed_rgbs + [model_input[:, 9:]], dim=1)


def _forward_target_light_outputs(model, model_input, highlight_block_masks):
    """ 作用：有反光时顺序执行三次目标专用前向并立即仅保留对应3通道，无反光时单次前向 """
    if not torch.any(highlight_block_masks > 0.5).item():
        output = model(model_input)
        return output[-1] if isinstance(output, list) else output
    selected_outputs = []
    for target_index in range(3):
        output = model(_build_target_light_input(
            model_input, highlight_block_masks, target_index
        ))
        output = output[-1] if isinstance(output, list) else output
        selected_outputs.append(output[:, target_index * 3:(target_index + 1) * 3].clone())
        del output
    return torch.cat(selected_outputs, dim=1)


def _blend_with_white_background(output_tensor, perfect_mask_tensor):
    """ 作用：完整保留板区内的九通道模型输出，并将板区外融合为纯白背景，消除背景色偏与噪点 """
    return output_tensor * perfect_mask_tensor + 1.0 * (1.0 - perfect_mask_tensor)


def _generate_tile_starts(length, tile_size, overlap):
    """ 作用：按固定步长生成单方向分块起点，当前分块覆盖图像末端后立即停止，避免边缘分块前移、重复或超大重叠 """
    stride = tile_size - overlap
    if tile_size <= 0 or overlap < 0 or stride <= 0:
        raise ValueError("tile_size必须大于0，且overlap必须满足0 <= overlap < tile_size")

    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + tile_size >= length:
            break
        start += stride
    return starts


def tile_inference(model, x, highlight_block_masks, tile_size=1024, overlap=128):
    """ 作用：同步切分输入与全图反光阻断掩膜，执行目标专用分块前向并羽化融合 """
    b, _, h, w = x.size()
    if tuple(highlight_block_masks.shape) != (b, 3, h, w):
        raise ValueError(f"反光阻断掩膜必须为{(b, 3, h, w)}，实际为{tuple(highlight_block_masks.shape)}")
    device = x.device

    out = torch.zeros((b, 9, h, w), device=device)
    weight = torch.zeros((b, 9, h, w), device=device)
    y_starts = _generate_tile_starts(h, tile_size, overlap)
    x_starts = _generate_tile_starts(w, tile_size, overlap)

    for y in y_starts:
        for wx in x_starts:
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = wx, min(wx + tile_size, w)

            crop = x[:, :, y1:y2, x1:x2]
            block_crop = highlight_block_masks[:, :, y1:y2, x1:x2]
            ch, cw = crop.shape[-2:]

            ph, pw = (16 - ch % 16) % 16, (16 - cw % 16) % 16
            if ph > 0 or pw > 0:
                crop = F.pad(crop, (0, pw, 0, ph), 'reflect')
                block_crop = F.pad(block_crop, (0, pw, 0, ph), 'reflect')

            with torch.no_grad():
                raw_preds = _forward_target_light_outputs(model, crop, block_crop)
                out_crop = raw_preds[:, :, :ch, :cw]

            mask = torch.ones((1, 1, ch, cw), device=device)
            if overlap > 0:
                ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, 3.1415926535, overlap, device=device))
                if y1 > 0: mask[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)
                if y2 < h: mask[:, :, -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
                if x1 > 0: mask[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)
                if x2 < w: mask[:, :, :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)

            out[:, :, y1:y2, x1:x2] += out_crop * mask
            weight[:, :, y1:y2, x1:x2] += mask

            del crop, block_crop, out_crop, mask
        torch.cuda.empty_cache()

    return out / (weight + 1e-8)


def process_single_fpc_group(img_id, model):
    """ 作用：核心物理流：18通道组装并执行模型恢复与色彩空间转换，确保输入输出符合框架规范 """
    print(f"🚀 正在处理 FPC 样本组: {img_id} ...")
    ext = '.png' if os.path.exists(os.path.join(IMAGE_DIR, f"{img_id}_coax.png")) else '.jpg'

    path_coax = os.path.join(IMAGE_DIR, f"{img_id}_coax{ext}")
    path_bar = os.path.join(IMAGE_DIR, f"{img_id}_bar{ext}")
    path_ring = os.path.join(IMAGE_DIR, f"{img_id}_ring{ext}")

    if not (os.path.exists(path_coax) and os.path.exists(path_bar) and os.path.exists(path_ring)):
        print(f"⚠️ 警告: 样本组 {img_id} 的 3光源图片不完整，已跳过。")
        return

    img_coax = convert_color_space(cv2.imread(path_coax)).astype(np.float32) / 255.0
    img_bar = convert_color_space(cv2.imread(path_bar)).astype(np.float32) / 255.0
    img_ring = convert_color_space(cv2.imread(path_ring)).astype(np.float32) / 255.0

    npy_coax_path = os.path.join(NPY_DIR, f"{img_id}_coax_enhanced_3d.npy")
    npy_bar_path = os.path.join(NPY_DIR, f"{img_id}_bar_enhanced_3d.npy")
    npy_ring_path = os.path.join(NPY_DIR, f"{img_id}_ring_enhanced_3d.npy")

    if not (os.path.exists(npy_coax_path) and os.path.exists(npy_bar_path) and os.path.exists(npy_ring_path)):
        print(f"⚠️ 警告: 样本组 {img_id} 的 3路先验 .npy 矩阵不完整，已跳过。")
        return

    npy_coax = np.load(npy_coax_path).astype(np.float32)
    npy_bar = np.load(npy_bar_path).astype(np.float32)
    npy_ring = np.load(npy_ring_path).astype(np.float32)

    perfect_mask = _compute_perfect_mask(img_coax, img_bar, img_ring)
    highlight_mask_np = HIGHLIGHT_DETECTOR._compute_global_highlight(
        img_coax, img_bar, img_ring
    )
    if highlight_mask_np.shape != (*img_coax.shape[:2], 3):
        raise ValueError(f"训练反光检测器返回了错误形状: {highlight_mask_np.shape}")

    # 统一转换并传输所有 Tensor 到 GPU 上
    global_highlight_mask = torch.from_numpy(
        np.ascontiguousarray(np.moveaxis(highlight_mask_np, -1, 0))
    ).unsqueeze(0).to(DEVICE)
    highlight_block_masks = _synthesize_highlight_block_masks(global_highlight_mask)
    perfect_mask_tensor = torch.from_numpy(perfect_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

    t_img_coax = torch.from_numpy(img_coax).permute(2, 0, 1)
    t_img_bar = torch.from_numpy(img_bar).permute(2, 0, 1)
    t_img_ring = torch.from_numpy(img_ring).permute(2, 0, 1)

    t_npy_coax = torch.from_numpy(npy_coax).permute(2, 0, 1)
    t_npy_bar = torch.from_numpy(npy_bar).permute(2, 0, 1)
    t_npy_ring = torch.from_numpy(npy_ring).permute(2, 0, 1)

    imgs_combined = torch.cat([t_img_coax, t_img_bar, t_img_ring], dim=0).to(DEVICE)
    priors_combined = torch.cat([t_npy_coax, t_npy_bar, t_npy_ring], dim=0).to(DEVICE)

    # 传入 perfect_mask_tensor 替代腐蚀掩膜，严格对齐训练端未腐蚀完整掩膜归零逻辑
    input_tensor = build_fpc_input_tensor(imgs_combined, priors_combined, global_highlight_mask, perfect_mask_tensor,
                                          PRIOR_SCALE)

    # 送入高效全 GPU 切块推理引擎
    output_tensor = tile_inference(
        model, input_tensor, highlight_block_masks, tile_size=1024, overlap=128
    )

    # 执行全 GPU 白色背景融合，板区内完整保留九通道模型输出
    output_tensor = _blend_with_white_background(output_tensor, perfect_mask_tensor)

    # 统一在准备保存文件前传输回 CPU 并转为 NumPy 数组
    output_tensor = output_tensor.squeeze(0).clamp(0.0, 1.0)

    out_coax_rgb = output_tensor[0:3, :, :].permute(1, 2, 0).cpu().numpy() * 255.0
    out_bar_rgb = output_tensor[3:6, :, :].permute(1, 2, 0).cpu().numpy() * 255.0
    out_ring_rgb = output_tensor[6:9, :, :].permute(1, 2, 0).cpu().numpy() * 255.0

    out_coax = convert_color_space(out_coax_rgb.astype(np.uint8), to_rgb=False)
    out_bar = convert_color_space(out_bar_rgb.astype(np.uint8), to_rgb=False)
    out_ring = convert_color_space(out_ring_rgb.astype(np.uint8), to_rgb=False)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_id}_coax_restored{ext}"), out_coax)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_id}_bar_restored{ext}"), out_bar)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_id}_ring_restored{ext}"), out_ring)
    print(f"✨ 成功导出样本组 {img_id} 的 3光源高清复原图！")


def main():
    """ 作用：主程序自动化寻址控制流，初始化路径、加载模型并循环调度复原任务 """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("====== ⚙️ FPC 18通道自适应恢复推理程序对接启动 ======")
    print(f"• 模型权重路径: {MODEL_PATH}")
    print(f"• 当前先验滑块调权系数 (PRIOR_SCALE): {PRIOR_SCALE}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 找不到预训练权重文件: {MODEL_PATH}")

    model = load_fpc_model(MODEL_PATH)

    coax_files = glob.glob(os.path.join(IMAGE_DIR, "*_coax.*"))
    coax_files = [f for f in coax_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not coax_files:
        print(f"❌ 错误: 在路径 {IMAGE_DIR} 中没有找到任何符合格式的图片！")
        return

    img_ids = []
    for f in coax_files:
        base_name = os.path.basename(f)
        match = re.match(r"(.+)_coax\.[a-zA-Z]+$", base_name)
        if match:
            img_ids.append(match.group(1))

    img_ids = sorted(list(set(img_ids)))
    print(f"• 共成功检索到 {len(img_ids)} 组完整的待修复 FPC 图像序列。\n")

    for img_id in img_ids:
        try:
            process_single_fpc_group(img_id, model)
        except Exception as e:
            print(f"💥 样本组 {img_id} 推理中突发异常: {str(e)}，已跳过以保护后续队列。")

    print(f"\n🎉 全部样本组处理完毕！所有复原好的高清图片已安全导出至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()