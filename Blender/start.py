import bpy
import os
import random
import math
import re
import numpy as np

# ==============================================================================
# 🎛️ 【前台黄金配置面板】（三种光源区间完全独立，像素自适应追踪，绝不锁死尺寸）
# ==============================================================================
# 📂 1. 工业级数据集输入/输出绝对路径总闸
INPUT_DIR = "/home/cgz/lzc/Blender/new"
OUTPUT_DIR = "/home/cgz/lzc/Blender/finish"

# 📏 2. 公共几何参数大闸（保证三种光源在一组内形变、高度、走势 100% 绝对相同）
SUBDIVISION_CUTS = 300                 # 网格精细度（300刀，配合NumPy加速，秒出结果）
CREASE_DEPTH_RANGE = (0.4, 0.8)         # 折痕突起高度的随机范围
CURVATURE_RANGE = (0.01, 0.05)           # 折痕自身脊线的轻微弯曲范围

# 🔴 3. 【同轴光 (Coaxial) 专属物理衰减控制区】
COAX_GLOBAL_BLUR_RANGE = (0, 0.007)     # 纹理模糊全局极限范围
COAX_SHADOW_BRIGHTNESS_RANGE = (0.01, 0.9) # 线性亮度保留倍率
COAX_SHADOW_EXPOSURE_RANGE = (-15, 0)   # 曝光级数裁剪（单位：EV/档）
COAX_HIGHLIGHT_ENABLE = True                # 是否启用折痕高光功能
COAX_HIGHLIGHT_PROBABILITY = 0.5            # 产生高光的几率
COAX_HIGHLIGHT_BRIGHTNESS_RANGE = (1, 5)   # 高光叠加亮度的随机范围
COAX_HIGHLIGHT_LINE_TIGHTNESS_RANGE = (25000.0, 40000) # 高光线条绝对粗细范围
COAX_HIGHLIGHT_LINE_RAND_FREQ_RANGE = (1, 6)     # 控制曝光点的长短跨度频率
COAX_HIGHLIGHT_LINE_RAND_PHASE_RANGE = (0, 30.0)   # 控制曝光点走向在空间上的随机偏移相位

# 🟡 4. 【条形光 (Bar) 专属物理衰减控制区】
BAR_GLOBAL_BLUR_RANGE = (0, 0.007)      # 纹理模糊全局极限范围
BAR_SHADOW_BRIGHTNESS_RANGE = (0.01, 0.8)  # 线性亮度保留倍率
BAR_SHADOW_EXPOSURE_RANGE = (-15, 0)    # 曝光级数裁剪（单位：EV/档）
BAR_HIGHLIGHT_ENABLE = True                 # 是否启用折痕高光功能
BAR_HIGHLIGHT_PROBABILITY = 0.8             # 产生高光的几率
BAR_HIGHLIGHT_BRIGHTNESS_RANGE = (1, 5)   # 高光叠加亮度的随机范围
BAR_HIGHLIGHT_LINE_TIGHTNESS_RANGE = (15000, 40000) # 高光反射线绝对粗细范围
BAR_HIGHLIGHT_LINE_RAND_FREQ_RANGE = (1, 6)     # 控制曝光点的长短跨度频率
BAR_HIGHLIGHT_LINE_RAND_PHASE_RANGE = (0, 30)  # 控制曝光点走向在空间上的随机偏移相位

# 🔵 5. 【环形光 (Ring) 专属物理衰减控制区】
RING_GLOBAL_BLUR_RANGE = (0, 0.007)     # 纹理模糊全局极限范围
RING_SHADOW_BRIGHTNESS_RANGE = (0.01, 0.7) # 线性亮度保留倍率
RING_SHADOW_EXPOSURE_RANGE = (-15, 0)    # 曝光级数裁剪（单位：EV/档）
RING_HIGHLIGHT_ENABLE = True                # 是否启用折痕高光功能
RING_HIGHLIGHT_PROBABILITY = 1.0            # 每次运行产生高光的几率
RING_HIGHLIGHT_BRIGHTNESS_RANGE = (2, 5)   # HDR 核心爆白能量增益
RING_HIGHLIGHT_LINE_TIGHTNESS_RANGE = (15000, 40000) # 高光反射线绝对粗细范围
RING_HIGHLIGHT_LINE_RAND_FREQ_RANGE = (1, 6)     # 控制曝光点的长短跨度频率
RING_HIGHLIGHT_LINE_RAND_PHASE_RANGE = (0, 30)   # 控制曝光点走向在空间上的随机偏移相位


# ==============================================================================
# 🚀 核心逻辑总线处理区（全自动批量、对齐渲染解耦内核）
# ==============================================================================
def render_single_source(gid, light_type, creases_params, img_file_name, width, height, aspect_ratio):
    """
    🚚 专属各向同性数乘级联物理减光渲染总线
    """
    # 为当前光源类型定制专属的哈希盐值种子，彻底切断三种光源在连续抽签时的数理相关性
    local_rand = random.Random(f"{gid}_{light_type}")

    if light_type == 'coax':
        g_blur = COAX_GLOBAL_BLUR_RANGE
        s_bright = COAX_SHADOW_BRIGHTNESS_RANGE
        s_exposure = COAX_SHADOW_EXPOSURE_RANGE
        h_enable = COAX_HIGHLIGHT_ENABLE
        h_prob = COAX_HIGHLIGHT_PROBABILITY
        h_bright = COAX_HIGHLIGHT_BRIGHTNESS_RANGE
        h_tightness = COAX_HIGHLIGHT_LINE_TIGHTNESS_RANGE
        h_rand_freq_range = COAX_HIGHLIGHT_LINE_RAND_FREQ_RANGE
        h_rand_phase_range = COAX_HIGHLIGHT_LINE_RAND_PHASE_RANGE
    elif light_type == 'bar':
        g_blur = BAR_GLOBAL_BLUR_RANGE
        s_bright = BAR_SHADOW_BRIGHTNESS_RANGE
        s_exposure = BAR_SHADOW_EXPOSURE_RANGE
        h_enable = BAR_HIGHLIGHT_ENABLE
        h_prob = BAR_HIGHLIGHT_PROBABILITY
        h_bright = BAR_HIGHLIGHT_BRIGHTNESS_RANGE
        h_tightness = BAR_HIGHLIGHT_LINE_TIGHTNESS_RANGE
        h_rand_freq_range = BAR_HIGHLIGHT_LINE_RAND_FREQ_RANGE
        h_rand_phase_range = BAR_HIGHLIGHT_LINE_RAND_PHASE_RANGE
    else:  # ring
        g_blur = RING_GLOBAL_BLUR_RANGE
        s_bright = RING_SHADOW_BRIGHTNESS_RANGE
        s_exposure = RING_SHADOW_EXPOSURE_RANGE
        h_enable = RING_HIGHLIGHT_ENABLE
        h_prob = RING_HIGHLIGHT_PROBABILITY
        h_bright = RING_HIGHLIGHT_BRIGHTNESS_RANGE
        h_tightness = RING_HIGHLIGHT_LINE_TIGHTNESS_RANGE
        h_rand_freq_range = RING_HIGHLIGHT_LINE_RAND_FREQ_RANGE
        h_rand_phase_range = RING_HIGHLIGHT_LINE_RAND_PHASE_RANGE

    # 🔒 严密清理上一轮网格堆积
    for obj in list(bpy.data.objects): 
        if obj.name != "Ortho_Camera":
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes): bpy.data.meshes.remove(mesh, do_unlink=True)
    
    img_full_path = os.path.join(INPUT_DIR, img_file_name)
    loaded_img = bpy.data.images.load(img_full_path)
    
    bpy.ops.mesh.primitive_plane_add(size=2)
    obj = bpy.context.object
    obj.name = "FPC_Board"
    obj.scale = (aspect_ratio, 1.0, 1.0)
    bpy.ops.object.transform_apply(scale=True, location=False, rotation=False)
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=SUBDIVISION_CUTS)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    mesh = obj.data
    num_verts = len(mesh.vertices)
    coords = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape((num_verts, 3))
    
    x, y = coords[:, 0], coords[:, 1]
    max_z = np.full(num_verts, -np.inf)
    shadow_mask = np.zeros(num_verts, dtype=np.float32)
    highlight_envelope = np.zeros(num_verts, dtype=np.float32)
    gauss_base = np.zeros(num_verts, dtype=np.float32)
    
    # 几何高度起伏计算
    for cp in creases_params:
        t = -math.sin(cp['theta']) * (x - cp['x0']) + math.cos(cp['theta']) * (y - cp['y0'])
        wave = cp['amplitude'] * np.sin(cp['frequency'] * t + cp['phase'])
        dist = math.cos(cp['theta']) * (x - cp['x0']) + math.sin(cp['theta']) * (y - cp['y0']) + wave
        
        z_contrib = cp['depth'] * np.exp(-4.5 * (dist / cp['width']) ** 2)
        max_z = np.maximum(max_z, z_contrib)
        
    min_z = np.min(max_z)
    max_height_val = np.max(max_z - min_z)
    coords[:, 2] = max_z - min_z  
    mesh.vertices.foreach_set("co", coords.ravel())
    
    has_highlight = h_enable and (local_rand.random() < h_prob)
    chosen_bright_multiplier = local_rand.uniform(*h_bright) if has_highlight else 0.0
    chosen_base_tightness = local_rand.uniform(*h_tightness)
    
    # 物理减光区域网格空间映射评估
    for cp in creases_params:
        t = -math.sin(cp['theta']) * (x - cp['x0']) + math.cos(cp['theta']) * (y - cp['y0'])
        wave = cp['amplitude'] * np.sin(cp['frequency'] * t + cp['phase'])
        dist = math.cos(cp['theta']) * (x - cp['x0']) + math.sin(cp['theta']) * (y - cp['y0']) + wave
        
        smaller_side_sign = 1.0 if np.sum(dist > 0) < np.sum(dist < 0) else -1.0
        side_dist = dist * smaller_side_sign
        
        chosen_blur_width = local_rand.uniform(0.35, 0.55)
        fade_factor = 0.88 + 0.12 * np.exp(-0.5 * (side_dist / chosen_blur_width) ** 2)
        
        decay_term = 0.90 + 0.10 * np.exp(-4.5 * (side_dist / chosen_blur_width))
        shadow_gate = np.where(side_dist > 0, (1.0 - np.exp(-100.0 * side_dist)) * decay_term, 0.0)
        crease_shadow = 1.0 * fade_factor * shadow_gate
        shadow_mask = np.maximum(shadow_mask, crease_shadow)
        
        if chosen_bright_multiplier > 0.0:
            new_rand_freq = local_rand.uniform(*h_rand_freq_range)
            new_rand_phase = local_rand.uniform(*h_rand_phase_range)
            
            f1 = new_rand_freq
            f2 = f1 * 2.37
            f3 = f1 * 5.11
            w1 = np.sin(f1 * t + new_rand_phase)
            w2 = np.cos(f2 * t + 2.15)
            w3 = np.sin(f3 * t + 4.73)
            combined_flux = (w1 + 0.55 * w2 + 0.35 * w3) / 1.9
            
            smooth_wave = np.clip(0.42 + 0.88 * combined_flux, 0.0, 1.0)
            smooth_wave = np.power(smooth_wave, 1.8)
            
            gaussian_wide = np.exp(- 1500.0 * dist ** 2)
            gauss_base = np.maximum(gauss_base, gaussian_wide)
            
            side_gate = np.where(side_dist <= 0, 1.0, np.exp(-180.0 * side_dist ** 2))
            crease_env = chosen_bright_multiplier * smooth_wave * side_gate
            highlight_envelope = np.maximum(highlight_envelope, crease_env)
            
    # 高度景深柔性模糊
    z_norm = coords[:, 2] / max_height_val if max_height_val > 1e-5 else np.ones(num_verts, dtype=np.float32)
    val1 = local_rand.uniform(*g_blur)
    val2 = local_rand.uniform(*g_blur)
    blur_mask = np.where(shadow_mask > 0.001, max(val1, val2) - (max(val1, val2) - min(val1, val2)) * z_norm, 0.0)
    
    if "FPC_Shadow" not in mesh.attributes: mesh.attributes.new(name="FPC_Shadow", type='FLOAT', domain='POINT')
    if "FPC_Blur" not in mesh.attributes: mesh.attributes.new(name="FPC_Blur", type='FLOAT', domain='POINT')
    if "FPC_Envelope" not in mesh.attributes: mesh.attributes.new(name="FPC_Envelope", type='FLOAT', domain='POINT')
    if "FPC_Gauss" not in mesh.attributes: mesh.attributes.new(name="FPC_Gauss", type='FLOAT', domain='POINT')
    
    mesh.attributes["FPC_Shadow"].data.foreach_set("value", shadow_mask)
    mesh.attributes["FPC_Blur"].data.foreach_set("value", blur_mask)
    mesh.attributes["FPC_Envelope"].data.foreach_set("value", highlight_envelope)
    mesh.attributes["FPC_Gauss"].data.foreach_set("value", gauss_base)
    
    mesh.update()
    for p in mesh.polygons: p.use_smooth = True
    bpy.ops.object.shade_smooth()

    # 3. 🎨 材质总线直通级联
    mat = bpy.data.materials.new(name="FPC_Perfect_Shadow_Mat")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 1.0
    
    tex_clean = nodes.new('ShaderNodeTexImage')
    tex_clean.image = loaded_img
    
    tex_blur = nodes.new('ShaderNodeTexImage')
    tex_blur.image = loaded_img
    tex_blur.interpolation = 'Linear'
    
    rgb_to_bw = nodes.new('ShaderNodeRGBToBW')
    links.new(tex_clean.outputs['Color'], rgb_to_bw.inputs['Color'])
    
    attr_shadow = nodes.new('ShaderNodeAttribute')
    attr_shadow.attribute_name = "FPC_Shadow"
    attr_blur = nodes.new('ShaderNodeAttribute')
    attr_blur.attribute_name = "FPC_Blur"
    attr_env = nodes.new('ShaderNodeAttribute')
    attr_env.attribute_name = "FPC_Envelope"
    attr_gauss = nodes.new('ShaderNodeAttribute')
    attr_gauss.attribute_name = "FPC_Gauss"
    
    # 专属 local_rand 抽取选定区间的亮度和曝光
    chosen_bright = local_rand.uniform(*s_bright)
    chosen_exposure = local_rand.uniform(*s_exposure)
    exposure_multiplier = math.pow(2.0, chosen_exposure)
    
    # 【一级线性亮度乘算】
    shadow_bright = nodes.new('ShaderNodeMix')
    shadow_bright.data_type = 'RGBA'
    shadow_bright.blend_type = 'MULTIPLY'
    shadow_bright.inputs['Factor'].default_value = 1.0
    shadow_bright.inputs['B'].default_value = (chosen_bright, chosen_bright, chosen_bright, 1.0)
    links.new(tex_blur.outputs['Color'], shadow_bright.inputs['A'])
    
    # 【二级指数曝光裁剪（Stops）】
    shadow_exposure = nodes.new('ShaderNodeMix')
    shadow_exposure.data_type = 'RGBA'
    shadow_exposure.blend_type = 'MULTIPLY'
    shadow_exposure.inputs['Factor'].default_value = 1.0
    shadow_exposure.inputs['B'].default_value = (exposure_multiplier, exposure_multiplier, exposure_multiplier, 1.0)
    links.new(shadow_bright.outputs['Result'], shadow_exposure.inputs['A'])
    
    # 切换总闸
    mix_color = nodes.new('ShaderNodeMix')
    mix_color.data_type = 'RGBA'
    mix_color.blend_type = 'MIX'
    links.new(tex_blur.outputs['Color'], mix_color.inputs['A'])               
    links.new(shadow_exposure.outputs['Result'], mix_color.inputs['B'])       
    links.new(attr_shadow.outputs['Fac'], mix_color.inputs['Factor'])         
    
    # 像素级着色器超频高光收缩
    math_power = nodes.new('ShaderNodeMath')
    math_power.operation = 'POWER'
    math_power.inputs[1].default_value = chosen_base_tightness / 1500.0
    links.new(attr_gauss.outputs['Fac'], math_power.inputs[0])
    
    math_high_base = nodes.new('ShaderNodeMath')
    math_high_base.operation = 'MULTIPLY'
    links.new(math_power.outputs['Value'], math_high_base.inputs[0])
    links.new(attr_env.outputs['Fac'], math_high_base.inputs[1])
    
    # 排线固有电路高频特征增益
    math_wire_gain = nodes.new('ShaderNodeMath')
    math_wire_gain.operation = 'MULTIPLY'
    math_wire_gain.inputs[1].default_value = 4.2            
    links.new(rgb_to_bw.outputs['Val'], math_wire_gain.inputs[0])
    
    math_wire_base = nodes.new('ShaderNodeMath')
    math_wire_base.operation = 'ADD'
    math_wire_base.inputs[1].default_value = 0.25           
    links.new(math_wire_gain.outputs['Value'], math_wire_base.inputs[0])
    
    math_high_intensity = nodes.new('ShaderNodeMath')
    math_high_intensity.operation = 'MULTIPLY'
    links.new(math_high_base.outputs['Value'], math_high_intensity.inputs[0])
    links.new(math_wire_base.outputs['Value'], math_high_intensity.inputs[1])
    
    mix_high = nodes.new('ShaderNodeMix')
    mix_high.data_type = 'RGBA'
    mix_high.blend_type = 'MIX'
    mix_high.inputs['B'].default_value = (1.0, 1.0, 1.0, 1.0)
    
    white_background_mix = nodes.new('ShaderNodeMix')
    white_background_mix.data_type = 'RGBA'
    white_background_mix.blend_type = 'MIX'
    white_background_mix.inputs['A'].default_value = (1.0, 1.0, 1.0, 1.0)
    
    alpha_threshold = nodes.new('ShaderNodeMath')
    alpha_threshold.operation = 'GREATER_THAN'
    alpha_threshold.inputs[1].default_value = 0.1
    
    # UV轴抖动
    tex_coord = nodes.new('ShaderNodeTexCoord')
    noise_blur = nodes.new('ShaderNodeTexNoise')
    noise_blur.inputs['Scale'].default_value = 8000.0  
    noise_blur.inputs['Detail'].default_value = 15.0
    
    mix_uv = nodes.new('ShaderNodeMix')
    mix_uv.data_type = 'RGBA'
    mix_uv.blend_type = 'LINEAR_LIGHT'
    
    links.new(tex_coord.outputs['UV'], mix_uv.inputs['A'])
    links.new(noise_blur.outputs['Color'], mix_uv.inputs['B'])
    links.new(attr_blur.outputs['Fac'], mix_uv.inputs['Factor']) 
    links.new(mix_uv.outputs['Result'], tex_blur.inputs['Vector']) 
    
    # 总线并联连通
    links.new(mix_color.outputs['Result'], mix_high.inputs['A'])
    links.new(math_high_intensity.outputs['Value'], mix_high.inputs['Factor']) 
    links.new(mix_high.outputs['Result'], white_background_mix.inputs['B'])
    links.new(tex_clean.outputs['Alpha'], alpha_threshold.inputs[0])
    links.new(alpha_threshold.outputs['Value'], white_background_mix.inputs['Factor'])
    
    links.new(white_background_mix.outputs['Result'], emission.inputs['Color']) 
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    if hasattr(mat, 'blend_method'): mat.blend_method = 'OPAQUE'
    if hasattr(mat, 'shadow_method'): mat.shadow_method = 'NONE'
    obj.data.materials.append(mat)
    
    # 🟢 📢 动态分辨率直通锁定：强制把当前图片的宽高绑定给镜头和物理输出端
    cam_obj = bpy.data.objects.get("Ortho_Camera")
    if not cam_obj:
        cam_data = bpy.data.cameras.new(name="Ortho_Camera")
        cam_obj = bpy.data.objects.new(name="Ortho_Camera", object_data=cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        
    cam_obj.location = (0.0, 0.0, max(5.0, max_height_val * 2.5))
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam_obj
    cam_obj.data.type = 'ORTHO'
    cam_obj.data.sensor_fit = 'HORIZONTAL'        
    cam_obj.data.ortho_scale = 2.0 * aspect_ratio  
    
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100
    
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, img_file_name)
    bpy.ops.render.render(write_still=True)
    
    bpy.data.images.remove(loaded_img)


def main_batch_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'
    bpy.context.scene.view_settings.exposure = 0.0
    bpy.context.scene.view_settings.gamma = 1.0
    
    all_files = sorted(os.listdir(INPUT_DIR))
    
    indexed_groups = {}
    for filename in all_files:
        if '_coax' in filename and filename.endswith('.png'):
            match = re.search(r'(\d+)', filename)
            if match:
                img_number = int(match.group(1))
                base_id = filename.split('_coax')[0]
                indexed_groups[img_number] = base_id
                
    sorted_numbers = sorted(list(indexed_groups.keys()))
    
    if len(sorted_numbers) == 0:
        print(f"❌ 警告：在指定路径 [{INPUT_DIR}] 中未检测到任何包含 '_coax' 命名的图片组！")
        return
        
    print(f"🚀 正式启动【图片数字编号严格升序逐个作业流水线】，共检索到 {len(sorted_numbers)} 组任务...")
    
    # 🟢 📢 彻底移除了最外层“一刀切”的预载尺寸逻辑，将其安全下沉到循环内部
    for count, img_num in enumerate(sorted_numbers):
        gid = indexed_groups[img_num]
        print(f"🎬 [{count+1}/{len(sorted_numbers)}] 流水线正在运行当前图片编号: #{img_num} [ID: {gid}]...")
        
        # 🟢 📢 【像素自适应追踪内核】：每处理一个编号，实时读取当前图片真正的物理高宽与 Alpha 通道
        current_img_name = f"{gid}_coax.png"
        current_img = bpy.data.images.load(os.path.join(INPUT_DIR, current_img_name))
        w_px, h_px = current_img.size[0], current_img.size[1] # 👈 动态抓取当前批次的分辨率
        aspect_ratio = w_px / h_px
        
        pixels = np.array(current_img.pixels)
        alpha_mask = pixels[3::4].reshape((h_px, w_px))
        valid_y, valid_x = np.where(alpha_mask > 0.05)
        bpy.data.images.remove(current_img) # 及时释放，杜绝溢出
        
        if len(valid_x) == 0:
            valid_x, valid_y = [w_px // 2], [h_px // 2]
            
        num_creases = 2 if random.random() < 0.20 else 1
        creases_params = []
        for _ in range(num_creases):
            theta = random.uniform(0, 2 * math.pi)          
            idx = random.randint(0, len(valid_x) - 1)
            x0 = ((valid_x[idx] / (w_px - 1)) * 2.0 - 1.0) * aspect_ratio
            y0 = (valid_y[idx] / (h_px - 1)) * 2.0 - 1.0
            
            amplitude = random.uniform(*CURVATURE_RANGE) 
            frequency = random.uniform(1.0, 1.5)                    
            phase = random.uniform(0, 2 * math.pi)                  
            depth = random.uniform(*CREASE_DEPTH_RANGE)
            
            creases_params.append({
                'theta': theta, 'x0': x0, 'y0': y0, 
                'amplitude': amplitude, 'frequency': frequency, 'phase': phase,
                'width': 2.5, 'depth': depth
            })
        
        # 严格执行三光流哈希隔离、自适应多尺寸分辨率直通渲染
        render_single_source(gid, 'bar', creases_params, f"{gid}_bar.png", w_px, h_px, aspect_ratio)
        render_single_source(gid, 'ring', creases_params, f"{gid}_ring.png", w_px, h_px, aspect_ratio)
        render_single_source(gid, 'coax', creases_params, f"{gid}_coax.png", w_px, h_px, aspect_ratio)
        
        for block in bpy.data.materials:
            if block.users == 0: bpy.data.materials.remove(block)
        for block in bpy.data.images:
            if block.users == 0: bpy.data.images.remove(block)
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    print(f"\n🎉 漏洞彻底堵死！全尺寸自适应追踪流水线完美闭合，图片绝对 1:1 输出。")

if __name__ == '__main__':
    main_batch_pipeline()
