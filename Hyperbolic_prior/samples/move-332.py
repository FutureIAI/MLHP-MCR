import os
import shutil

# ================= 配置区域 =================
SRC_DIR = '/home/cgz/lzc/FPC_Dataset/FPC/2026FPC-tan'
DST_DIR = '/home/cgz/lzc/FPC_Dataset/FPC/2026FPC-tan_Selected_227'
TARGET_GROUPS = 227

# 选择操作模式：
# 'copy' -> 复制到新文件夹（推荐，防丢失数据）
# 'move' -> 剪切到新文件夹（如果你的硬盘空间不够）
ACTION = 'move'


# ============================================

def main():
    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
        print(f"📁 已创建目标文件夹: {DST_DIR}")

    # 1. 扫描并提取所有组的前缀 (如 'img0001')
    all_files = os.listdir(SRC_DIR)
    base_names = set()
    for f in all_files:
        if f.endswith('.jpg') or f.endswith('.xml'):
            # 通过下划线切分，获取 img0xxx 前缀
            base_name = f.split('_')[0]
            base_names.add(base_name)

    # 必须排序，保证按你原本的型号顺序排列
    base_names = sorted(list(base_names))
    total_groups = len(base_names)

    print(f"📊 扫描完毕：共发现 {total_groups} 组完整数据。")

    if total_groups < TARGET_GROUPS:
        print(f"❌ 错误：源文件夹中的组数 ({total_groups}) 小于目标提取组数 ({TARGET_GROUPS})！")
        return

    # 2. 核心数学逻辑：均匀采样计算索引
    # 利用 int(i * 总数 / 目标数) 保证在整个序列中完美均匀散布
    selected_indices = [int(i * total_groups / TARGET_GROUPS) for i in range(TARGET_GROUPS)]
    selected_bases = [base_names[i] for i in selected_indices]

    print(f"🎯 开始均匀提取 {TARGET_GROUPS} 组数据...")

    # 每组包含的 6 个固定后缀
    suffixes = [
        '_bar.jpg', '_bar.xml',
        '_coax.jpg', '_coax.xml',
        '_ring.jpg', '_ring.xml'
    ]

    # 3. 执行文件迁移
    success_count = 0
    missing_files = []

    for base in selected_bases:
        group_intact = True

        # 先检查这组的 6 个文件是不是都齐
        for suffix in suffixes:
            if not os.path.exists(os.path.join(SRC_DIR, base + suffix)):
                group_intact = False
                missing_files.append(base + suffix)

        if group_intact:
            for suffix in suffixes:
                src_path = os.path.join(SRC_DIR, base + suffix)
                dst_path = os.path.join(DST_DIR, base + suffix)

                if ACTION == 'copy':
                    shutil.copy2(src_path, dst_path)
                elif ACTION == 'move':
                    shutil.move(src_path, dst_path)
            success_count += 1
        else:
            print(f"⚠️ 警告：组 {base} 文件不全，已跳过！")

    # 4. 打印最终报告
    print("\n================ 执行报告 ================")
    print(f"✅ 成功处理组数: {success_count} 组 (共 {success_count * 6} 个文件)")
    print(f"📁 数据已存入: {DST_DIR}")
    if missing_files:
        print(f"❌ 发现缺失文件: {len(missing_files)} 个")
        print(f"缺失列表前5个: {missing_files[:5]}")
    print("==========================================")


if __name__ == "__main__":
    main()