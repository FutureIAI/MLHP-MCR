import os
import cv2
import xml.etree.ElementTree as ET
import shutil


def resize_image_and_xml(src_dir, dst_dir, max_dim=2000):
    """
    终极安全版：等比例缩放超大图像及对应的 XML 标注文件
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    valid_exts = ['.jpg', '.jpeg', '.png']
    img_files = [f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    print(f"🔍 开始处理 {len(img_files)} 个样本...")

    for img_name in img_files:
        base_name, _ = os.path.splitext(img_name)
        xml_name = base_name + '.xml'

        src_img_path = os.path.join(src_dir, img_name)
        src_xml_path = os.path.join(src_dir, xml_name)
        dst_img_path = os.path.join(dst_dir, img_name)
        dst_xml_path = os.path.join(dst_dir, xml_name)

        img = cv2.imread(src_img_path)
        if img is None: continue

        H, W = img.shape[:2]

        # 1. 严格等比例判断
        if max(H, W) > max_dim:
            scale = max_dim / max(H, W)
            new_W = int(W * scale)
            new_H = int(H * scale)

            resized_img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst_img_path, resized_img)
            print(f"✅ [缩放] {img_name}: -> {new_W}x{new_H} (缩放比: {scale:.4f})")
        else:
            scale = 1.0
            new_W, new_H = W, H
            shutil.copy2(src_img_path, dst_img_path)

        # 2. 军工级 XML 安全重写
        if os.path.exists(src_xml_path):
            tree = ET.parse(src_xml_path)
            root = tree.getroot()

            if scale != 1.0:
                # 安全修改全图宽高
                size_node = root.find('size')
                if size_node is not None:
                    if size_node.find('width') is not None: size_node.find('width').text = str(new_W)
                    if size_node.find('height') is not None: size_node.find('height').text = str(new_H)

                # 安全修改所有坐标框
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        # 提取老坐标
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)

                        # 等比例缩放并四舍五入
                        new_xmin = int(round(xmin * scale))
                        new_ymin = int(round(ymin * scale))
                        new_xmax = int(round(xmax * scale))
                        new_ymax = int(round(ymax * scale))

                        # ================= 核心防崩保护区 =================
                        # 保护1：防止越出图片左上边界 (通常 VOC 格式坐标从 1 开始)
                        new_xmin = max(1, new_xmin)
                        new_ymin = max(1, new_ymin)

                        # 保护2：防止越出图片右下边界
                        new_xmax = min(new_xmax, new_W)
                        new_ymax = min(new_ymax, new_H)

                        # 保护3：防止零面积框 (如果缩放后框太小被挤没了，强行撑开 1 像素)
                        if new_xmax <= new_xmin: new_xmax = new_xmin + 1
                        if new_ymax <= new_ymin: new_ymax = new_ymin + 1
                        # =================================================

                        # 重新写入 XML
                        bndbox.find('xmin').text = str(new_xmin)
                        bndbox.find('ymin').text = str(new_ymin)
                        bndbox.find('xmax').text = str(new_xmax)
                        bndbox.find('ymax').text = str(new_ymax)

            tree.write(dst_xml_path, encoding='utf-8', xml_declaration=True)


if __name__ == "__main__":
    SOURCE_FOLDER = r"/home/cgz/lzc/FPC_Dataset/FPC/出错-227"
    DESTINATION_FOLDER = r"/home/cgz/lzc/FPC_Dataset/FPC/resized_227"

    resize_image_and_xml(src_dir=SOURCE_FOLDER, dst_dir=DESTINATION_FOLDER, max_dim=3500)