import os
import xml.etree.ElementTree as ET
import shutil
import random


def convert_annotation(xml_file, output_path, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # 检查宽度和高度是否为零
    if width == 0 or height == 0:
        raise ValueError(f"Invalid image dimensions in {xml_file}: width={width}, height={height}")

    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in classes:
                continue
            class_id = classes.index(name)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                 int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            bbox = [(b[0] + b[2]) / 2 / width, (b[1] + b[3]) / 2 / height,
                    (b[2] - b[0]) / width, (b[3] - b[1]) / height]
            f.write(f"{class_id} " + " ".join([str(a) for a in bbox]) + '\n')


# 类别名称
classes = ['apple', 'banana', 'orange', 'mixed']

# 路径
data_dirs = ['dataset/label_data', 'dataset/test']
train_image_dir = 'dataset/train/images'
train_label_dir = 'dataset/train/labels'
val_image_dir = 'dataset/val/images'
val_label_dir = 'dataset/val/labels'

# 创建目录
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 收集所有文件
all_files = []
for data_dir in data_dirs:
    files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    for file in files:
        all_files.append((os.path.join(data_dir, file), os.path.join(data_dir, os.path.splitext(file)[0] + '.xml')))

# 打乱文件顺序并划分训练集和验证集（例如80%作为训练集，20%作为验证集）
random.shuffle(all_files)
split_idx = int(0.8 * len(all_files))

train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

# 处理训练集
for image_file, xml_file in train_files:
    shutil.copy(image_file, train_image_dir)
    try:
        convert_annotation(xml_file,
                           os.path.join(train_label_dir, os.path.splitext(os.path.basename(image_file))[0] + '.txt'),
                           classes)
    except ValueError as e:
        print(f"Skipping file {image_file} due to error: {e}")

# 处理验证集
for image_file, xml_file in val_files:
    shutil.copy(image_file, val_image_dir)
    try:
        convert_annotation(xml_file,
                           os.path.join(val_label_dir, os.path.splitext(os.path.basename(image_file))[0] + '.txt'),
                           classes)
    except ValueError as e:
        print(f"Skipping file {image_file} due to error: {e}")
