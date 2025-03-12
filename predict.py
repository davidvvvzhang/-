import os
import numpy as np
from ultralytics import YOLO
import cv2

# 加载训练好的模型权重
model = YOLO('runs/detect/train5/weights/best.pt')

# 定义图片和标注文件夹路径
image_dir = 'test/images'
label_dir = 'test/labels'
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

fruit_classes = ['apple', 'orange', 'banana']
class_map = {0: 'apple', 1: 'banana', 2: 'orange'}  # 修改为您的类别映射


# 读取 YOLO 格式的 txt 文件
def read_yolo_label(txt_file, img_shape):
    h, w = img_shape[:2]
    objects = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:]]
            # 转换归一化坐标为实际坐标
            x_center, y_center, box_w, box_h = bbox
            xmin = int((x_center - box_w / 2) * w)
            xmax = int((x_center + box_w / 2) * w)
            ymin = int((y_center - box_h / 2) * h)
            ymax = int((y_center + box_h / 2) * h)
            objects.append({'class': class_map[class_id], 'bbox': [xmin, ymin, xmax, ymax]})
    return objects


# 用于存储所有图片的检测结果和ground truth
detections = []
ground_truths = []

# 对每张图片进行预测
for image_path in image_paths:
    # 读取图片
    img0 = cv2.imread(image_path)  # BGR
    results = model(img0)

    # 获取图片文件名
    image_name = os.path.basename(image_path).split('.')[0]

    # 读取对应的标注文件
    label_file = os.path.join(label_dir, f"{image_name}.txt")
    img_shape = img0.shape
    gt_objects = read_yolo_label(label_file, img_shape)

    for obj in gt_objects:
        ground_truths.append({
            "image": image_name,
            "class": obj["class"],
            "bbox": obj["bbox"]
        })

    for result in results:
        for box in result.boxes:
            cls = box.cls.item()  # 获取类别索引
            class_name = model.names[int(cls)]  # 获取类别名称
            if class_name in fruit_classes:
                bbox = box.xyxy[0].cpu().numpy()  # 确保bbox格式正确
                detections.append({
                    "image": image_name,
                    "class": class_name,
                    "bbox": bbox,
                    "confidence": box.conf.item()
                })


# 函数：计算IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


# 计算Precision, Recall, IoU 和 mAP50
def calculate_metrics(detections, ground_truths):
    # 初始化
    metrics = {
        "all": {"iou": [], "precision": [], "recall": [], "ap50": []},
        "apple": {"iou": [], "precision": [], "recall": [], "ap50": []},
        "banana": {"iou": [], "precision": [], "recall": [], "ap50": []},
        "orange": {"iou": [], "precision": [], "recall": [], "ap50": []},
    }

    for gt in ground_truths:
        gt_class = gt["class"]
        gt_bbox = gt["bbox"]
        gt_image = gt["image"]

        # 找到与该ground truth匹配的检测
        matched_detections = [
            det for det in detections
            if det["image"] == gt_image and det["class"] == gt_class
        ]

        # 如果没有检测到对应的类
        if not matched_detections:
            metrics[gt_class]["recall"].append(0)
            metrics["all"]["recall"].append(0)
            continue

        # 计算IoU
        ious = [calculate_iou(det["bbox"], gt_bbox) for det in matched_detections]
        best_iou = max(ious)
        best_det = matched_detections[ious.index(best_iou)]

        # 更新指标
        metrics[gt_class]["iou"].append(best_iou)
        metrics[gt_class]["precision"].append(1 if best_iou > 0.5 else 0)
        metrics[gt_class]["recall"].append(1)
        metrics[gt_class]["ap50"].append(best_det["confidence"] if best_iou > 0.5 else 0)

        metrics["all"]["iou"].append(best_iou)
        metrics["all"]["precision"].append(1 if best_iou > 0.5 else 0)
        metrics["all"]["recall"].append(1)
        metrics["all"]["ap50"].append(best_det["confidence"] if best_iou > 0.5 else 0)

    # 计算平均值
    for key in metrics.keys():
        metrics[key]["iou"] = np.mean(metrics[key]["iou"])
        metrics[key]["precision"] = np.mean(metrics[key]["precision"])
        metrics[key]["recall"] = np.mean(metrics[key]["recall"])
        metrics[key]["ap50"] = np.mean(metrics[key]["ap50"])

    return metrics


# 计算指标
metrics = calculate_metrics(detections, ground_truths)

# 输出结果
print("Class\tIoU\tPrecision\tRecall\tmAP50")
for key in metrics.keys():
    print(
        f"{key}\t{metrics[key]['iou']:.3f}\t{metrics[key]['precision']:.3f}\t{metrics[key]['recall']:.3f}\t{metrics[key]['ap50']:.3f}")

# 示例输出格式:
# Class     IoU     Precision    Recall  mAP50
# all       0.921   0.875        0.900   0.867
# apple     0.900   0.850        0.880   0.850
# banana    0.950   0.900        0.920   0.900
# orange    0.915   0.880        0.895   0.875
