import torch
import numpy as np
import torch.nn as nn
from torchvision.transforms import functional as F
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from got10k.trackers import Tracker
from PIL import Image, ImageFilter

# LTMU-H
from ltmuh import LTMUH

def calculate_iou(box1, box2):
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[0] + box1[2], box2[0] + box2[2])
    y2_i = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

# Pyramid
def image_pyramid(image):
    pyramid_outputs = [image]

    for _ in range(2):
        image = image.filter(ImageFilter.GaussianBlur(radius=2))
        width, height = image.size
        image = image.resize((width // 2, height // 2))
        pyramid_outputs.append(image)

    return pyramid_outputs

# Deformable DETR pretrained
class DeformableDetrTransformer(nn.Module):
    def __init__(self):
        super(DeformableDetrTransformer, self).__init__()

    def init(self):
        self.image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        self.deformable_detr = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

    def forward(self, x):
        image_tensor = self.image_processor(images=x, return_tensors="pt")
        outputs = self.deformable_detr(**image_tensor)

        target_sizes = torch.tensor([x.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

        box = results['boxes'][0]
        box = [round(i, 2) for i in box.tolist()]

        x, y, w, h = box

        return [float(x), float(y), float(w), float(h)]

transformer_model = DeformableDetrTransformer()
tracker_1 = LTMUH()
tracker_2 = LTMUH()
tracker_3 = LTMUH()

class LTMUPym(Tracker):
    def __init__(self):
        super(LTMUPym, self).__init__(name='LTMU-Pym', is_deterministic=False)

    def init(self, image, box):
        images = image_pyramid(image)

        self.last_gt = box

        transformer_model.init()

        tracker_1.init(images[0], box * 1.0)
        tracker_2.init(images[1], box * 0.5)
        tracker_3.init(images[2], box * 0.25)

    def update(self, image):
        images = image_pyramid(image)
        
        transformer_outputs = [transformer_model(images[0]), transformer_model(images[1]), transformer_model(images[2])]

        tracker_outputs = [tracker_1.update(images[0]), tracker_2.update(images[1]), tracker_3.update(images[2])]

        output = self.combine_outputs(transformer_outputs, tracker_outputs)
        return output
    
    def combine_outputs(self, transformer_outputs, tracker_outputs):
        gt_box = self.last_gt

        ltmu_boxes = []
        detr_boxes = []
        scaling_factors = [1.0, 0.5, 0.25]

        for transformer_output, tracker_output, scaling_factor in zip(transformer_outputs, tracker_outputs, scaling_factors):
            x, y, w, h = transformer_output
            scaled_transformer_boxes = [float(x / scaling_factor), float(y / scaling_factor), float(w / scaling_factor), float(h / scaling_factor)]

            x, y, w, h = tracker_output
            scaled_tracker_boxes = [float(x / scaling_factor), float(y / scaling_factor), float(w / scaling_factor), float(h / scaling_factor)]
            
            detr_boxes.append(scaled_transformer_boxes)
            ltmu_boxes.append(scaled_tracker_boxes)

        # Check which is better
        iou_ltmu = calculate_iou(gt_box, ltmu_boxes[0])
        iou_detr = calculate_iou(gt_box, detr_boxes[0])

        if (iou_ltmu >= iou_detr):
            selected_box = np.mean(ltmu_boxes, axis=0)
        else:
            selected_box = np.mean(detr_boxes, axis=0)

        x, y, w, h = selected_box
        bbox = [float(x), float(y), float(w), float(h)]
        self.last_gt = bbox
        return bbox