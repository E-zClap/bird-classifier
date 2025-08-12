import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import shutil

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

input_folder = "images_test" 
output_folder = "images_cropped"
processed_folder = "images_original_cropped"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        outputs = predictor(image)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        boxes = instances.pred_boxes.tensor.cpu().numpy() 

        for i, (pred_class, box) in enumerate(zip(pred_classes, boxes)):
            if pred_class == 14:
                x1, y1, x2, y2 = box
                cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
                output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_bird_{i}.jpg")
                cv2.imwrite(output_path, cropped_image)
                print(f"Bird image {i} saved at {output_path}")

        shutil.move(image_path, os.path.join(processed_folder, filename))