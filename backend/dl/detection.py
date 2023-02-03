from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
import torchvision
import io
import base64
from PIL import Image, ImageDraw
from backend.aws_helpers.aws_rekognition_utils.rekognition_client import rekognition_detection
import os
from itertools import cycle
from backend.dl.dl_model_parser import parse_deep_user_architecture
import csv

def transform_image(img_file, transforms):
    img = Image.open(img_file)
    for x in transforms:
        print(x)
    transforms = parse_deep_user_architecture(transforms)
    transforms = torchvision.transforms.Compose(
        [x for x in transforms])
    img = transforms(img)
    return torchvision.transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img

def yolo_detection(image):
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-base')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-base')

    inputs = feature_extractor(images=image.convert("RGB"), return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    colors = ['aqua', 'red', 'white', 'blue', 'yellow', 'green']
    names = []
    box_sets = []
    label_set = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box_sets.append(box)
        name = model.config.id2label[label.item()]
        names.append(name)
        label_set.append({"name" : name, "confidence" : score.item()})
    im_b64 = show_bounding_boxes(image, box_sets, names, colors)
    return im_b64, label_set

def show_bounding_boxes(image, box_sets, names, colors):
    draw = ImageDraw.Draw(image)
    for box, color, name in zip(box_sets, cycle(colors), names):
        box = box.tolist()
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left, top), name, fill = "black")
    im_file = io.BytesIO()
    image.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64 

def detection_img_drive(IMAGE_UPLOAD_FOLDER, detection_type, problem_type, transforms):   
    for x in os.listdir(IMAGE_UPLOAD_FOLDER):
        if x != ".gitkeep":
            img_file = os.path.join(
                os.path.abspath(IMAGE_UPLOAD_FOLDER), x)
            break
    image = transform_image(img_file, transforms)
    if (detection_type == "rekognition"):
        im_b64, label_set = rekognition_detection(image, problem_type)
    elif (detection_type == "yolo"):
        im_b64, label_set = yolo_detection(image)
    if label_set:
        backend_dir = (
            ""
            if (os.getcwd()).split("\\")[-1].split("/")[-1] == "backend"
            else "backend"
            if ("backend" in os.listdir(os.getcwd()))
            else "../backend"
        )
        keys = label_set[0].keys()
        with open(os.path.join(backend_dir, "detection_results.csv"), 'w', newline = '') as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(label_set)
    return { "auxiliary_outputs" : { "image_data" : im_b64.decode('ascii') }, "dl_results" : label_set }