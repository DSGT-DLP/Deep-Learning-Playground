import cv2
from PIL import Image
import torchvision
import torch 
from backend.dl.dl_model_parser import parse_deep_user_architecture

def transform_image(img_file, transforms):
    img = Image.open(img_file)
    for x in transforms:
        print(x)
    transforms = parse_deep_user_architecture(transforms)
    transforms = torchvision.transforms.Compose(
        [x for x in transforms])
    img = transforms(img)
    return torchvision.transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img