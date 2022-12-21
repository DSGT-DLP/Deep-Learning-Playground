import os
import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw
import io
import base64

class RekognitionImage():
    def __init__ (self, image, image_name, rekognition_client):
        self.image = image
        self.image_name = image_name
        self.rekognition_client = rekognition_client

    @classmethod
    def from_file(cls, file_name, rekognition_client):
        image = {'Bytes': open(file_name, 'rb').read()}
        return cls(image, file_name, rekognition_client)

    def detect_labels(self, max_labels):
        try:
            response = self.rekognition_client.detect_labels(
                Image=self.image, MaxLabels=max_labels)
        except ClientError:
            raise Exception("AWS client error")
        else:
            return response['Labels']

    def recognize_celebrities(self):
        try:
            response = self.rekognition_client.recognize_celebrities(
                Image=self.image)
            celebrities = response['CelebrityFaces']
            other_faces = response['UnrecognizedFaces']
        except ClientError:
            raise Exception("AWS client error")
        else:
            return celebrities, other_faces

    def compare_faces(self, target_image, similarity):
        try:
            response = self.rekognition_client.compare_faces(
                SourceImage=self.image,
                TargetImage=target_image.image,
                SimilarityThreshold=similarity)
            matches = response['FaceMatches']
            unmatches = response['UnmatchedFaces']
        except ClientError:
             raise Exception("AWS client error")
        else:
            return matches, unmatches

def show_bounding_boxes(image_bytes, box_sets, names, colors):
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    for boxes, color, name in zip(box_sets, colors, names):
        for box in boxes:
            print(box['Left'])
            left = image.width * box['Left']
            top = image.height * box['Top']
            right = (image.width * box['Width']) + left
            bottom = (image.height * box['Height']) + top
            draw.rectangle([left, top, right, bottom], outline=color, width=3)
            draw.text((left, top), name, fill = "black")
    im_file = io.BytesIO()
    image.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64 

def detect_labels_from_file(file_name, problem_type):
    rekognition_client = boto3.client('rekognition')
    image = RekognitionImage.from_file(file_name, rekognition_client)
    
    names = []
    box_sets = []
    label_set = []
    if problem_type == "labels":
        labels = image.detect_labels(100)
        for label in labels:
            if(label.get('Instances')):
                name = label.get('Name')
                names.append(name)
                label_set.append({"name" : name, "confidence" : label.get('Confidence')})
                box_sets.append([inst['BoundingBox'] for inst in label.get('Instances')])
        colors = ['aqua', 'red', 'white', 'blue', 'yellow', 'green']
        im_b64 = show_bounding_boxes(image.image['Bytes'], box_sets, names, colors[:len(names)])
        return im_b64, label_set
    elif problem_type == "celebrities":
        labels, _ = image.recognize_celebrities()
        for label in labels:
            name = label.get('Name')
            names.append(name)
            label_set.append({"name" : name, "confidence" : label.get('Face').get('Confidence')})
            box_sets.append([label.get('Face').get('BoundingBox')])
        im_b64 = show_bounding_boxes(image.image['Bytes'], box_sets, names, ['aqua'])
        return im_b64, label_set

def demo():
    file_name = "C:/Users/austi/Downloads/download.jpg"
    detect_labels_from_file(file_name)


def rekognition_img_drive(IMAGE_UPLOAD_FOLDER, problem_type):
    for x in os.listdir(IMAGE_UPLOAD_FOLDER):
        if x != ".gitkeep":
            img_file = os.path.join(
                os.path.abspath(IMAGE_UPLOAD_FOLDER), x)
            break
    im_b64, label_set = detect_labels_from_file(img_file, problem_type)
    return { "image_data" : im_b64.decode('ascii'), "labels" : label_set }

