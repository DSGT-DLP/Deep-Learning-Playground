import boto3


class RekognitionImage():
    def __init__ (self, image, image_name, rekognition_client):
        self.image = image
        self.image_name = image_name
        self.rekognition_client = rekognition_client

    def from_file(cls, file_name, rekognition_client):
        image = {'Bytes': open(file_name, 'rb').read()}
        return cls(image, file_name, rekognition_client)

    def detect_labels()
        try:
            response = self.rekognition_client.detect_labels(
                Image=self.image, MaxLabels=max_labels)
            labels = [RekognitionLabel(label) for label in response['Labels']]
        except ClientError:
        else:
            return labels

def detect_labels_from_file(file_name):
    rekognition_client = boto3.client('rekognition')
    image = RekognitionImage.from_file(file_name, rekognition_client)
    labels = image.detect_labels()
    for label in labels:
        pprint(label.to_dict())