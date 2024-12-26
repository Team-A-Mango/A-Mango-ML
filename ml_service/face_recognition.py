import cv2
import numpy as np
from keras_facenet import FaceNet
import mediapipe as mp
import os
import requests
import logging
import absl.logging

# TensorFlow 경고 억제 설정
mp.solutions.drawing_utils.OPTIONAL_LOGGING = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # WARNING 이상
logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)


class FaceRecognition:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.3)
        self.embedder = FaceNet()


    @staticmethod
    def image_url_downloader(url):
        # 다운받을 이미지 URL
        image = requests.get(url)
        # 이미지 열기
        img_array = np.asarray(bytearray(image.content), dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img_cv

    def extract_face(self, img, is_path=False):
        if is_path:
            img = cv2.imread(img)
        detector_results = self.detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        box = list()
        if detector_results.detections:
            for i, detection in enumerate(detector_results.detections):
                box.append(detection.location_data.relative_bounding_box)

            xmin = max(int(box[0].xmin * img.shape[1]), 0)
            ymin = max(int(box[0].ymin * img.shape[0]), 0)
            width = int(box[0].width * img.shape[1])
            height = int(box[0].height * img.shape[0])

            face_img = img[ymin:ymin + height, xmin:xmin + width]
            cv2.imwrite(f'image/face.jpg', face_img)


        else:
            print('얼굴이 검출되지 않았습니다!')


        return face_img




    def face_embedding(self, face_img):
        face_img = face_img.astype('float32')  # 3D (160 x 160 x 3)
        face_img = np.expand_dims(face_img, axis=0)  # 4D (none x 160 x 160 x 3)
        face_embedding = self.embedder.embeddings(face_img)

        return face_embedding[0]  # 512D (1 x 1 x 512)




    def compute_similarity_distance(self, query_face_embedding, registered_face_embedding):
        query_face_embedding = np.array(query_face_embedding)
        registered_face_embedding = np.array(registered_face_embedding)
        distance = np.linalg.norm(query_face_embedding - registered_face_embedding)  # L2 Norm
        print(distance)


        return distance

