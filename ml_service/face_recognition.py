import cv2
import numpy as np
from keras_facenet import FaceNet
import mediapipe as mp


class FaceRecognition:
    detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.1)
    embedder = FaceNet()

    def __init__(self, filepath):
        self.img = cv2.imread(filepath)
        self.detector_results = self.detector.process(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

    def extract_face(self):
        box = list()

        if self.detector_results.detections:
            for i, detection in enumerate(self.detector_results.detections):
                box.append(detection.location_data.relative_bounding_box)

            xmin = max(int(box[0].xmin * self.img.shape[1]), 0)
            ymin = max(int(box[0].ymin * self.img.shape[0]), 0)
            width = int(box[0].width * self.img.shape[1])
            height = int(box[0].height * self.img.shape[0])

            face_img = self.img[ymin:ymin + height, xmin:xmin + width]
            cv2.imwrite(f'face.jpg', face_img)


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

