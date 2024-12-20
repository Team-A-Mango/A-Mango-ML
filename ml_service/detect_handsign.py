import os
import cv2
from ultralytics import YOLOv10


class HandSignDetector:
    def __init__(self, saved_model_path):
        self.model_path= saved_model_path
        self.detector = YOLOv10(saved_model_path)


    def detect(self, image_path):
        image = cv2.imread(image_path)
        detected_results = self.detector(image)

        if detected_results and len(detected_results) > 0:  # 결과가 존재하는지 확인
            for result in detected_results:  # 여러 이미지의 결과가 있을 수 있음
                if result.boxes and result.boxes.cls is not None:
                    class_indices = result.boxes.cls.cpu().numpy()  # 클래스 인덱스 추출
                    class_names = [result.names[int(cls_idx)] for cls_idx in class_indices]  # 클래스 이름 변환
                    print("Predicted class names:", class_names)
        else:
            print("No objects detected.")

        return class_names