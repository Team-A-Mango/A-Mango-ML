mport cv2
import mediapipe as mp
import time
import os
from ultralytics import YOLOv10


class HandSignDetector:
    def __init__(self, saved_model_path):
        self.model_path= saved_model_path
        self.detector = YOLOv10(saved_model_path)


    def detect(self, image):
        detected_results = self.detector(image)
        class_names = list()

        if detected_results and len(detected_results) > 0: 
            for result in detected_results:  # ì¬ë¬ ì´ë¯¸ì§€ì ê²°ê³¼ê°€ ìì ì ìì
                if result.boxes and result.boxes.cls is not None:
                    class_indices = result.boxes.cls.cpu().numpy()  # í´ëì¤ ì¸ë±ì¤ ì¶ì¶
                    class_names = [result.names[int(cls_idx)] for cls_idx in class_indices]  # í´ëì¤ ì´ë¦ ë³€í
                    print("Predicted class names:", class_names)
        else:
            print("No objects detected.")

        return class_names

# Mediapipe ???
mp_face_detection = mp.solutions.face_detection
hand_sign_detector = HandSignDetector("/home/pi/Desktop/best.pt")
mp_drawing = mp.solutions.drawing_utils

# ??? ???
#!/usr/bin/python3

import cv2

from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

# Relay 1
GPIO.setup(21, GPIO.OUT)
# Relay 2
GPIO.setup(26, GPIO.OUT)

# Grab images as numpy arrays and leave everything else to OpenCV.
cv2.startWindowThread()
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1200, 900)}))
picam2.start()


# ?? ?? ?? ?? ??
face_detected_time = None  # ??? ?? ??? ??
save_image_delay = 3  # ??? ??? ? ????? ?? ?? (?)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
    while True:
        im = picam2.capture_array()
        image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      
        
        import requests

        url = "http://34.42.223.216:3505/api/a-mango/return_locking"
        response = requests.get(url)
        
        if response.json()[0] == 'lock':
                GPIO.output(21, GPIO.HIGH)
                print('Relay 1 ON')
                
        if response.json()[1] == 'lock':
                GPIO.output(26, GPIO.HIGH)
                print('Relay 2 ON')
                
        if response.json()[0] == 'unlock':
                GPIO.output(21, GPIO.LOW)
                print("Relay 1 OFF")
        if response.json()[1] == 'unlock':
                GPIO.output(26, GPIO.LOW)
                print("Relay 2 OFF")
        # ?? ?? ??
        results = face_detection.process(image)
        detected_hand_sign = hand_sign_detector.detect(image)

        # ???? BGR? ?? (OpenCV?)
       
        if results.detections:
            if detected_hand_sign is not None:
                # ??? ???
                if face_detected_time is None:
                    # ??? ?? ???? ? ?? ??
                    face_detected_time = time.time()
                else:
                    # ??? ??? 3? ???? ??
                    if time.time() - face_detected_time >= save_image_delay:
                        # ??? ??
                        cv2.imwrite('/home/pi/Pictures/saved_image.jpg', image_rgb)
                        print("image saved!")

                        
                        import requests

                        # ???? ?? ??
                        file_path = "/home/pi/Pictures/saved_image.jpg"
                        # ??? ?? POST ???? ??
                        with open(file_path, 'rb') as file:
                                files = {'file': (file_path, file, 'image/jpeg')}
                                response = requests.post("http://34.42.223.216:3505/api/a-mango/face_recognition", files=files)

                        # ?? ?? ??
                        if response.status_code == 200:
                                print("Success:", response.json())
                        else:
                                print("Error:", response.json())


                        print(response.json())
                        
                        if response.json()['first_storage_room_lock'] == 'unlock':
                                GPIO.output(21, GPIO.LOW)
                                print('1번 보관함 OPEN')
                        
                        if response.json()['second_storage_room_lock'] == 'unlock':
                                GPIO.output(26, GPIO.LOW)
                                print('2번 보관함 OPEN')

		if response.json()['first_storage_room_lock'] == 'lock':
		        GPIO.output(21, GPIO.HIGH)
		        print('1번 보관함 LOCK')

		if response.json()['second_storage_room_lock'] == 'lock':
		        GPIO.output(26, GPIO.HIGH)
		         print('2번 보관함 LOCK')		
		
                            

                        face_detected_time = None  # ?? ???
                        
                        
                         
            else:
                # ??? ???? ??? ?? ???
                face_detected_time = None

        # ??? ?? ?? ???
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # ??? ??
        #cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        # ????? imshow ?? ?? ??

        # ESC? ?? ??
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()