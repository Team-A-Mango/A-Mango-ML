from fastapi import APIRouter
from ml_service.detect_handsign import HandSignDetector
from ml_service.face_recognition import FaceRecognition
from fastapi import Body,HTTPException
from pydantic import BaseModel



class FaceRecognitionUnlockResult(BaseModel):
    result: str

router = APIRouter(
    prefix="/api",
)


@router.post("/a-mango/face_recognition", response_model=FaceRecognitionUnlockResult)
async def amango_face_recognition(image_url: str = "default_url", hand_sign: str = "default_value"):
    try:
        face_recognition = FaceRecognition()
        hand_sign_detector = HandSignDetector("saved_model/best.pt")
        registered_img = face_recognition.image_url_downloader(image_url)
        query_img = "image/IU.jpeg"
        query_face_img = face_recognition.extract_face(query_img, is_path=True)
        registered_face_img = face_recognition.extract_face(registered_img)
        embedded_res_img = face_recognition.face_embedding(registered_face_img)
        embedded_query_img = face_recognition.face_embedding(query_face_img)

        computed_distance = face_recognition.compute_similarity_distance(embedded_query_img, embedded_res_img)

        detected_hand_sign = hand_sign_detector.detect(query_img)


        if computed_distance < 0.8:
            face_lock = True
            if detected_hand_sign[0] == hand_sign:

                return {"result": "unlocked"}

            else:
                return {"result": "locked"}


        else:
            face_lock = False
            return {"result": "locked"}



        # 라즈베리파이 카메라에서 받은 Face 이미지 (라즈베리파이에 Mediapipe Face Detector를 Live로 돌리고 만약 사람 얼굴이 디텍팅 되면
        # 둘의 L2 distance 컴퓨트 하는 코드
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


