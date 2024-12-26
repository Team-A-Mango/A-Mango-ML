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
async def amango_face_recognition(image_url: str = Body("default_url", embed=False),
                                  hand_sign: str = Body("default_value", embed=False)
):
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
                print("unlock")
                pass

            else:
                print("lock")
                pass

        else:
            face_lock = False
            print("lock")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


