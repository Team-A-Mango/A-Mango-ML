from fastapi import APIRouter
from ml_service.detect_hand_sign import HandSignDetector
from ml_service.face_recognition import FaceRecognition
from fastapi import Body,HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from domain.json_utils import save_to_json, load_json
import numpy as np
import cv2


class FaceRecognitionUnlockResult(BaseModel):
    first_storage_room_lock: str
    second_storage_room_lock: str


router = APIRouter(
    prefix="/api",
)


@router.get("/a-mango/return_locking")
async def return_locking():
    registered_storage_room_data = load_json("registered_storage_room_data.json")

    first_storage_locking = registered_storage_room_data["1_storage_room"]["locking"]
    second_storage_locking = registered_storage_room_data["2_storage_room"]["locking"]

    print(f"1번 물품보관함 잠김 여부 : {first_storage_locking}, 2번 물품보관함 잠김 여부 : {second_storage_locking}")
    return first_storage_locking, second_storage_locking



@router.post("/a-mango/get_parameters")
async def get_parameters(image_url: str = Body("default_url", embed=False),
                         hand_sign: str = Body("default_value", embed=False),
                         storage_room_number: str = Body("default_value", embed=False)
):

    storage_room_data = load_json("registered_storage_room_data.json")
    if storage_room_number == "1":
        storage_room_data["1_storage_room"]["image_url"] = image_url
        storage_room_data["1_storage_room"]["hand_sign"] = hand_sign

    elif storage_room_number == "2":
        storage_room_data["2_storage_room"]["image_url"] = image_url
        storage_room_data["2_storage_room"]["hand_sign"] = hand_sign

    if storage_room_number == "1":
        if storage_room_data["1_storage_room"]["locking"] == 'unlock':
            storage_room_data["1_storage_room"]["locking"] = "lock"


    elif storage_room_number == "2":
        if storage_room_data["2_storage_room"]["locking"] == 'unlock':
            storage_room_data["2_storage_room"]["locking"] = "lock"


    save_to_json("registered_storage_room_data.json", storage_room_data)







@router.post("/a-mango/face_recognition", response_model=FaceRecognitionUnlockResult)
async def amango_face_recognition(file: UploadFile = File(...)):
    try:
        face_recognition = FaceRecognition()
        hand_sign_detector = HandSignDetector("saved_model/best.pt")
        registered_storage_room_data = load_json("registered_storage_room_data.json")
        computed_distance_list = list()
        detected_hand_sign_list = list()
        first_storage_room_lock = 'lock'
        second_storage_room_lock = 'lock'
        query_data = await file.read()

        # 이미지를 메모리에서 바로 읽어서 OpenCV로 처리
        np_array = np.frombuffer(query_data, np.uint8)  # 바이트 데이터를 NumPy 배열로 변환
        query_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # OpenCV로 이미지를 디코딩

        for i in range(2):
            query_face_img = face_recognition.extract_face(query_image)

            registered_img = face_recognition.image_url_downloader(registered_storage_room_data[f"{str(i+1)}"+"_storage_room"]["image_url"])
            registered_face_img = face_recognition.extract_face(registered_img)

            embedded_res_img = face_recognition.face_embedding(registered_face_img)
            embedded_query_img = face_recognition.face_embedding(query_face_img)

            computed_distance = face_recognition.compute_similarity_distance(embedded_query_img, embedded_res_img)
            computed_distance_list.append(computed_distance)
            detected_hand_sign = hand_sign_detector.detect(query_image)
            detected_hand_sign_list.append(detected_hand_sign)

            print(f"{i+1}.computed distance = {computed_distance}")





        for i in range(2):
            if computed_distance_list[i] < 0.9:
                if detected_hand_sign_list[i][0] == registered_storage_room_data[f"{str(i+1)}"+"_storage_room"]["hand_sign"]:
                    if i == 0:
                        first_storage_room_lock = "unlock"
                        print("1번 보관함 open")
                        registered_storage_room_data["1_storage_room"]["locking"] = 'unlock'


                    elif i == 1:
                        second_storage_room_lock = "unlock"
                        print("2번 보관함 open")
                        registered_storage_room_data["2_storage_room"]["locking"] = 'unlock'


        print(f"1번 보관함 : {first_storage_room_lock}, 2번 보관함 : {second_storage_room_lock}")
        save_to_json("registered_storage_room_data.json", registered_storage_room_data)

        return {
                "first_storage_room_lock" : first_storage_room_lock,
                "second_storage_room_lock" : second_storage_room_lock
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


