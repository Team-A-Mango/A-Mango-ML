from fastapi import FastAPI
from domain import face_and_hand_sign_recognition_router

app = FastAPI()

app.include_router(face_and_hand_sign_recognition_router.router)
