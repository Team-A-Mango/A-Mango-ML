import requests
from pydantic import BaseModel, validator, ValidationError


class ImageDownloader(BaseModel):
    url: str

    @validator('url')
    def validate_and_download_image(cls, v):
        if not v or not v.strip():
            raise ValueError('이미지 URL이 비어 있습니다.')

        try:
            response = requests.get(v, timeout=10)
            response.raise_for_status()  # 상태 코드 확인
        except requests.exceptions.RequestException as e:
            raise ValueError(f'이미지 다운로드 실패: {e}')

        # 다운로드한 이미지 데이터가 비어 있는지 확인
        if not response.content:
            raise ValueError('다운로드한 이미지 데이터가 비어 있습니다.')

        return v
