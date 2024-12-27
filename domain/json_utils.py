import json
def save_to_json(file_name, data):
    """
    Save the given data to a JSON file.

    :param file_name: Name of the JSON file to save.
    :param data: Data to save (should be serializable).
    """
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # Define the data to save


def load_json(file_path):
    """주어진 파일 경로에서 JSON 데이터를 로드하는 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except json.JSONDecodeError:
        print("JSON 형식이 잘못되었습니다.")
        return None