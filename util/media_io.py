# image_utils.py
import gradio as gr
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Union, Tuple, List, Dict
import secrets


def fetch_image_as_numpy(image_url: str) -> np.ndarray:
    """
    从给定的FastAPI接口获取图片并转换为NumPy数组
    :param image_url: FastAPI提供的图片URL
    :return: 转换后的NumPy数组
    """
    # 获取图片内容
    response = requests.get(image_url)
    if response.status_code == 200:
        # 使用PIL打开图片
        img = Image.open(BytesIO(response.content))
        # 转换为NumPy数组
        img_array = np.array(img)
        return img_array
    else:
        raise Exception(f"Failed to fetch image from {image_url}")


# 检查文件后缀格式
def allowed_file(filename: str) -> bool:
    # ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}
    ALLOWED_EXTENSIONS = {"png"}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_random_dirname(length: int = 16) -> str:
    '''
    generate random file name
    :param length:
    :return:
    '''
    # 使用 secrets 模块生成一个安全的随机字符串作为文件名
    random_name = secrets.token_hex(length // 2)  # 每两个十六进制字符代表一个字节
    return f"{random_name}"


def generate_random_filename(extension: str, length: int = 16) -> str:
    '''
    generate random file name
    :param extension:
    :param length:
    :return:
    '''
    # 使用 secrets 模块生成一个安全的随机字符串作为文件名
    random_name = secrets.token_hex(length // 2)  # 每两个十六进制字符代表一个字节
    return f"{random_name}.{extension}"


def on_file_upload(data: Union[str, gr.File], server_url: str, other_info_dict: Dict[str, object] = None):
    if isinstance(data, str):

        try:
            with open(data, 'rb') as file:
                files = {'file': (data, file)}

                if other_info_dict is not None:
                    response = requests.post(server_url, files=files, data=other_info_dict)
                else:
                    response = requests.post(server_url, files=files)

            # 返回服务器的响应内容
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "上传失败", "detail": response.text}
        except Exception as e:
            return {"error": "客户端错误", "detail": str(e)}
    else:
        assert "Currently, we only support data acquired from local file system. Other upload type for Image component doesn't support!!"


def on_image_upload(data: Union[str, gr.Image], server_url: str, other_info_dict: Dict[str, object] = None):
    if isinstance(data, str):
        if not allowed_file(data):
            assert "Your upload image is illegal format"

        from PIL import Image
        from io import BytesIO
        import requests

        try:
            # 打开图片并将其保存为二进制数据
            with Image.open(data) as img:
                img_bytes = BytesIO()
                img.save(img_bytes, format=img.format)  # 使用图片的原始格式保存
                img_bytes.seek(0)  # 重置文件指针

            # 使用 requests 上传图片
            files = {"file": (data, img_bytes)}
            if other_info_dict is not None:
                response = requests.post(server_url, files=files, data=other_info_dict)
            else:
                response = requests.post(server_url, files=files)

            # 返回服务器的响应内容
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "上传失败", "detail": response.text}

        except Exception as e:
            return {"error": "客户端错误", "detail": str(e)}

    else:
        assert "Currently, we only support data acquired from local file system. Other upload type for Image component doesn't support!!"


def generate_animation_post(server_url: str, other_info_dict: Dict[str, object] = None):
    try:
        if other_info_dict is not None:
            response = requests.get(server_url, data=other_info_dict)
        else:
            response = requests.get(server_url)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "获取结果失败", "detail": response.text}
    except Exception as e:
        return {"error": "客户端错误", "detail": str(e)}


if __name__ == '__main__':
    res = fetch_image_as_numpy(image_url="http://localhost:8000/static/drawing_demo/char1.png")
    print(res)
