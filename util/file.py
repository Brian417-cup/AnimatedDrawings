import secrets


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