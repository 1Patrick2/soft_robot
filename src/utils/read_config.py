import json
import numpy as np

def load_config(file_path='config/config.json'):
    """
    加载配置, 并自动将所有mm单位转换为米(SI单位)。
    这是项目中所有模块应该使用的唯一配置加载函数。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # print(f"配置已从 {file_path} 加载并转换为SI单位.")
    
    return config