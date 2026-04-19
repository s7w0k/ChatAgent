"""
为整个工程提供统一的绝对路径
"""
import os

def get_project_root() -> str:
    current_file = os.path.abspath(__file__)  # 当前文件的绝对路径
    current_dir = os.path.dirname(current_file) # 当前文件所在文件夹的绝对路径
    project_root = os.path.dirname(current_dir) # 工程根目录
    return project_root

def get_abs_path(relative_path):
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path