import os

# root directory is the parent directory
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(path):
    return os.path.join(ROOT_PATH, path)