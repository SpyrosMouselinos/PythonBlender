import os
from typing import Union, List

MAYBE_INT = Union[int, None]
MAYBE_FLOAT = Union[float, None]
MAYBE_LIST_INT = List[MAYBE_INT]
MAYBE_LIST_FLOAT = List[MAYBE_FLOAT]
NESTED_MAYBE_LIST_INT = List[MAYBE_LIST_INT]
NESTED_MAYBE_LIST_FLOAT = List[MAYBE_LIST_FLOAT]

translation = {
    "cube": 0,
    "sphere": 1,
    "cylinder": 2,
    "gray": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
    "brown": 4,
    "purple": 5,
    "cyan": 6,
    "yellow": 7,
    "rubber": 0,
    "metal": 1,
    "large": 0,
    "small": 1
}


def find_platform() -> str:
    if os.name == 'nt':
        PLATFORM = 'WIN'
    elif os.name == 'posix':
        PLATFORM = 'LINUX'
    else:
        raise EnvironmentError("Where are you running?\n")
    return PLATFORM


def find_platform_slash() -> str:
    platform = find_platform()
    if platform == 'WIN':
        return '\\'
    elif platform == 'LINUX':
        return '/'


def find_platform_exec() -> str:
    platform = find_platform()
    if platform == 'WIN':
        return 'blender'
    elif platform == 'LINUX':
        return 'sudo ./blender2.79/blender'

PLATFORM_SLASH = find_platform_slash()
UP_TO_HERE_ = PLATFORM_SLASH.join(os.path.abspath(__file__).split(PLATFORM_SLASH)[:-2]).replace(PLATFORM_SLASH, '/')
SPLIT_ = 'Rendered'
OUTPUT_IMAGE_DIR_ = UP_TO_HERE_ + '/images'
OUTPUT_SCENE_DIR_ = UP_TO_HERE_ + '/scenes'
OUTPUT_SCENE_FILE_ = UP_TO_HERE_ + f"/scenes/CLEVR_{SPLIT_}_scenes.json"
OUTPUT_QUESTION_FILE_ = UP_TO_HERE_ + f"/questions/CLEVR_{SPLIT_}_questions.json"
