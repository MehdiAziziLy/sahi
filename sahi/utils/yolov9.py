import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolov9TestConstants:
    YOLOV9T_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt"
    YOLOV9T_MODEL_PATH = "tests/data/models/yolov9/yolov9t.pt"

    YOLOV9S_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9s.pt"
    YOLOV9S_MODEL_PATH = "tests/data/models/yolov9/yolov9s.pt"

    YOLOV9M_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9m.pt"
    YOLOV9M_MODEL_PATH = "tests/data/models/yolov9/yolov9m.pt"

    YOLOV9C_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9c.pt"
    YOLOV9C_MODEL_PATH = "tests/data/models/yolov9/yolov9c.pt"

    YOLOV9E_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9e.pt"
    YOLOV9E_MODEL_PATH = "tests/data/models/yolov9/yolov9e.pt"

    YOLOV9C_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c-seg.pt"
    YOLOV9C_SEG_MODEL_PATH = "tests/data/models/yolov9/yolov9c-seg.pt"

    YOLOV9E_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt"
    YOLOV9E_SEG_MODEL_PATH = "tests/data/models/yolov9/yolov9e-seg.pt"


def download_yolov9t_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9T_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9T_MODEL_URL,
            destination_path,
        )


def download_yolov9s_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9S_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9S_MODEL_URL,
            destination_path,
        )


def download_yolov9m_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9M_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9M_MODEL_URL,
            destination_path,
        )


def download_yolov9c_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9C_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9C_MODEL_URL,
            destination_path,
        )


def download_yolov9e_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9E_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9E_MODEL_URL,
            destination_path,
        )


def download_yolov9c_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9C_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9C_SEG_MODEL_URL,
            destination_path,
        )


def download_yolov9e_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9E_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9E_SEG_MODEL_URL,
            destination_path,
        )
