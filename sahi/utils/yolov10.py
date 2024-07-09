import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolov10TestConstants:
    YOLOV10N_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"
    YOLOV10N_MODEL_PATH = "tests/data/models/yolov10/yolov10n.pt"

    YOLOV10S_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt"
    YOLOV10S_MODEL_PATH = "tests/data/models/yolov10/yolov10s.pt"

    YOLOV10M_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt"
    YOLOV10M_MODEL_PATH = "tests/data/models/yolov10/yolov10m.pt"

    YOLOV10B_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10b.pt"
    YOLOV10B_MODEL_PATH = "tests/data/models/yolov10/yolov10b.pt"

    YOLOV10L_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt"
    YOLOV10L_MODEL_PATH = "tests/data/models/yolov10/yolov10l.pt"

    YOLOV10X_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt"
    YOLOV10X_MODEL_PATH = "tests/data/models/yolov10/yolov10x.pt"


def download_yolov10n_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov10TestConstants.YOLOV10N_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov10TestConstants.YOLOV10N_MODEL_URL,
            destination_path,
        )


def download_yolov10s_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov10TestConstants.YOLOV10S_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov10TestConstants.YOLOV10S_MODEL_URL,
            destination_path,
        )


def download_yolov10m_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov10TestConstants.YOLOV10M_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov10TestConstants.YOLOV10M_MODEL_URL,
            destination_path,
        )


def download_yolov10b_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov10TestConstants.YOLOV10B_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov10TestConstants.YOLOV10B_MODEL_URL,
            destination_path,
        )


def download_yolov10l_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov10TestConstants.YOLOV10L_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov10TestConstants.YOLOV10L_MODEL_URL,
            destination_path,
        )


def download_yolov10x_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov10TestConstants.YOLOV10X_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov10TestConstants.YOLOV10X_MODEL_URL,
            destination_path,
        )
