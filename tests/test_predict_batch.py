# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.yolov8 import Yolov8TestConstants, download_yolov8n_model, download_yolov8n_seg_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 640


class TestPredict(unittest.TestCase):
    def test_prediction_score(self):
        from sahi.prediction import PredictionScore

        prediction_score = PredictionScore(np.array(0.6))
        self.assertEqual(type(prediction_score.value), float)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.5), True)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.7), False)

    def test_object_prediction(self):
        from sahi.prediction import ObjectPrediction

    def test_get_prediction_yolov8(self):
        from sahi.models.yolov8 import Yolov8DetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.yolov8 import Yolov8TestConstants, download_yolov8n_model

        # init model
        download_yolov8n_model()

        yolov8_detection_model = Yolov8DetectionModel(
            model_path=Yolov8TestConstants.YOLOV8N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        yolov8_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=yolov8_detection_model, shift_amount=[0, 0], full_shape=None, postprocess=None
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 2)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 2)

    def test_get_prediction_automodel_yolov8(self):
        from sahi.auto_model import AutoDetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.yolov8 import Yolov8TestConstants, download_yolov8n_model

        # init model
        download_yolov8n_model()

        yolov8_detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=Yolov8TestConstants.YOLOV8N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        yolov8_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=yolov8_detection_model, shift_amount=[0, 0], full_shape=None, postprocess=None
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 2)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 2)

    def test_get_sliced_prediction_yolov8(self):
        from sahi.models.yolov8 import Yolov8DetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.yolov8 import Yolov8TestConstants, download_yolov8n_model

        # init model
        download_yolov8n_model()

        yolov8_detection_model = Yolov8DetectionModel(
            model_path=Yolov8TestConstants.YOLOV8N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        yolov8_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            images=image_path,
            detection_model=yolov8_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type=postprocess_type,
            postprocess_match_threshold=match_threshold,
            postprocess_match_metric=match_metric,
            postprocess_class_agnostic=class_agnostic,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 19)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 2
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 19)
