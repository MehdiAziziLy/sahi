# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.yolov8 import Yolov8TestConstants, download_yolov8n_model, download_yolov8n_seg_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 640


class TestPredictBatch(unittest.TestCase):
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
        self.assertEqual(len(object_prediction_list), 13)
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
        self.assertEqual(num_car, 12)

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
        self.assertEqual(len(object_prediction_list), 13)
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
        self.assertEqual(num_car, 12)

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

        # get sliced prediction with batch
        prediction_result2 = get_sliced_prediction(
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
            batch=6,
        )
        object_prediction_list2 = prediction_result2.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list2), len(object_prediction_list))
        num_person = 0
        num_person2 = 0
        for object_prediction, object_prediction2 in zip(object_prediction_list, object_prediction_list2):
            if object_prediction.category.name == "person":
                num_person += 1
            if object_prediction2.category.name == "person":
                num_person2 += 1
        self.assertEqual(num_person, num_person2)
        num_truck = 0
        num_truck2 = 0
        for object_prediction, object_prediction2 in zip(object_prediction_list, object_prediction_list2):
            if object_prediction.category.name == "truck":
                num_truck += 2
            if object_prediction2.category.name == "truck":
                num_truck2 += 2
        self.assertEqual(num_truck, num_truck2)
        num_car = 0
        num_car2 = 0
        for object_prediction, object_prediction2 in zip(object_prediction_list, object_prediction_list2):
            if object_prediction.category.name == "car":
                num_car += 1
            if object_prediction2.category.name == "car":
                num_car2 += 1
        self.assertEqual(num_car, num_car2)

    def test_get_sliced_prediction_batch(self):
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
        image_path = ["tests/data/small-vehicles1.jpeg", "tests/data/coco_utils/terrain4.png"]

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result_batch = get_sliced_prediction(
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
            batch=6,
        )
        object_prediction_list_batch0 = prediction_result_batch[0].object_prediction_list
        object_prediction_list_batch1 = prediction_result_batch[1].object_prediction_list

        prediction_result_non_batch0 = get_sliced_prediction(
            images=image_path[0],
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
        object_prediction_list_non_batch0 = prediction_result_non_batch0.object_prediction_list

        prediction_result_non_batch1 = get_sliced_prediction(
            images=image_path[1],
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
        object_prediction_list_non_batch1 = prediction_result_non_batch1.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list_batch0), len(object_prediction_list_non_batch0))
        self.assertEqual(len(object_prediction_list_batch1), len(object_prediction_list_non_batch1))
        num_person_batch = 0
        num_person_non_batch = 0
        for object_prediction_batch, object_prediction_non_batch in zip(
            object_prediction_list_batch0, object_prediction_list_non_batch0
        ):
            if object_prediction_batch.category.name == "person":
                num_person_batch += 1
            if object_prediction_non_batch.category.name == "person":
                num_person_non_batch += 1
        self.assertEqual(num_person_batch, num_person_non_batch)

        num_truck_batch = 0
        num_truck_non_batch = 0
        for object_prediction_batch, object_prediction_non_batch in zip(
            object_prediction_list_batch0, object_prediction_list_non_batch0
        ):
            if object_prediction_batch.category.name == "truck":
                num_truck_batch += 1
            if object_prediction_non_batch.category.name == "truck":
                num_truck_non_batch += 1
        self.assertEqual(num_truck_batch, num_truck_non_batch)

        num_car_batch = 0
        num_car_non_batch = 0
        for object_prediction_batch, object_prediction_non_batch in zip(
            object_prediction_list_batch0, object_prediction_list_non_batch0
        ):
            if object_prediction_batch.category.name == "car":
                num_car_batch += 1
            if object_prediction_non_batch.category.name == "car":
                num_car_non_batch += 1
        self.assertEqual(num_car_batch, num_car_non_batch)

        num_person_batch = 0
        num_person_non_batch = 0
        for object_prediction_batch, object_prediction_non_batch in zip(
            object_prediction_list_batch1, object_prediction_list_non_batch1
        ):
            if object_prediction_batch.category.name == "person":
                num_person_batch += 1
            if object_prediction_non_batch.category.name == "person":
                num_person_non_batch += 1
        self.assertEqual(num_person_batch, num_person_non_batch)

        num_truck_batch = 0
        num_truck_non_batch = 0
        for object_prediction_batch, object_prediction_non_batch in zip(
            object_prediction_list_batch1, object_prediction_list_non_batch1
        ):
            if object_prediction_batch.category.name == "truck":
                num_truck_batch += 1
            if object_prediction_non_batch.category.name == "truck":
                num_truck_non_batch += 1
        self.assertEqual(num_truck_batch, num_truck_non_batch)

        num_car_batch = 0
        num_car_non_batch = 0
        for object_prediction_batch, object_prediction_non_batch in zip(
            object_prediction_list_batch1, object_prediction_list_non_batch1
        ):
            if object_prediction_batch.category.name == "car":
                num_car_batch += 1
            if object_prediction_non_batch.category.name == "car":
                num_car_non_batch += 1
        self.assertEqual(num_car_batch, num_car_non_batch)
