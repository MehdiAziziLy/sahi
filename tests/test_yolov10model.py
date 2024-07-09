# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.yolov10 import Yolov10TestConstants, download_yolov10n_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 640


class TestYolov10DetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.models.yolov10 import Yolov10DetectionModel

        download_yolov10n_model()

        yolov10_detection_model = Yolov10DetectionModel(
            model_path=Yolov10TestConstants.YOLOV10N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(yolov10_detection_model.model, None)

    def test_set_model(self):
        from ultralytics import YOLO

        from sahi.models.yolov10 import Yolov10DetectionModel

        download_yolov10n_model()

        yolo_model = YOLO(Yolov10TestConstants.YOLOV10N_MODEL_PATH)

        yolov10_detection_model = Yolov10DetectionModel(
            model=yolo_model,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(yolov10_detection_model.model, None)

    def test_convert_original_predictions(self):
        from sahi.models.yolov10 import Yolov10DetectionModel

        # init model
        download_yolov10n_model()

        yolov10_detection_model = Yolov10DetectionModel(
            model_path=Yolov10TestConstants.YOLOV10N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get raw predictions for reference
        original_results = yolov10_detection_model.model.predict(image_path, conf=CONFIDENCE_THRESHOLD)[0].boxes
        num_results = len(original_results)

        # perform inference
        yolov10_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        yolov10_detection_model.convert_original_predictions()
        object_prediction_list = yolov10_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), num_results)

        # loop through predictions and check that they are equal
        for i in range(num_results):
            desired_bbox = [
                original_results[i].xyxy[0][0],
                original_results[i].xyxy[0][1],
                original_results[i].xywh[0][2],
                original_results[i].xywh[0][3],
            ]
            desired_cat_id = int(original_results[i].cls[0])
            self.assertEqual(object_prediction_list[i].category.id, desired_cat_id)
            predicted_bbox = object_prediction_list[i].bbox.to_xywh()
            margin = 2
            for ind, point in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        for object_prediction in object_prediction_list:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
