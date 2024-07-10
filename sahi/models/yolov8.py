# OBSS SAHI Tool
# Code written by AnNT, 2023.

import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from sahi.models.yolov8plus import Yolov8PlusDetectionModel


class Yolov8DetectionModel(Yolov8PlusDetectionModel):
    def check_dependencies(self) -> None:
        return super().check_dependencies()

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        return super().load_model()

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv8 model.
        Args:
            model: Any
                A YOLOv8 model
        """

        return super().set_model(model)

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        If predictions have masks, each prediction is a tuple like (boxes, masks).
        Args:
            image: np.ndarray or list of np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.

        """

        return super().perform_inference(image)

    @property
    def category_names(self):
        return self.model.names.values()

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return self.model.overrides["task"] == "segment"

    def _create_object_prediction_list_from_original_predictions(
        self, shift_amount_list: List[List[int]] | None = ..., full_shape_list: List[List[int]] | None = None
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """

        return super()._create_object_prediction_list_from_original_predictions(shift_amount_list, full_shape_list)
