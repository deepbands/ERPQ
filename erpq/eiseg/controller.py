# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os.path as osp
import time
import json
import logging

import cv2
import numpy as np
from skimage.measure import label
import paddle

from .inference import clicker
from .inference.predictor import get_predictor
from .util.vis import draw_with_blend_and_clicks
from .util.polygon import get_polygon
from .util.label import LabelList
from .models import EISegModel


class InteractiveController:
    def __init__(
        self,
        predictor_params: dict = None,
        prob_thresh: float = 0.5,
    ):
        """Initialize the controller.

        Parameters
        ----------
        predictor_params : dict
            Reasoner configuration
        prob_thresh : float
            Threshold to distinguish between foreground and background results

        """
        self.predictor_params = predictor_params
        self.prob_thresh = prob_thresh
        self.model = None
        self.image = None
        # self.rawImage = None
        self.predictor = None
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.polygons = []

        # 用于redo
        self.undo_states = []
        self.undo_probs_history = []

        self.curr_label_number = 0
        self._result_mask = None
        self.labelList = LabelList()
        self.lccFilter = False
        self.log = logging.getLogger(__name__)

    def filterLargestCC(self, do_filter: bool):
        """Set whether to keep only the largest connected block in the inference result

        Parameters
        ----------
        do_filter : bool
            Whether to save only the largest connected block in the inference result
        """
        if not isinstance(do_filter, bool):
            return
        self.lccFilter = do_filter

    def setModel(self, param_path=None, use_gpu=None):
        """Set up its model for inference.

        Parameters
        ----------
        params_path : str
            Model path

        use_gpu : bool
            None:Check, according to paddle version
            bool:Whether to turn on the GPU according to the specified

        Returns
        -------
        bool, str
            Whether to successfully set up the model, Reason for failure

        """
        if param_path is not None:
            model_path = param_path.replace(".pdiparams", ".pdmodel")
            if not osp.exists(model_path):
                raise Exception(f"Not present {model_path} Locate the model file")
            if use_gpu is None:
                if paddle.device.is_compiled_with_cuda():  # TODO: Can use GPU but returns False
                    use_gpu = True
                else:
                    use_gpu = False
            tic = time.time()
            try:
                self.model = EISegModel(model_path, param_path, use_gpu)
                self.reset_predictor()  # Effective immediately
            except KeyError as e:
                return False, str(e)
            return True, "The model is set successfully"

    def setImage(self, image: np.array):
        """Set the currently marked picture

        Parameters
        ----------
        image : np.array
            The currently marked picture

        """
        if self.model is not None:
            self.image = image
            self._result_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            self.resetLastObject()

    # Label operation
    def setLabelList(self, labelList: json):
        """Set the label list, it will overwrite the existing label list

        Parameters
        ----------
        labelList : json
            The format of the tag list is
            {
                {
                    "idx" : int         (like 0 or 1 or 2)
                    "name" : str        (like "car"　or "airplan")
                    "color" : list      (like [255, 0, 0])
                },
                ...
            }

        Returns
        -------
        type
            Description of returned object.

        """
        self.labelList.clear()
        labels = json.loads(labelList)
        for lab in labels:
            self.labelList.add(lab["id"], lab["name"], lab["color"])

    def addLabel(self, id: int, name: str, color: list):
        self.labelList.add(id, name, color)

    def delLabel(self, id: int):
        self.labelList.remove(id)

    def clearLabel(self):
        self.labelList.clear()

    def importLabel(self, path):
        self.labelList.importLabel(path)

    def exportLabel(self, path):
        self.labelList.exportLabel(path)

    # Click action
    def addClick(self, x: int, y: int, is_positive: bool):
        """Add a point and run inference, save history for undo

        Parameters
        ----------
        x : int
            The abscissa of the click
        y : int
            The ordinate of the click
        is_positive : bool
            Is it punctual?

        Returns
        -------
        bool, str
            Click whether to add successfully, Reason for failure

        """

        # 1. Sure you can click
        if not self.inImage(x, y):
            return False, "Click out of bounds"
        if not self.modelSet:
            return False, "Model not loaded"
        if not self.imageSet:
            return False, "Image is not set"

        if len(self.states) == 0:  # Save an empty state
            self.states.append(
                {
                    "clicker": self.clicker.get_state(),
                    "predictor": self.predictor.get_states(),
                }
            )

        # 2. Add click, run reasoning
        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker)

        # 3. Save state
        self.states.append(
            {
                "clicker": self.clicker.get_state(),
                "predictor": self.predictor.get_states(),
            }
        )
        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][1], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        # After clicking, you can't continue the previous history redo
        self.undo_states = []
        self.undo_probs_history = []
        return True, "Click to add successfully"

    def undoClick(self):
        """
        undo一Step click
        """
        if len(self.states) <= 1:  # == 1 there is only one empty state left, no need to return
            return
        self.undo_states.append(self.states.pop())
        self.clicker.set_state(self.states[-1]["clicker"])
        self.predictor.set_states(self.states[-1]["predictor"])
        self.undo_probs_history.append(self.probs_history.pop())
        if not self.probs_history:
            self.reset_init_mask()

    def redoClick(self):
        """
        redo一Step click
        """
        if len(self.undo_states) == 0:  # If it has not been revoked
            return
        if len(self.undo_probs_history) >= 1:
            next_state = self.undo_states.pop()
            self.states.append(next_state)
            self.clicker.set_state(next_state["clicker"])
            self.predictor.set_states(next_state["predictor"])
            self.probs_history.append(self.undo_probs_history.pop())

    def finishObject(self):
        """
        End the current object labeling and prepare to label the next one
        """
        object_prob = self.current_object_prob
        if object_prob is None:
            return None, None
        object_mask = object_prob > self.prob_thresh
        if self.lccFilter:
            object_mask = self.getLargestCC(object_mask)
        polygon = get_polygon(object_mask.astype(np.uint8) * 255)
        if polygon is not None:
            self._result_mask[object_mask] = self.curr_label_number
            self.resetLastObject()
            self.polygons.append([self.curr_label_number, polygon])
        return object_mask, polygon

    # Polygon
    def getPolygon(self):
        return self.polygon

    def setPolygon(self, polygon):
        self.polygon = polygon

    # mask
    def getMask(self):
        s = self.imgShape
        img = np.zeros([s[0], s[1]])
        for poly in self.polygons:
            pts = np.int32([np.array(poly[1])])
            cv2.fillPoly(img, pts=pts, color=poly[0])
        return img

    def setCurrLabelIdx(self, number):
        if not isinstance(number, int):
            return False
        self.curr_label_number = number

    def resetLastObject(self, update_image=True):
        """
        Reset controller state
        Parameters
            update_image(bool): Whether to update the image
        """
        self.states = []
        self.probs_history = []
        self.undo_states = []
        self.undo_probs_history = []
        # self.current_object_prob = None
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()

    def reset_predictor(self, predictor_params=None):
        """
        Reset the reasoner, you can change the reasoning configuration
        Parameters
            predictor_params(dict): Inference configuration
        """
        if predictor_params is not None:
            self.predictor_params = predictor_params
        if self.model.model:
            self.predictor = get_predictor(self.model.model, **self.predictor_params)
            if self.image is not None:
                self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self.clicker.click_indx_offset = 0

    def getLargestCC(self, mask):
        mask = label(mask)
        if mask.max() == 0:
            return mask
        mask = mask == np.argmax(np.bincount(mask.flat)[1:]) + 1
        return mask

    def get_visualization(self, alpha_blend: float, click_radius: int):
        if self.image is None:
            return None
        # 1. The mask being marked
        # results_mask_for_vis = self.result_mask  # Add the previously marked mask
        results_mask_for_vis = np.zeros_like(self.result_mask)
        results_mask_for_vis *= self.curr_label_number
        if self.probs_history:
            results_mask_for_vis[
                self.current_object_prob > self.prob_thresh
            ] = self.curr_label_number
        if self.lccFilter:
            results_mask_for_vis = (
                self.getLargestCC(results_mask_for_vis) * self.curr_label_number
            )
        vis = draw_with_blend_and_clicks(
            self.image,
            mask=results_mask_for_vis,
            alpha=alpha_blend,
            clicks_list=self.clicker.clicks_list,
            radius=click_radius,
            palette=self.palette,
        )
        return vis

    def inImage(self, x: int, y: int):
        s = self.image.shape
        if x < 0 or y < 0 or x >= s[1] or y >= s[0]:
            return False
        return True

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        return result_mask

    @property
    def palette(self):
        if self.labelList:
            colors = [ml.color for ml in self.labelList]
            colors.insert(0, [0, 0, 0])
        else:
            colors = [[0, 0, 0]]
        return colors

    @property
    def current_object_prob(self):
        """
        Get the current inference label
        """
        if self.probs_history:
            _, current_prob_additive = self.probs_history[-1]
            return current_prob_additive
        else:
            return None

    @property
    def is_incomplete_mask(self):
        """
        Returns
            bool: Is the current object still marked?
        """
        return len(self.probs_history) > 0

    @property
    def imgShape(self):
        return self.image.shape[1::-1]

    @property
    def modelSet(self):
        return self.model is not None

    @property
    def modelName(self):
        return self.model.__name__

    @property
    def imageSet(self):
        return self.image is not None
