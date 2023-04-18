import json
from pathlib import Path
import zipfile

import numpy as np
import torch
import cv2
from tqdm import tqdm

from segment_anything.utils.transforms import ResizeLongestSide

from get_sam_model import get_sam_model
from get_unannotated_experiments import get_experiments_with_incomplete_annotations


def binary_mask_to_polygon(mask: torch.tensor):
    """
    Convert a binary mask to a polygon.
    :param mask: Binary mask
    :return: list of the polygon coordinates
    """
    contours, _ = cv2.findContours(mask.numpy().astype(np.uint8)[0,:,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contour to list of tuples of coordinates
    contours = [c.reshape(-1, 2).tolist() for c in contours]
    return contours[0]


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()


class Experiment:
    """
    Class to represent an experiment. Contains images and annotations for a single experiment.
    Used to update the data folder with missing annotations.
    """

    def __init__(self, folder):
        self.annotations = None
        self.zip_path: Path = Path(folder + "data.zip")
        self.folder: str = folder
        self.rgb: np.ndarray = np.array([])

        self.load_json()

    def load_json(self):
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            annotations_json = zip_ref.read("annotations.json").decode("utf-8")

        # load json from string
        content = json.loads(annotations_json)
        self.__dict__.update(content)

    # def save_json(self):
    #     """
    #     Update the json file in the zip file with the current state of the object.
    #     :return:
    #     """
    #     rem_list = ['built_path', 'folder', 'rgb']
    #     content = {key: self.__dict__[key] for key in self.__dict__ if key not in rem_list}
    #
    #     with zipfile.ZipFile(self.built_path, 'w') as myzip:
    #         zip_contents = myzip.infolist()
    #         for item in zip_contents:
    #             if item.filename.endswith('.json'):
    #                 updated_json_string = json.dumps(content)
    #                 updated_json_info = zipfile.ZipInfo(item.filename)
    #                 updated_json_info.compress_type = zipfile.ZIP_DEFLATED
    #                 myzip.writestr(updated_json_info, updated_json_string)
    #
    #             else:
    #                 file_data = myzip.read(item.filename)
    #                 myzip.writestr(item, file_data)

    def save_json(self):
        """Saves the annotations.json file to the experiment folder"""
        rem_list = ['zip_path', 'folder', 'rgb']
        content = {key: self.__dict__[key] for key in self.__dict__ if key not in rem_list}

        # create annotations.json with content in experiment folder
        with open(self.folder + "annotations.json", "w") as f:
            json.dump(content, f)

    def get_rgb(self, name: str = "mixed_0") -> None:
        """
        Get an RGB image from the experiment. Extracted from the .zip file.
        :param name: Name of the image to extract
        :return: None
        """
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            # extract annotations.yaml as a string
            rgb = zip_ref.read(f"{name}.png")

        rgb = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
        self.rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        return self.rgb

    def check_status(self) -> bool:
        """
        Check if the experiment is complete.
        :return: True if complete, False if not
        """
        for annotation in self.annotations:
            if len({'label', 'bbox', 'sam_mask', 'seg_mask'} & set(annotation.keys())) == 4:
                return True
        return False

    def add_sam_masks(self, masks) -> None:
        for i, annotation in enumerate(self.annotations):
            annotation['sam_mask'] = binary_mask_to_polygon(masks[i])
        self.status = 'bbox_sam_mask'

    def get_bboxs(self) -> torch.tensor:
        bboxs = [annotation['bbox'] for annotation in self.annotations]
        bboxs_array = torch.tensor(np.array(bboxs), device=device)
        return bboxs_array


if __name__ == "__main__":
    base_path = 'data/examples/'
    device = 'cpu' # 'cuda:0'
    batch_size = 2

    experiment_files = get_experiments_with_incomplete_annotations(base_path)

    sam = get_sam_model(device=device)

    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    for i in tqdm(range(0, len(experiment_files), batch_size)):
        batch = experiment_files[i:i + batch_size]
        batch = [Experiment(base_path + file + '/') for file in batch]

        batched_input = [
            {'image': prepare_image(experiment.get_rgb(), resize_transform, sam),
             'boxes': experiment.get_bboxs(),
             'original_size': experiment.rgb.shape[:2]} for experiment in batch
        ]

        batched_output = sam(batched_input, multimask_output=False)

        for i, experiment in enumerate(batch):
            experiment.add_sam_masks(batched_output[i]['masks'])
            experiment.save_json()
