import zipfile
import os
from typing import List

import yaml


def get_list_of_folders(path: str) -> List[str]:
    """
    Returns a list of subdirectories in a directory
    :param path: path to directory
    :return: list of subdirectories
    """
    return [f.name for f in os.scandir(path) if f.is_dir()]


def get_list_of_completed_experiments(path: str) -> List[str]:
    """
    Returns a list of subdirectories that contain a 'data.zip' file
    :param path: path to directory
    :return: list of subdirectories
    """
    return [f.name for f in os.scandir(path) if f.is_dir() and os.path.isfile(os.path.join(f.path, 'data.zip'))]


def annotation_status(directory: str) -> str:
    """
    Returns the status of an experiment's annotations.
    :param directory: path to experiment directory
    :return: 'bbox_only', 'bbox_sam', or 'complete'
    """
    with zipfile.ZipFile(os.path.join(directory, 'data.zip'), 'r') as zip_ref:
        if 'annotations.yaml' in zip_ref.namelist():
            with zip_ref.open('annotations.yaml') as f:
                annotations = yaml.load(f, Loader=yaml.FullLoader)
                if 'sam_mask' in annotations[0]:
                    return 'bbox_sam'
                elif 'seg_mask' in annotations[0]:
                    return 'complete'
                else:
                    return 'bbox_only'


def get_experiments_with_incomplete_annotations(path: str) -> List[str]:
    """
    Returns a list of subdirectories that contain a 'data.zip' file but are not complete.
    :param path: path to directory
    :return: list of subdirectories
    """
    return [f.name for f in os.scandir(path)
            if f.is_dir()
            and os.path.isfile(os.path.join(f.path, 'data.zip'))
            and annotation_status(f.path) != 'complete']


if __name__ == '__main__':
    # path to data folder
    data_path = r'data/examples'

    # get list of subdirectories
    folders = get_list_of_folders(data_path)

    # get list of completed experiments
    completed_experiments = get_list_of_completed_experiments(data_path)

    # get list of incomplete experiments
    incomplete_experiments = get_experiments_with_incomplete_annotations(data_path)

    print(folders)
    print(completed_experiments)
    print(incomplete_experiments)
