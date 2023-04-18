# SAM Segmentation

This repository contains three Python scripts: get_sam_masks.py, get_sam_model.py, and get_unannotated_experiments.py.

## Script Descriptions

- get_sam_masks.py: This script extracts RGB images and bounding boxes from a JSON file, and uses them to create segmentation masks using the Segment Anything Model (SAM) from Facebook. The resulting segmentation masks are saved into an annotations.yaml file in the respective subdirectory of the data folder.
- get_sam_model.py: This script loads the pre-trained SAM model from Facebook.
- get_unannotated_experiments.py: This script scans the data folder and returns the names of the subdirectories that do not contain a SAM segmentation mask in the annotations.json file.

## Installation
1. Clone this repository to your local machine.
2. Install the required packages using pip install -r requirements.txt.
3. Download the pre-trained [SAM model checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints) "sam_vit_h_4b8939" from meta and place it in the model/checkpoint directory of this repository.

## Usage
### Running get_sam_masks.py
1. Make sure the data folder is properly structured. Each subdirectory in data should contain an RGB image file named "mixed_0" and a JSON file with named "annotations.json" in the "data.zip"-folder.
This is project specific, so you will have to adapt the script to your needs if you want to use it for your own data.
2. Open a terminal window and navigate to the root directory of this repository.
3. Run the command python get_sam_masks.py.

### Running get_unannotated_experiments.py
1. Open a terminal window and navigate to the root directory of this repository.
2. Run the command python get_unannotated_experiments.py.

## License
This repository is licensed under the MIT License. See the LICENSE file for details.





