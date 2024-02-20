%cd /content
!wget https://huggingface.co/camenduru/densepose/raw/main/Base-DensePose-RCNN-FPN.yaml -O /content/Base-DensePose-RCNN-FPN.yaml
!wget https://huggingface.co/camenduru/densepose/raw/main/densepose_rcnn_R_50_FPN_s1x.yaml -O /content/densepose_rcnn_R_50_FPN_s1x.yaml
!pip install -q gradio 
!pip install -q https://github.com/camenduru/wheels/releases/download/colab/detectron2-0.6-cp310-cp310-linux_x86_64.whl
!pip install -q https://github.com/camenduru/wheels/releases/download/colab/detectron2_densepose-0.6-py3-none-any.whl

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer
from google.colab.patches import cv2_imshow

def process_image(input_image_path):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file("/content/densepose_rcnn_R_50_FPN_s1x.yaml")
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(input_image_path)
    outputs = predictor(image)['instances']
    results = DensePoseResultExtractor()(outputs)
    cmap = cv2.COLORMAP_VIRIDIS
    output_image = Visualizer(alpha=1, cmap=cv2.COLORMAP_BONE).visualize(frame, results)

    return output_image

input_image_path = "" #video path
output_image = process_image(input_image_path)

cv2_imshow("DensePose Output", output_image)
