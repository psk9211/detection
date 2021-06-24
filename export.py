"""
Convert pytorch pth model to TorchScript file using tracing
"""

import sys
import os
import os.path
import argparse
import copy

import torch
import torch.quantization.quantize_fx as quantize_fx
import torch.quantization as quantization
import torchvision

"""
__all__ = [
    "FasterRCNN", "fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn"
]
__all__ = [
    "MaskRCNN", "maskrcnn_resnet50_fpn",
]
__all__ = [
    "RetinaNet", "retinanet_resnet50_fpn"
]
__all__ = ['SSD', 'ssd300_vgg16']
__all__ = ['ssdlite320_mobilenet_v3_large']

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FX Quantization Using PyTorch')
    
    parser.add_argument('--data-dir', type=str, help='Dataset for calibration', 
                        default='/home/workspace/datasets/coco')
    parser.add_argument('--model', type=str, help='Specifiy architecture')
    parser.add_argument('--output', type=str, default='./quantized',
                        help='Path where TorchScript output model will be stored')
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    args = parser.parse_args()

    model = torchvision.models.detection.__dict__[args.model](pretrained=args.pretrained)
    model.to('cpu')

    #
    # post training dynamic/weight_only quantization
    #

    # we need to deepcopy if we still want to keep model_fp unchanged after quantization since quantization apis change the input model
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.eval()
    
    model_dynamic = quantization.quantize_dynamic(model_to_quantize, dtype=torch.qint8)
    model_dynamic.eval()

    trace

    