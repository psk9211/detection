import sys
import os
import os.path
import argparse
import numpy
import pdb
import torch
import torch.quantization.quantize_fx as quantize_fx
import torch.quantization as quantization
import torchvision

import onnx
import onnxruntime as ort

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FX Quantization Using PyTorch')
    
    parser.add_argument('--data-dir', type=str, help='Dataset for calibration', 
                        default='/home/workspace/datasets/coco')
    parser.add_argument('--model', type=str, help='Model file')
    parser.add_argument('--output', type=str,
                        help='Path where onnx output model will be stored')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    model = torch.jit.load(args.model)
    model.eval()
    torch_outs = model(list(x))
    pdb.set_trace()
    print(torch_outs)

    torch.onnx.export(model, list(x), args.output, export_params=True,
                    opset_version=12, do_constant_folding=True,
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}}
    )

    if args.test:
        ort_session = ort.InferenceSession(args.output)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)

        print(ort_outs)


    