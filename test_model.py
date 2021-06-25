import sys
import os
import os.path
import pdb
import argparse

import torch
import torch.quantization.quantize_fx as quantize_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, get_default_qconfig

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset

from models import faster_rcnn
from coco_utils import get_coco, get_coco_kp
from engine import evaluate

import presets
import utils


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    return presets.DetectionPresetTrain() if train else presets.DetectionPresetEval()


def main(args):
    # Data loading code
    print("Loading data")
    dataset_test, num_classes = get_dataset('coco', "val", get_transform(train=False), '/home/workspace/datasets/coco')

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=4,
        collate_fn=utils.collate_fn)

    if args.quant:
        model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
        model.load_state_dict(torch.load("/home/workspace/detection/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))
        model.to('cpu')
        torch.backends.quantized.engine = 'qnnpack'
        model.eval()

        if args.mode =='static':
            model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer)
            
            model.fuse_model()
            torch.quantization.prepare(model, inplace=True)
            
            #calibration for static
            evaluate(model, data_loader_test, device='cpu')
            
            torch.quantization.convert(model, inplace=True)
            
            evaluate(model, data_loader_test, device='cpu')
            torch.jit.save(torch.jit.script(model), '/home/workspace/detection/faster_rcnn_r50_static_quant.pth')

            print(model)
        
        elif args.mode == 'dynamic':
            q_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Conv2d, torch.nn.Linear}
            )
            print(q_model)
            evaluate(q_model, data_loader_test, device='cpu')
            torch.jit.save(torch.jit.script(model), '/home/workspace/detection/faster_rcnn_r50_dynamic_quant.pth')

    if args.eval:
        model = torch.jit.load(args.model)
        evaluate(model, data_loader_test, device='cpu', type='ts')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--mode', type=str, default='static')
    parser.add_argument('--model', type=str, help='Model file path')

    args = parser.parse_args()

    main(args)