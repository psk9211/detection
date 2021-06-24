from ctypes import resize
import sys
import os
import os.path
import argparse
import copy
import tqdm
import presets
import pdb

import torch
import torch.quantization.quantize_fx as quantize_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, get_default_qconfig

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.utils.logger import setup_logger

from coco_utils import *
from coco_eval import *
from engine import evaluate
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


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91)
        #"coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(train):
    return presets.DetectionPresetTrain() if train else presets.DetectionPresetEval()


def prepare_data_loaders(data_path='/home/workspace/datasets/coco', eval_batch_size=32, dataset=None):

    tf = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize([480,640])
    ])

    if dataset is None:
        dataset_test = CocoDetection(data_path, 
                    os.path.join(data_path, 'annotations/instances_val2017.json'),
                    tf)
    else:
        dataset_test = dataset
    
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader_test


def eval_model(model, data_loader):
    model.to('cpu')
    if not args.custom:
        model.eval()

    evaluate(model, data_loader, device='cpu')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FX Quantization Using PyTorch')
    
    parser.add_argument('--data-dir', type=str, help='Dataset path', 
                        default='/home/workspace/datasets/coco')
    parser.add_argument('--model-fp', type=str, help='FP32 Model file path')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--quant', action='store_true')
    parser.add_argument("--custom", action="store_true",)
    parser.add_argument('--output', type=str, help='Path where quantized output model will be stored')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    args = parser.parse_args()

    logger = setup_logger()
    logger.setLevel('INFO')
    logger.info("Command line arguments: " + str(args))

    # Get Dataset
    dataset_test, num_classes = get_dataset('coco', "val", get_transform(train=False), args.data_dir)
    #dataset_train, num_classes = get_dataset('coco', "train", get_transform(train=True), args.data_dir)

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    #train_sampler = torch.utils.data.SequentialSampler(dataset_train)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    """
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=32,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    """
    
    print("Creating model")
    
    if args.custom:
        model_fp = torch.jit.load(args.model_fp)
    else:
        kwargs = {
            "trainable_backbone_layers": args.trainable_backbone_layers
        }
        if "rcnn" in args.model_fp:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh
        model_fp = torchvision.models.detection.__dict__[args.model_fp](num_classes=num_classes, pretrained=args.pretrained,
                                                                    **kwargs)

    logger.debug(model_fp)

    

    if args.quant:
        #
        # post training dynamic/weight_only quantization
        #

        # we need to deepcopy if we still want to keep model_fp unchanged after quantization since quantization apis change the input model
        model_to_quantize = copy.deepcopy(model_fp)
        model_to_quantize.eval()
        model_quantized = torch.quantization.quantize_dynamic(model_to_quantize, dtype=torch.qint8)
        model_quantized.eval()
        pdb.set_trace()
        logger.debug(model_quantized)

        ##########################
        # FX Quantization
        ##########################

        # Static
        qconfig = get_default_qconfig("fbgemm")
        qconfig_dict = {"": qconfig}
        prepared_model = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)

        calibrate(prepared_model, data_loader_test)
        quantized_model = quantize_fx.convert_fx(prepared_model)
        print(quantized_model)


    if args.eval:
        eval_model(quantized_model, data_loader_test)

