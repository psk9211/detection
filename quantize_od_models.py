import argparse
import logging

import torch

import models
from coco_utils import get_coco, get_coco_kp
from engine import evaluate

import presets
import utils


pretrained_model= [
    "fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn",
    "retinanet_resnet50_fpn",
    "ssd300_vgg16", "ssdlite320_mobilenet_v3_large",
    "maskrcnn_resnet50_fpn", "keypointrcnn_resnet50_fpn"
]


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
    dataset, _ = get_dataset('coco', "val", get_transform(train=False), args.data_dir)
    #dataset_train, _ = get_dataset('coco', "train", get_transform(train=True), args.data_dir)

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        sampler=test_sampler, num_workers=4,
        collate_fn=utils.collate_fn)

    # Get model

    if args.quant:
        model = models.__dict__[args.arch].__dict__[args.backend](pretrained=args.pretrained, pretrained_backbone=False)
        if not args.pretrained:
            model.load_state_dict(torch.load(args.model_file))
        model.to('cpu')
        torch.backends.quantized.engine = 'qnnpack'
        model.eval()

        if args.mode =='static':
            model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer)
            
            mainlogger.info('Fusing model...')
            model.fuse_model()
            torch.quantization.prepare(model, inplace=True)
            
            #calibration for static
            mainlogger.info('Begin calibration...')
            evaluate(model, data_loader, device='cpu')

            torch.quantization.convert(model, inplace=True)
            torch.jit.save(torch.jit.script(model), args.output)
            mainlogger.info(f'Quantized model saved at: {args.output}')
            mainlogger.debug(f'Static quantized model info: \n {model}')
        
        elif args.mode == 'dynamic':
            q_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Conv2d, torch.nn.Linear}
            )
            print(q_model)
            evaluate(q_model, data_loader, device='cpu')
            torch.jit.save(torch.jit.script(model), args.output)
            mainlogger.info(f'Quantized model saved at: {args.output}')
            mainlogger.debug(f'Dynamic quantized model info: \n {model}')

    if args.eval:
        if args.pretrained:
            model = models.__dict__[args.arch].__dict__[args.backend](pretrained=True, pretrained_backbone=False)
        else:
            if args.ts:
                model = torch.jit.load(args.model_file)
                evaluate(model, data_loader, device='cpu', type='ts')
                return
            else:
                model = models.__dict__[args.arch].__dict__[args.backend](pretrained=False, pretrained_backbone=False)
                model.load_state_dict(torch.load(args.model_file))

        evaluate(model, data_loader, device='cpu', type='pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/workspace/datasets/coco')
    parser.add_argument('--output', type=str, help='Path for the output quantized model')
    parser.add_argument('--model-file', type=str, help='Model file path')
    
    parser.add_argument('--arch', type=str, help='Model architechrue', default='faster_rcnn')
    parser.add_argument('--backend', type=str, help='Model backend', default='fasterrcnn_resnet50_fpn')

    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--mode', type=str, default='static')

    parser.add_argument('--ts', action='store_true')
    
    parser.add_argument('--logging-level', type=str,
                        help='Define logging level: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]',
                        default='DEBUG')

    args = parser.parse_args()

    # Set Logger & Handlers
    mainlogger = logging.getLogger()
    mainlogger.setLevel(logging.INFO)

    log_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
                                    datefmt="%m-%d %H:%M")
    
    console = logging.StreamHandler()
    console.setLevel(args.logging_level)
    console.setFormatter(log_formatter)
    
    mainlogger.addHandler(console)

    main(args)