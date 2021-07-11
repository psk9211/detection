import logging
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

import models
from utils import *

__callable_pretrained_models__ = ['googlenet', 'inception_v3', 'mobilenet_v2', 'mobilenet_v3_large',
                                'resnet18', 'resnet50', 'resnext101_32x8d',
                                'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 
                                'shufflenet_v2_x2_0']

def main(args):
    mainlogger.info("Loading data...")
    eval_batch_size = 1
    num_eval_batches = 30
    data_loader_train, data_loader_test = prepare_data_loaders(args.data_dir, eval_batch_size=eval_batch_size)

    # Eager Mode Quantization
    if args.quant:
        if args.pretrained:
            # Don't use quantized models from model zoo by using 'quantize=True' argument.
            # Pre-quantized model's backend is FBGEMM, except MobileNets, which cannot be run in ARM device.
            model = torchvision.models.quantization.__dict__[args.arch](pretrained=True, quantize=False)
        else:
            model = models.__dict__[args.arch](pretrained=False)
            model.load_state_dict(torch.load(args.model_file))
            
        model.to('cpu')
        torch.backends.quantized.engine = 'qnnpack'
        model.eval()
        
        if args.mode =='static':
            model.qconfig = torch.quantization.get_default_qconfig(backend='qnnpack')
            
            mainlogger.info('Fusing model...')
            model.fuse_model()
            #pdb.set_trace()
            torch.quantization.prepare(model, inplace=True)
            
            # calibration for static
            # Use evaluate() in utils.py (not in the one 'engine.py')
            mainlogger.info('Begin calibration...')
            evaluate(model, data_loader_train, neval_batches=100)
            
            qmodel = torch.quantization.convert(model)
            torch.jit.save(torch.jit.script(qmodel), args.output)

            mainlogger.info(f'Quantized model saved at: {args.output}')
            mainlogger.debug(f'Static quantized model info: \n {model}')            
        
        elif args.mode == 'dynamic':
            q_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Conv2d, torch.nn.Linear}
            )
            print(q_model)
            torch.jit.save(torch.jit.script(model), args.output)
            
            mainlogger.info(f'Quantized model saved at: {args.output}')
            mainlogger.debug(f'Dynamic quantized model info: \n {model}')

    # For evaluation only
    if args.eval:
        if args.pretrained:
            # Don't use quantized models from model zoo by using 'quantize=True' argument.
            # Pre-quantized model's backend is FBGEMM, except MobileNets, which cannot be run in ARM device.
            model = torchvision.models.quantization.__dict__[args.arch](pretrained=True, quantize=False)
            model.eval()
        else:
            model = torch.jit.load(args.model_file)
        mainlogger.info(f'Evaluate Model: {args.model_file}')
        run_iterative_benchmark(model, data_loader_test, num_eval_batches, eval_batch_size, name=args.arch, iter=10)
        #evaluate(model, data_loader_test)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/workspace/datasets/imagenet_1k')
    parser.add_argument('--arch', type=str, help='Model architecture')
    parser.add_argument('--model-file', type=str, help='Model file')
    parser.add_argument('--output', type=str, default=os.getcwd() + '/test.pth')

    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--mode', type=str, default='static')

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