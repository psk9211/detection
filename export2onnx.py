import argparse
import pdb
import torch

import onnxruntime as ort

import models

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_cnn_model(args):
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    # Load model
    if args.pretrained:
        model = models.__dict__[args.arch](pretrained=True)
    else:
        if args.ts:
            model = torch.jit.load(args.model_file)
        else:
            model = models.__dict__[args.arch](pretrained=False)
            model.load_state_dict(torch.load(args.model_file))

    model.eval()
    torch_outs = model(list(x))
    
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
        
        print(f'Torch output: {torch_outs}')
        print(f'ORT session output: {ort_outs}')


def export_od_model(args):
    batch_size = 1
    x = torch.randn(batch_size, 3, 400, 400, requires_grad=True)

    # Load model
    if args.pretrained:
        model = models.__dict__[args.arch].__dict__[args.backend](pretrained=True, pretrained_backbone=False)
    else:
        if args.ts:
            model = torch.jit.load(args.model_file)
        else:
            model = models.__dict__[args.arch].__dict__[args.backend](pretrained=False, pretrained_backbone=False)
            model.load_state_dict(torch.load(args.model_file))

    model.eval()
    torch_outs = model(list(x))
    
    torch.onnx.export(model, list(x), args.output, export_params=True,
                    opset_version=12, do_constant_folding=True,
                    input_names = ['input'],   # the model's input names
                    output_names = ['boxes', 'scores', 'labels'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'boxes' : {0 : 'batch_size'}, 'scores' : {0 : 'batch_size'}, 
                                'labels' : {0 : 'batch_size'}}
    )

    if args.test:
        ort_session = ort.InferenceSession(args.output)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        
        print(f'Torch output: {torch_outs}')
        print(f'ORT session output: {ort_outs}')


def main(args):
    if args.type == 'cnn':
        export_cnn_model(args)
    elif args.type == 'od':
        export_od_model(args)
    else:
        print(f'{args.type}')
        raise ValueError



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX Export')

    parser.add_argument('--model-file', type=str, help='Model file')
    parser.add_argument('--output', type=str,
                        help='Path where onnx output model will be stored')
    parser.add_argument('--arch', type=str, help='Model architechrue', default='faster_rcnn')
    parser.add_argument('--backend', type=str, help='Model backend', default='fasterrcnn_resnet50_fpn')

    parser.add_argument('--type', type=str, default='cnn', help='Choose ["cnn", "od"]')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--ts', action='store_true', help='Load TorchScript model')

    args = parser.parse_args()

    main(args)




    