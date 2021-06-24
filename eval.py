r"""PyTorch Detection Training.
To run in a multi-gpu environment, use the distributed launcher::
    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU
The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3
Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

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
    print(args)
    device = torch.device('cpu')

    # Data loading code
    print("Loading data")
    dataset_test, num_classes = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    
    if args.custom:
        model = torch.jit.load(args.model)
    else:
        kwargs = {
            "trainable_backbone_layers": args.trainable_backbone_layers
        }
        if "rcnn" in args.model:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh
        model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained,
                                                                    **kwargs)

                                                                  
    model.to(device)
    if not args.custom:
        model.eval()

    evaluate(model, data_loader_test, device=device, is_d2=args.custom)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/home/workspace/datasets/coco', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', 
                        default='/home/psk/projects/detectron2/tools/deploy/faster_rcnn_R_50_FPN_3x-script/model.ts',
                        help='model')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument("--custom", action="store_true",)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )


    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)