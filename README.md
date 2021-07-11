# Description
Object detection 모델과 CNN 모델을 TorchScript export 및 quantization 할 수 있도록 만든 코드

# 사용법

## Object Detection 모델

### Quantization
- Pretrained 모델을 사용할 떄
```cmd
python quantize_od_models.py --quant --mode='static' --pretrained \
--arch='faster_rcnn', --backend='fasterrcnn_resnet50_fpn' \
--data_dir='../datasets/coco' \
--output='./faster_rcnn_r50_static.pth'
```


- 다른 FP32 pth 파일을 사용할 때
```cmd
python quantize_od_models.py --quant --mode='static' \
--arch='faster_rcnn', --backend='fasterrcnn_resnet50_fpn' \
--data_dir='../datasets/coco' \
--model_file='./fasterrcnn_resnet50_fpn_coco-258fb6c6.pth' \
--output='./faster_rcnn_r50_static.pth'
```

### Evalutaion
- Quantization된 모델을 Evalutation
```cmd
python quantize_od_models.py --eval --model_file='./faster_rcnn_r50_static.pth' --ts \
--data_dir='../datasets/coco'
```

- FP32 모델 evaluation
```cmd
python quantize_od_models.py --eval \
--arch='faster_rcnn', --backend='fasterrcnn_resnet50_fpn' \
--model_file='./fasterrcnn_resnet50_fpn_coco-258fb6c6.pth' \
--data_dir='../datasets/coco'
```

## CNN 모델

### Quantization

```cmd
# Pretrain 모델 사용
python quantize_cnn_models.py --quant --mode='static' --arch='resnet18' --pretrained \
--output='./pth/resnet18_static.pth'

python quantize_cnn_models.py --quant --mode='static' --arch='resnet18' \
--model-file='../checkpoints/resnet50-0676ba61.pth'
--output='./pth/resnet18_static.pth'
```

### Evaluation
```cmd
# TorchScript 모델
python quantize_cnn_models.py --eval --model-file='./pth/resnet50_static.pth' --arch='resnet50'

# Pretrained 모델
python quantize_cnn_models.py --eval --model-file='/home/psk/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth' \
--pretrained --arch='resnet50'
```

## Export to ONNX
```cmd
# Export torchscript model
python export2onnx.py --type='cnn' --ts --test\
--model-file='./pth/resnet50_static.pth

# Export FP32 model
python export2onnx.py --type='cnn' --arch='resnet50' --test\
--model-file='./pth/resnet50-0676ba61.pth
```