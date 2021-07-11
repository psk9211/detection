python main.py --data-dir='/home/workspace/datasets/coco' \
    --fp-model='/home/psk/projects/detectron2/tools/deploy/faster_rcnn_R_50_FPN_3x/model.pth'

python eval.py --custom

python eval.py --pretrained --model fasterrcnn_resnet50_fpn

python main.py --model-fp fasterrcnn_resnet50_fpn --quant --eval --pretrained

python quantize_od_models.py --quant --model ./fasterrcnn_resnet50_fpn_coco-258fb6c6.pth 



# quantize CNN models
python quantize_cnn_models.py --pretrained --quant --mode static --model resnet18 --output ./pth/resnet18_static.pth
python quantize_cnn_models.py --eval --model-file ./pth/resnet18_static.pth --model resnet18
python quantize_cnn_models.py --eval --model-file /home/psk/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth --pretrained --model resnet18

python quantize_cnn_models.py --pretrained --quant --mode static --model resnet50 --output ./pth/resnet50_static.pth
python quantize_cnn_models.py --eval --model-file ./pth/resnet50_static.pth --model resnet50
python quantize_cnn_models.py --eval --model-file /home/psk/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth --pretrained --model resnet50
