python main.py --data-dir='/home/workspace/datasets/coco' \
    --fp-model='/home/psk/projects/detectron2/tools/deploy/faster_rcnn_R_50_FPN_3x/model.pth'

python eval.py --custom

python eval.py --pretrained --model fasterrcnn_resnet50_fpn

python main.py --model-fp fasterrcnn_resnet50_fpn --quant --eval --pretrained