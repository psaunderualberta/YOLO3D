docker run -it --rm --gpus all -v ${PWD}:/yolo3d -it ruhyadi/yolo3d:latest

# Inference script
# python inference.py \
#     --weights yolov5s.pt \
#     --source data/KITTI/data_object_image_2/training \
#     --reg_weights weights/resnet18.pkl \
#     --model_select resnet18 \
#     --output_path runs/ \
#     --save_result