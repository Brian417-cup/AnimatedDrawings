#convert into onnx
python pytorch2onnx.py --config ../sketch_config/config.py --checkpoint ../sketch_weight/best_AP_epoch_72.pth --output-file output/human_skeleton.onnx
#remove warning
python remove_initializer_from_input.py --input output/human_skeleton.onnx --output output/human_skeleton.onnx