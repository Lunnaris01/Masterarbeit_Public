python gradcam_pipeline.py --model_name "simple_cnn" --model_checkpoint "model/simple_cnn_1_1.tar" --target_layer "conv3"
python PRISM_pipeline.py --model_name "simple_cnn" --model_checkpoint "model/simple_cnn_1_1.tar"
python lime_pipeline.py --model_name "simple_cnn" --model_checkpoint "model/simple_cnn_1_1.tar"
python XRAI_pipeline.py --model_name "simple_cnn" --model_checkpoint "model/simple_cnn_1_1.tar"
python IGF_pipeline.py --model_name "simple_cnn" --model_checkpoint "model/simple_cnn_1_1.tar"

python gradcam_pipeline.py --model_name "resnet50" --model_checkpoint "model/resnet50_1_1.tar" --target_layer "layer4.-1.conv3"
python PRISM_pipeline.py --model_name "resnet50" --model_checkpoint "model/resnet50_1_1.tar"
python lime_pipeline.py --model_name "resnet50" --model_checkpoint "model/resnet50_1_1.tar"
python XRAI_pipeline.py --model_name "resnet50" --model_checkpoint "model/resnet50_1_1.tar"
python IGF_pipeline.py --model_name "resnet50" --model_checkpoint "model/resnet50_1_1.tar"

python gradcam_pipeline.py --model_name "convnext_tiny" --model_checkpoint "model/convnext_tiny_1_1.tar" --target_layer "features.-1.-1.block.0"
python PRISM_pipeline.py --model_name "convnext_tiny" --model_checkpoint "model/convnext_tiny_1_1.tar"
python lime_pipeline.py --model_name "convnext_tiny" --model_checkpoint "model/convnext_tiny_1_1.tar"
python XRAI_pipeline.py --model_name "convnext_tiny" --model_checkpoint "model/convnext_tiny_1_1.tar"
python IGF_pipeline.py --model_name "convnext_tiny" --model_checkpoint "model/convnext_tiny_1_1.tar"

python gradcam_pipeline.py --model_name "vgg16" --model_checkpoint "model/vgg16_1_1.tar" --target_layer "features.-3"
python PRISM_pipeline.py --model_name "vgg16" --model_checkpoint "model/vgg16_1_1.tar"
python lime_pipeline.py --model_name "vgg16" --model_checkpoint "model/vgg16_1_1.tar"
python XRAI_pipeline.py --model_name "vgg16" --model_checkpoint "model/vgg16_1_1.tar"
python IGF_pipeline.py --model_name "vgg16" --model_checkpoint "model/vgg16_1_1.tar"

python heatmap_smoothing.py
python occlusion_revelation_dataset_creation.py
# Heatmap Smoothing and occlusion_revelation_dataset_creation by default runs for every single model and every single xai method covered in the thesis.
# If you only want to run it for certain methods/models use --model_type simple_cnn convnext_tiny --xai_type gradcam lime prism

