# Masterarbeit Public
Codes used for my master thesis "Evaluation of XAI-Algorithms Using Neural Networks"


# Training Process

The used models are mostly from torchvision with slight modifications to e.g. the classification head.
You can find the models in [model.py](https://github.com/Lunnaris01/Masterarbeit_Public/blob/main/Code/model.py)

[training.py](https://github.com/Lunnaris01/Masterarbeit_Public/blob/main/Code/training.py) is used for training. 

# XAI Heatmaps

Each XAI method has it's own "pipeline" like [gradcam_pipeline.py](https://github.com/Lunnaris01/Masterarbeit_Public/blob/main/Code/gradcam_pipeline.py)

# Occlusion and Revelation Dataset.

The datasets are created using the [occlusion_revelation_dataset_creation.py](https://github.com/Lunnaris01/Masterarbeit_Public/blob/main/Code/occlusion_revelation_dataset_creation.py) file. One can easily expand it for different experiments like occlusion with different colors or noise! [commands_pipeline.sh ](https://github.com/Lunnaris01/Masterarbeit_Public/blob/main/Code/commands_pipeline.sh) can be used and provides a guideline for the workflow.
