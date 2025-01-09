1) Dataset: Custom Odometer Dataset (CQ)
Dataset Type: Image-based object detection (Odometer readings as the object class)
Number of Classes: 1 (odometer)
Data Split: Training, Validation, and Test datasets are organized into respective folders.
------------------------------------------------------------------------------------------
2) The data annotations are parsed from the via_region_data.json file.
The labels are converted into YOLO format (class, x_center, y_center, width, height).
Images are resized to a consistent input size suitable for training (1024x1024).
------------------------------------------------------------------------------------------
3.1) Hyperparametrs for the YOLOv5 model:

lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.2
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
-------------------------------------------------------------------------------------------
3.2) Training Configuration for YOLOv5Model

Epochs: 100
Batch Size: 16
Input Image Size: 1024x1024
Optimizer: SGD (Stochastic Gradient Descent) with momentum
Scheduler: Cosine Annealing (for learning rate decay)
Early Stopping: Applied to avoid overfitting (if the validation mAP doesn’t improve for several epochs).
-------------------------------------------------------------------------------------------
3.3) Command to Run Inference: (YOLOv5 model) (Only for debugging)

```python detect.py --weights runs/train/exp3/weights/best.pt --img 1024 --conf-thres 0.5 --iou-thres 0.4 --source ../CQ_CustomTest/ --save-txt```
-------------------------------------------------------------------------------------------
4.1) Vision Transformer (ViT) Encoder: The ViT model is used for extracting visual features from images by dividing them into patches, processing them using self-attention, and then using the output for downstream tasks.

ViT Encoder Configuration:

{
  "attention_probs_dropout_prob": 0.0,
  "encoder_stride": 16,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 1024,
  "image_size": 384,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "model_type": "vit",
  "num_attention_heads": 16,
  "num_channels": 3,
  "num_hidden_layers": 24,
  "patch_size": 16,
  "qkv_bias": false,
  "transformers_version": "4.47.1"
}

The ViT encoder works by dividing the input image into non-overlapping patches, linearly embedding each patch, and passing these embeddings through a multi-layer transformer network to capture spatial relationships and global features.
-------------------------------------------------------------------------------------------

4.2) TrOCR Decoder

The TrOCR (Transformer OCR) decoder is designed for sequence generation tasks, specifically for Optical Character Recognition (OCR). It decodes the features extracted by the ViT encoder and generates text sequences (e.g., detecting text in images).

{
  "activation_dropout": 0.0,
  "activation_function": "relu",
  "add_cross_attention": true,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "classifier_dropout": 0.0,
  "cross_attention_hidden_size": 1024,
  "d_model": 1024,
  "decoder_attention_heads": 16,
  "decoder_ffn_dim": 4096,
  "decoder_layerdrop": 0.0,
  "decoder_layers": 12,
  "decoder_start_token_id": 2,
  "dropout": 0.1,
  "eos_token_id": 2,
  "init_std": 0.02,
  "is_decoder": true,
  "layernorm_embedding": false,
  "max_position_embeddings": 1024,
  "model_type": "trocr",
  "pad_token_id": 1,
  "scale_embedding": true,
  "tie_word_embeddings": false,
  "transformers_version": "4.47.1",
  "use_cache": false,
  "use_learned_position_embeddings": false,
  "vocab_size": 50265
}

This TrOCR decoder is tasked with converting the encoder's visual feature map into a sequence of tokens (odometer readings), making it suitable for OCR tasks.
-------------------------------------------------------------------------------------------

I have also attached the .ipynb files I have used for both training and inference for further reference.