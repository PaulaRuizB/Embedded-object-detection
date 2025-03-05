# Embedded-object-detection
Real-time unsupervised video object detection on the edge
### Prerequisites
1. Clone this repository with git
```
git clone https://github.com/PaulaRuizB/Embedded-object-detection
```
2. What you need to use the codes:
   
Python 3.8.10, TensorRT 8.4.1, cuDNN 8.4.1 and requirements video_detection_requirements.txt into venv_requirements folder

3. If you want to use the same images: https://www.airport.gdansk.pl/lotnisko/kamery-internetowe-p30.html

4. Pretrained models
   
   In pretrained_models folder:
   * best.pt: Region Proposal Network (RPN)
   * last_302.pt: Region Proposal Network (RPN) quantized to INT8
   * supcon.pth: Region Encoding Network (REN), original pretrained weights available at: https://github.com/HobbitLong/SupContrast
   * model_cluster.pth: UMAP model (UMAP-M). Note that this model is adapted to our dataset only.
   
### Train pipeline with PyTorch:
Mains folder:
```
python3 Xavier_pipeline_train_global.py --path_images /path_dataset/JPEGImages/ --kind_of normal --path_trt None --path_trt_descriptor None --output_path /experiments_path/ --kind_detection xavier --kind_descriptor xavier --version_detection yolov4_resnet50 --version_descriptor resnet --clustering_algorithm hdbscan --compression_algorithm model_umap --torch_use --path_trt_cluster None
```
### Optimize models with TensorRT
From PyTorch to ONNX (utils folder):

* Region Proposal Network
```
python3 rpn_torch_to_onnx.py --weights /path_model/
```
* Region Encoding Network
```
python3 ren_torch_to_onnx.py --weights /path_model/ --save_path /save_path/
```
* UMAP model
```
python3 cluster_torch_to_onnx.py --weights /path_model/ --save_path /save_path/
```
From ONNX to TensorRT
* FP32 GPU
```
/usr/src/tensorrt/bin/trtexec --onnx=/path_onnx_model/ --saveEngine=/path_save_trt/
```
* FP16 GPU
```
/usr/src/tensorrt/bin/trtexec --onnx=/path_onnx_model/ --saveEngine=/path_save_trt/ --fp16
```
* INT8 GPU
```
/usr/src/tensorrt/bin/trtexec --onnx=/path_onnx_model/ --saveEngine=/path_save_trt/ --int8
```
* FP32 GPU + DLA
```
/usr/src/tensorrt/bin/trtexec --onnx=/path_onnx_model/ --saveEngine=/path_save_trt/ --useDLACore=0 --allowGPUFallback
```
* FP16 GPU + DLA
```
/usr/src/tensorrt/bin/trtexec --onnx=/path_onnx_model/ --saveEngine=/path_save_trt/ --fp16 --useDLACore=0 --allowGPUFallback
```
* INT8 GPU + DLA
```
/usr/src/tensorrt/bin/trtexec --onnx=/path_onnx_model/ --saveEngine=/path_save_trt/ --int8 --useDLACore=0 --allowGPUFallback
```
### Test pipeline with optimized TensorRT models:
```
python3 TRT_Xavier_pipeline_test_global.py --path_trt /path_detector_model_trt/ --path_trt_descriptor /path_descriptor_model_trt/ --path_trt_cluster /path_umap_model/ --kind_of normal --path_images /path_dataset/JPEGImages/ --output_path /experiments_path/ --path_clusters /path_clusters/ --kind_detection trt --kind_descriptor trt --version_detection yolov4_resnet50 --version_descriptor resnet --clustering_algorithm hdbscan --compression_algorithm model_umap
```
More options:
* If you want to use DLAs (Deep Learning Accelerators) add --dla after transforming the models to be deployed on DLAs
* To measure inference time and energy consumption add --measure_gpu to measure processes deployed on GPU, --measure_cpu for CPU and --measure_dla for DLAs.

### mAP and CorLoc test:

Complete parameters list (lines 15-22) and labels_map in line 369:
```
python3 compute_metrics_corloc_map.py
```

### Our [paper](https://www.sciencedirect.com/science/article/pii/S0167739X25000329)
If you find this code useful in your research, please consider citing:

    @article{ruiz2025real,
      title={Real-time unsupervised video object detection on the edge},
      author={Ruiz-Barroso, Paula and Castro, Francisco M and Guil, Nicol{\'a}s},
      journal={Future Generation Computer Systems},
      pages={107737},
      year={2025},
      publisher={Elsevier}
    }
