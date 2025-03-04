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
```
python3 Xavier_pipeline_train_global.py --path_images /path_dataset/JPEGImages/ --kind_of normal --path_trt None --path_trt_descriptor None --output_path /experiments_path/prueba_torch_paper/ --kind_detection xavier --kind_descriptor xavier --version_detection yolov4_resnet50 --version_descriptor resnet --clustering_algorithm hdbscan --compression_algorithm model_umap --torch_use --path_trt_cluster None
```
### Optimize models with TensorRT

### Test pipeline:

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
