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

4. Region Encoding Network: ResNet50 with Supervised Contrastive Loss, pretrained weights available: https://github.com/HobbitLong/SupContrast


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
