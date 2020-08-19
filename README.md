# fGraph

Experiments with graphs of spatial temporal relationships of CNN features in video scenes

This repository contains various, loosely organized, experiments, attempts, and iterations on the same general idea, to build graphs of CNN features for tasks such as instance retrieval and object detection.

The most recent iteration is located in `src/VGG16/window_walker_a.ipynb` and `src/VGG16/vgg16_window_walker_lib_a.py`

The long term objective is an unsupervised system that if provided with unlabeled image / video data, can be queried with a sample object and should be able to return all the palaces it has seen the same instance, and potentially the same class of object. 


## Python
```
# python -V
> 3.8.5
# python -m pip install numpy
# python -m pip install scipy
# python -m pip install opencv-python
# python -m pip install tensorflow
# python -m pip install hnswlib
# python -m pip install networkx
# python -m pip install plyvel
# python -m pip install matplotlib
# python -m pip install jupyterlab
```