# SWNet
ASW-Net: A Deep Learning-based Tool for Cell Nuclei Segmentation of Fluorescence Microscopy

## Overview
ASW-Net is a deep learning-based tool for cell nucleus segmentation of fluorescence microscopy. As a simplified W-net, ASW-Net has the potential to extract more features from raw images compared with U-net, and it is lighter than W-net at the same time. The attention mechanism also endows the model with better learning ability and interpretability.
<p align="center"><img width="100%" src="pipeline.png" /></p>
The detailed structure of ASW-Net is shown as below:
<p align="center"><img width="80%" src="model.png" /></p>

## Quick Start

### Requirements
- Python 3.6+
- Keras == 2.2.4, Tensoflow == 1.14.0

### Download SWNet
```shell
git clone https://github.com/Liuzhe30/SWNet
```