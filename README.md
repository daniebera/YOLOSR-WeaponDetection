# YOLOSR: Enhancing Weapon Detection in Video Surveillance with Super Resolution

## Overview
This repository contains the code for our research article, "**Edge Artificial Intelligence and Super-Resolution for Enhanced Weapon Detection in Video Surveillance**," published in *Engineering Applications of Artificial Intelligence* by Elsevier. 

Our work introduces YOLOSR, a novel framework that integrates YOLOv8 with Enhanced Deep Super Resolution (EDSR) to improve the detection of small-sized weapons in video surveillance footage. By incorporating a Super Resolution (SR) branch during training and removing it during inference, YOLOSR achieves significant accuracy improvements while maintaining low computational cost, making it ideal for real-time edge applications.

## Paper Abstract
The paper presents YOLOSR, a deep learning model designed to enhance the detection of small-sized weapons in video surveillance footage. By integrating a Super Resolution (SR) branch during training, the model significantly improves detection accuracy without increasing computational demands during inference, offering an optimal trade-off for real-time weapon detection in resource-constrained environments.

## Citation
If you find our work helpful or inspiring, please cite it using the following format:

```bibtex
@article{berardini2025edge,
  title={Edge Artificial Intelligence and Super-Resolution for Enhanced Weapon Detection in Video Surveillance},
  author={Berardini, Daniele and Migliorelli, Lucia and Galdelli, Alessandro and Marín-Jiménez, Manuel J.},
  journal={Engineering Applications of Artificial Intelligence},
  volume={140},
  pages={109684},
  year={2025},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.engappai.2024.109684}
}
```

## Model Architecture
YOLOSR utilizes YOLOv8-small for weapon detection, enhanced with an EDSR-based Super Resolution auxiliary branch during training. This dual-branch architecture improves feature representation of small objects, which is crucial for detecting weapons in surveillance videos. 

Diagrams and detailed architectural descriptions are provided in the [paper](https://doi.org/10.1016/j.engappai.2024.109684).

## Datasets
Our model was validated on the custom-built WeaponSense dataset, which includes various indoor scenarios with subjects holding weapons. To request access to the WeaponSense dataset for research purposes, please complete and return the following [Data Request Form](https://univpm-my.sharepoint.com/:b:/g/personal/p018352_staff_univpm_it/EeDprfkt-BFHpsa8dFfc_H4BspcuH0JA_4sNzbKFa7g7NQ?e=N4K3qM).

## Results
YOLOSR demonstrates superior performance in weapon detection accuracy compared to existing methods. It achieves significant improvements in Average Precision (AP50) without increasing the computational load on edge devices like the NVIDIA Jetson Nano. 

Detailed results, including tables and charts, are available in the [paper](https://doi.org/10.1016/j.engappai.2024.109684).

## Dependencies
- Python 3.8+
- PyTorch 1.12+
- CUDA (for NVIDIA GPU support)
- Additional requirements are listed in `requirements.txt`, reflecting the Ultralytics requirements.

## Installation
To set up the environment and dependencies, you can choose between `venv` or `conda` for creating a virtual environment:

### Option 1: Using `venv`
1. Clone this repository:
   ```bash
   git clone https://github.com/daniebera/YOLOSR-WeaponDetection.git
   cd YOLOSR-WeaponDetection
   ```
2. Create and activate a virtual environment (N.B. tested with Python 3.8+):
   ```bash
   python -m venv yolosr_env
   source yolosr_env/bin/activate   # On Windows: yolosr_env\Scripts\activate
   ```
3. Install the repository in editable mode:
   ```bash
   pip install -e .
   ```

### Option 2: Using `conda`
1. Clone this repository:
   ```bash
   git clone https://github.com/daniebera/YOLOSR-WeaponDetection.git
   cd YOLOSR-WeaponDetection
   ```
2. Create and activate a `conda` environment with Python 3.8:
   ```bash
   conda create -n yolosr_env python=3.8 -y
   conda activate yolosr_env
   ```
3. Install the repository in editable mode:
   ```bash
   pip install -e .
   ```
## Usage
To train the model with the Super Resolution (SR) feature enabled, use the following command:

```bash
yolo detect train data=<your_dataset>.yaml model=yolov8s.yaml epochs=300 imgsz=640 batch=32 project=runs/yolo8s/ name=YOLOSR sr=True
```

This command is similar to the standard Ultralytics training command but includes an additional flag:
- `sr=True`: Enables the SR branch during training.

### Notes:
1. **Image Size:** The training code assumes that the original image size is at least **1280 pixels** for its larger dimension. Using smaller images may lead to degradation in detection performance due to insufficient resolution for Super Resolution enhancements. 

	You can tweak this behavior by changing the `hr_imgsz` parameter (default: `1280`) and adjusting the `imgsz` parameter accordingly, ensuring `imgsz` is **half of `hr_imgsz`**. For example:

- For an `hr_imgsz` of 1024, set:
  ```bash
  yolo detect train data=<dataset>.yaml model=yolov8s.yaml epochs=300 imgsz=512 batch=32 project=runs/yolo8s/ name=YOLOSR sr=True hr_imgsz=1024
  ```
	This flexibility allows training on images with smaller resolutions, though the performance may still vary based on the dataset characteristics.

2. **Model Architecture:**  
   The architecture can be modified by editing the `yolov8.yaml` file. For example, changing the model size from **small (s)** to **medium (m)** or **large (l)** requires updates to the corresponding `.yaml` file. Basic instructions for customizing channels and parameters are provided as comments within the file. Ensure you adjust the architecture appropriately to fit your use case and computational resources.

## License
This repository is licensed under the [MIT License](LICENSE).

## Acknowledgments
This code builds on the [YOLOv8 GitHub repository](https://github.com/ultralytics/ultralytics). Special thanks to the authors for sharing their codebase.

## Contact
For inquiries related to the paper or this repository, feel free to contact:

- Daniele Berardini, [d.berardini@univpm.it]