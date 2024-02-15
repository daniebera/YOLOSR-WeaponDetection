# YOLOSR: Enhancing Weapon Detection in Video Surveillance with Super Resolution

## Overview
This repository is dedicated to the code for our latest research presented in "Edge AI and Super-Resolution for Enhanced Weapon Detection in Video Surveillance", submitted for review to Engineering Applications of Artificial Intelligence. Our work combines the strengths of the new YOLOv8-small with Enhanced Deep Super Resolution (EDSR) for improved detection of handheld weapons in video surveillance data. Our approach addresses the challenge of detecting small-sized weapons by integrating a Super Resolution (SR) branch during the model training phase, which is subsequently removed during inference to maintain low computational cost.

**Note:** The codebase will be made available here following the paper's acceptance. We thank you for your interest and patience.

## Paper Abstract
The paper presents YOLOSR, a novel deep learning model that enhances the detection of small-sized weapons in video surveillance footage. By incorporating a Super Resolution (SR) branch during training, our model significantly improves detection accuracy without increasing computational demands during inference, offering an optimal solution for real-time weapon detection in resource-constrained environments.

## Citation
Please cite our work using the following format if it helps or inspires your research:

Details will be provided upon paper acceptance.

## Model Architecture
YOLOSR utilizes YOLOv8-small for weapon detection, augmented with an EDSR-based Super Resolution branch during training. This dual-branch architecture enhances feature representation of small objects, crucial for detecting weapons in surveillance videos.

Diagrams and flowcharts will be included upon paper acceptance.

## Datasets
Our model was validated on the custom-built WeaponSense dataset, showcasing various indoor scenarios with subjects holding weapons. Further details on dataset preparation and preprocessing will be shared alongside the code release.

## Results
YOLOSR demonstrates superior performance in weapon detection accuracy compared to existing methods, achieving significant improvements in Average Precision (AP50) without increasing the computational load on edge devices like NVIDIA Jetson Nano.

Tables, graphs, and charts will be included upon paper acceptance.

## Dependencies
- Python 3.8+
- PyTorch 1.12+
- CUDA (for NVIDIA GPU support)
- Additional requirements will be listed in `requirements.txt`

## Installation
Installation instructions will be provided, including steps for setting up the environment and dependencies necessary to run YOLOSR.

## Usage
This section will be updated once the code is released.

## Contributing
Contributions to improve the model or extend its capabilities are welcome.

## License
Details on the licensing of the software will be specified upon code release.

## Acknowledgments
This code is built on the [YOLOv8 GitHub repository](https://github.com/ultralytics/ultralytics). We thank the authors for sharing the codes.

## Contact
For inquiries related to the paper or the forthcoming code, please contact:

- Daniele Berardini, [d.berardini@pm.univpm.it]
