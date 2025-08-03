# Medical Image Enhancement with GAN

## Table of contents

- [Description](#description)
- [Setup](#setup)
- [Running the Files](#running-the-files)
- [License](#license)
- [Credits](#credits)

## Description

This project allows the enhancement of low-quality Diffusion-Weighted MRI brain scans using a Pix2Pix GAN model. It includes a Streamlit-based web app for image comparison and metric evaluation. 

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Jeffrey2177/Capstone_AI_Image_Enhancement.git
cd Capstone_AI_Image_Enhancement

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit web app
streamlit run app.py

```

## Running the Files

## How to use


## License 

This project is designed to be open source and available under the MIT License

## Credits

This project builds upon the foundational work of [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), an open-source PyTorch implementation of CycleGAN and Pix2Pix by Jun-Yan Zhu and collaborators.  
Their repository provided the base architecture and training framework for the GAN model used in this project.

Additional contributions and customisations were made for:
- Diffusion-Weighted MRI data handling
- Synthetic degradation and enhancement workflows
- Web-based visualization and evaluation
