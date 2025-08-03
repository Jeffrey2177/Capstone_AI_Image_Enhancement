# Medical Image Enhancement with GAN

## Table Of Contents

- [Description](#description)
- [Setup](#setup)
- [Running the Files](#running-the-files)
- [License](#license)
- [Credits](#credits)


## Description

This project allows the enhancement of low-quality Diffusion-Weighted MRI brain scans using a Pix2Pix GAN model. It includes a Streamlit-based web app for image comparison and metric evaluation. 


## Setup

To run this project locally:

1. Open a terminal or command prompt (e.g. Terminal on macOS/Linux, Command Prompt or PowerShell on Windows)
2. Then run the following commands:

```bash

# 1. Clone the repository
git clone https://github.com/Jeffrey2177/Capstone_AI_Image_Enhancement.git

# 2. Change directory to the project folder
cd Capstone_AI_Image_Enhancement

# 3. (Optional) Create a virtual environment
python -m venv venv
source venv\Scripts\activate 

# 4. Install dependencies
pip install -r requirements.txt

```
3. Download the latest Pix2pix model (latest_net_G.pth) weight from [Google Drive](https://drive.google.com/drive/folders/1J7mjHB8N-ZNUiDiHw-8DUzIlxTjt-dff)
4. Insert the downloaded weight from Goolge Drive into ```Capstone\pytorch-CycleGAN-and-pix2pix\checkpoints\pix2pix_main ```


## How To Use

### Setting The Prerequisites
1. Open a new terminal or command prompt window
2. Change directory to the project folder
```cd Capstone_AI_Image_Enhancement```

3. Run the following Python scripts:
   
```bash
python HQ_Image_Generation.py
python LQ_Image_Generation.py
python Pix2Pix_TrainingTesting_Image_Splitter.py
python CycleGAN_TrainingTesting_Image_Splitter.py
```
### Running The app

1. In the same project directory, start the application with the following command:
``` streamlit run app.py ```
2. Alternatively visit this link [Medical Image Enhancer App](https://Jeffrey2177-Capstone.hf.space) which hosts the web application online 
 
### App Usage

In either methods of opening the application, drag and drop (or click to upload) low-quality and high-quality Diffusion-MRI brain images on the left toolbar. The application will process and display the predicted high-quality version(s) with the option to download. Metric results will also be shown for comparison purposes.


## License 

This project is designed to be open source and available under the MIT License


## Credits

This project builds upon the foundational work of [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), an open-source PyTorch implementation of CycleGAN and Pix2Pix by Jun-Yan Zhu and collaborators.  
Their repository provided the base architecture and training framework for the GAN model used in this project.

Additional contributions and customisations were made for:
- Diffusion-Weighted MRI data handling
- Synthetic degradation and enhancement workflows
- Web-based visualization and evaluation
