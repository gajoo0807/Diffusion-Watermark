# Watermark Diffusion Implementation in PyTorch

## What Does it Do
This code implements a watermark diffusion technique for generating watermarked images for image classification models using Stable Diffusion.

## Setup

To use this code, follow the steps below:

1. Download the repository and create the environment:
    ```bash
    git clone https://github.com/gajoo0807/Diffusion-Watermark.git
    cd Diffusion-Watermark
    conda env create -f environment.yml
    ```

2. Generate Noise Sample & Model Distribution:
    ```bash
    python -m fingerprint.generate_dist
    ```

3. Train the Diffusion Model:
    ```bash
    python -m tools.train_ddpm_cond
    ```

4. Generate specific images using the watermark model's Out Distribution:
    ```bash
    python -m tools.sample_ddpm_cond --model_name {your model name} --sample {sample index you want to inference}
    ```
    For example, to generate images with model model_2 and sample index 4:
    ```bash
    python -m tools.sample_ddpm_cond --model_name model_2 --sample 4
    ```

