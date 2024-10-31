# FedNH

This repo provides an implementation of `FedNH` proposed in [Tackling Data Heterogeneity in Federated Learning with Class Prototypes](https://arxiv.org/abs/2212.02758), which is accepted by AAAI2023. In companion, we also provide our implementation of benchmark algorithms.

## Prepare Dataset

Please create a folder `data` under the root directory.

```
mkdir ~/data
```

- **Cifar10, Cifar100**: No extra steps are required.

- **TinyImageNet**:
  - Download the dataset:
    ```
    cd ~/data && wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    ```
  - Unzip the file:
    ```
    unzip tiny-imagenet-200.zip
    ```

## Environment Setup

To set up the environment, you can create a Conda environment using the provided `environment.yml` file. Follow these steps:

1. **Install Anaconda or Miniconda**: If you haven't already, install Anaconda or Miniconda from [the official site](https://www.anaconda.com/products/distribution).

2. **Create the environment**: Navigate to the directory containing the `environment.yml` file and run the following command:

   ```
   conda env create -f environment.yml
   ```

3. **Activate the environment**:

   ```
   conda activate your_environment_name
   ```

   Replace `your_environment_name` with the name specified in the `environment.yml` file.

### CUDA Configuration

Ensure that the CUDA version matches your GPU's configuration. You can check your GPU's specifications using the following command:

```
nvidia-smi
```

- If you see `CUDA Version: 0.X`, it may indicate that the CUDA toolkit is not correctly installed or configured. Make sure to install the compatible version based on your GPU. You can find the appropriate version on the [NVIDIA CUDA Toolkit page](https://developer.nvidia.com/cuda-downloads).

### Recommended Hardware Configuration

For optimal performance, it is recommended to use a machine with the following specifications:

- **GPU**: NVIDIA GPU with CUDA support (preferably at least a GTX 1060 or equivalent).
- **RAM**: 16GB or more.
- **Processor**: Multi-core CPU (Intel i5/i7 or AMD Ryzen).
- **Storage**: SSD for faster data access.

## Run scripts

We prepared a Python file `/experiments/gen_script.py` to generate bash commands to run experiments.

To reproduce the results for Cifar10/Cifar100, just set the variable `purpose` to `Cifar` in the `gen_script.py` file. Similarly, set `purpose` to `TinyImageNet` to run experiments for TinyImageNet.

`gen_script.py` will create a set of bash files named as `[method]_dir.sh`. Then use, for example, `bash FedAvg.sh` to run experiments.

We include a set of bash files to run experiments on `Cifar` in this submission.

### Tips for Enhanced Experiment Tracking

To utilize WandB for tracking your experiments, follow these steps:

1. **Activate WandB**: Ensure you set the WandB tracking flag to `True` in your script. This will enable logging for your runs.

2. **Link Your WandB Account**:

   - In your terminal, run the following command to log in to your WandB account:
     ```
     wandb login
     ```
   - Follow the prompts to enter your API key, which you can find on your WandB account settings page.

3. **Monitor Your Experiments**: Once WandB is set up, you can monitor your experiments through the WandB dashboard. This will provide insights into metrics, visualizations, and comparisons across runs.

### Running Scripts in the Background

If you want to run experiments without keeping the terminal open, you can use the `nohup` command in bash. This allows your script to continue running even after you log out of the terminal. For example:

```
nohup bash FedAvg.sh > FedAvg_output.txt 2>&1 &
```

- This command runs `FedAvg.sh` in the background and directs both standard output and error messages to `FedAvg_output.txt`.
- The `&` at the end of the command allows the script to run in the background, freeing up your terminal for other tasks.

## Organization of the Code

The core code can be found at `src/flbase/`. Our framework builds upon three abstract classes: `server`, `clients`, and `model`. Their concrete implementations can be found in the `models` directory and the `strategies` directory, respectively.

- **`src/flbase/models`**: We implemented or borrowed the implementation of (1) Convolution Neural Network and (2) ResNet18.
- **`src/flbase/strategies`**: We implement `CReFF`, `Ditto`, `FedAvg`, `FedBABU`, `FedNH`, `FedPer`, `FedProto`, `FedRep`, `FedROD`. Each file provides the concrete implementation of the corresponding `server` class and `client` class.

Helper functions, for example, generating non-iid data partition, can be found in `src/utils.py`.

## Credits

The code base is developed with extensive references to the following GitHub repos. Some code snippets are directly taken from the original implementation.

1. FedBABU: [https://github.com/jhoon-oh/FedBABU](https://github.com/jhoon-oh/FedBABU)
2. CReFF: [https://github.com/shangxinyi/CReFF-FL](https://github.com/shangxinyi/CReFF-FL)
3. FedROD: [https://openreview.net/revisions?id=I1hQbx10Kxn](https://openreview.net/revisions?id=I1hQbx10Kxn)
4. Personalized Federated Learning Platform: [https://github.com/TsingZ0/PFL-Non-IID](https://github.com/TsingZ0/PFL-Non-IID)
5. FedProxL: [https://github.com/litian96/FedProx](https://github.com/litian96/FedProx)
6. NIID-Bench: [https://github.com/Xtra-Computing/NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench)
7. FedProto: [https://github.com/yuetan031/fedproto](https://github.com/yuetan031/fedproto)
