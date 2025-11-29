# Bimodal 1D Gaussian SDE Experiments

This repository contains code for experiments related to stochastic differential equations (SDEs) and MMD-based penalty methods, as presented in "A kernel-based method for Schrödinger bridges" (https://arxiv.org/abs/2310.14522).

## Project Structure

```
.
├── bimodal_1d_gauss.py        # Core implementation of SDE and MMD penalty
├── run_experiment.py          # Script to run training experiments
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── results/                   # (Optional) Directory for saving experiment results
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/username/bimodal-1d-gauss-sde.git
cd bimodal-1d-gauss-sde
pip install -r requirements.txt
```

*Replace `username` with your GitHub username.*

## Usage

Run the experiment with default settings:
```bash
python run_experiment.py
```

You can also customize the number of epochs, learning rate, or batch size:
```bash
python run_experiment.py --num_epochs 500 --lr 5e-4 --batch_size 1024
```

After training, the script will generate plots in the `results/` folder.

## Requirements

The required Python packages are listed in `requirements.txt`. Key dependencies include:
- PyTorch
- torchsde
- numpy
- matplotlib
- tqdm

To install all dependencies:
```bash
pip install -r requirements.txt
```

## Results

Sample histograms of generated data are saved in the `results/` directory after training.  
You can modify the plotting function in `bimodal_1d_gauss.py` for custom visualizations.

## Citation

If you use this code in your research, please cite our paper:

```
@article{nakano2023kernel,
  title={A kernel-based method for Schr{\"o}dinger bridges},
  author={Yumiharu Nakano},
  journal={arXiv:2310.14522[math.OC]},
  year={2023}
}
```

## License

This project is licensed under the MIT License.
