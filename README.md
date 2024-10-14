# Latent Point Collapse Induces an Information Bottleneck in Deep Neural Network Classifiers

### Abstract
The information-bottleneck principle suggests that the foundation of learning lies in the ability to create compact representations. In machine learning, this goal can be formulated as a Lagrangian optimization problem, where the mutual information between the input and latent representations must be minimized without compromising the correctness of the model's predictions.
Unfortunately, mutual information is difficult to compute in deterministic deep neural network classifiers, which greatly limits the application of this approach to challenging scenarios. In this paper, we tackle this problem from a different perspective that does not involve direct computation of the mutual information. We develop a method that induces the collapse of latent representations belonging to the same class into a single point. Such a point collapse yields a significant decrease in the entropy associated with the latent distribution, thereby creating an information bottleneck. Our method is straightforward to implement, and we demonstrate that it enhances the robustness, generalizability, and reliability of the network.

### Dependencies 
Code was tested on Python 3.9, PyTorch 2.4.1, NumPy 1.26, scikit-learn 1.5.2, and SciPy 1.13.1.

### Install
```
pip install .
```

### Reproduce results 
Results can be reproduced on a GPU cluster using a slurm script that we provide in `scripts/generate_submit_slurm_jobs`. The script is assumed to be run in a conda environment named _lpb_ib_ where the dependencies indicated above and the package defined in this repository is installed. We utilized NVIDIA A100 GPUs, as indicated in the _gres_ argument in the script. Trainings can be run with the following commands:

```
python scripts/generate_submit_slurm_jobs.py --hours 10 --n-gpus 1 --config configs/svhn.yml --id-name svhn --dataset-dir datasets --results-dir results --output-dir jobs_outputs
python scripts/generate_submit_slurm_jobs.py --hours 10 --n-gpus 1 --config configs/cifar10.yml --id-name cifar10 --dataset-dir datasets --results-dir results --output-dir jobs_outputs
python scripts/generate_submit_slurm_jobs.py --hours 10 --n-gpus 1 --config configs/cifar100.yml --id-name cifar100 --dataset-dir datasets --results-dir results --output-dir jobs_outputs

```

where the `svhn.yml`, `cifar10.yml` and `cifar100.yml` files contain all training hyperparameters, the `datasets` directory is created to store the loaded datasets, the `results` directory is created to store all training results, and the `jobs_output` directory is created to store all slurms jobs outputs. 
For each of the 5 experiments, a number of training results are produced with different learning rates, and the best results in each experiment can be picked with `scripts/find_best_results.py`:

```
python scripts/find_best_results.py --results-dir results/svhn --output-dir results/svhn
python scripts/find_best_results.py --results-dir results/cifar10 --output-dir results/cifar10
python scripts/find_best_results.py --results-dir results/cifar100 --output-dir results/cifar100
```

Results can finally be visualized using the Jupyter notebook `notebooks/plots.ipynb`.


