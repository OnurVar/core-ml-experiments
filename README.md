# core-ml-experiments

Experiments on CoreML

# Prerequisites

- Anaconda or Miniconda installed
- Python 3.11

# Installation

```bash
conda create -n core-ml python=3.11
conda activate core-ml
conda install tensorflow=2.10 numpy
pip install coremltools
```

# Usage

```bash
python run.py --unit ALL # Default
python run.py --unit CPU_ONLY
python run.py --unit CPU_AND_GPU
python run.py --unit CPU_AND_NE
```

# Examples

To run the experiments on CPU only:

```bash
python run.py --unit CPU_ONLY
```

To run the experiments on both CPU and GPU:

```bash
python run.py --unit CPU_AND_GPU
```

To run the experiments on CPU and Neural Engine:

```bash
python run.py --unit CPU_AND_NE
```

# Troubleshooting

If you encounter issues during installation or running the experiments, try the following steps:

1. Ensure that you have the correct versions of Python and TensorFlow installed.
2. Verify that your environment is activated using `conda activate core-ml`.
3. Check for any error messages and consult the documentation for the respective tools (TensorFlow, CoreMLTools).
4. If the issue persists, consider creating a new environment and reinstalling the dependencies.
