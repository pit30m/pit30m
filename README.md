# Pit30M Development Kit

## Overview
This is the Python software development kit for the Pit30M benchmark for large-scale global localization.

The dataset is not live yet as of November 5th 2022, but we are hard at work getting it ready for release. The devkit software
is in a very early stage and does not yet have true data to read.

In the meantime, consider checking out [the original paper](https://arxiv.org/abs/2012.12437). If you would like to, you
could also follow some of the authors' social media, e.g., [Andrei's](https://twitter.com/andreib) in order to be among
the first to hear of any updates!


## Getting Started

PyPI package coming soon (`pip install pit30m`).

To preview some example very early stage data, check out the [tutorial notebook](examples/tutorial_00_introduction.ipynb).
[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/pit30m/pit30m/blob/main/examples/tutorial_00_introduction.ipynb)

More tutorials coming soon.

### Torch Data Loading

We provide some early stage log-based torch dataloaders. Visual-localization-specific ones are coming soon. To see an
example on how to use one of these dataloaders, have a look at `demo_dataloader` in `torch/dataset.py`.

An example command:

```
python -m pit30m.torch.dataset --root-uri s3://pit30m/ --logs 00682fa6-2183-4a0d-dcfe-bc38c448090f,021286dc-5fe5-445f-e5fa-f875f2eb3c57,1c915eda-c18a-46d5-e1ec-e4f624605ff0 --num-workers 16 --batch-size 64
```

## Features

 * Framework-agnostic multiprocessing-safe log reader objects
 * PyTorch dataloaders

### In-progress
 * Very efficient native S3 support through [AWS-authored PyTorch-optimized S3 DataPipes](https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/).
 * Support for non-S3 data sources, for users who wish to copy the dataset, or parts of it, to their own storage.
 * Tons of examples and tutorials. See `examples/` for more information.


## Development

Package development, testing, and releasing is performed with `poetry`. If you just want to use the `pit30m` package, you don't need to care about this section; just have a look at "Getting Started" above!

 1. [Install poetry](https://python-poetry.org/docs/)
 2. Setup/update your `dev` virtual environments with `poetry install --with=dev` in the project root
    - If you encounter strange keyring/credential errors, you may need `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`
 3. Develop away
    - run commands like `poetry run python -m pit30m.cli`
 4. Test with `poetry run pytest`


Note that in the pre-release time, `torch` will be a "dev" dependency, since it's necessary for all tests to pass.


## Citation

```bibtex
@inproceedings{martinez2020pit30m,
  title={Pit30m: A benchmark for global localization in the age of self-driving cars},
  author={Martinez, Julieta and Doubov, Sasha and Fan, Jack and B{\^a}rsan, Ioan Andrei and Wang, Shenlong and M{\'a}ttyus, Gell{\'e}rt and Urtasun, Raquel},
  booktitle={{IROS}},
  pages={4477--4484},
  year={2020},
  organization={IEEE}
}
```