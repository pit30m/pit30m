# Pit30M Development Kit

## Overview
This is the Python software development kit for the Pit30M benchmark for large-scale global localization.

The dataset is not live yet as of 2022-10-23 but we are hard at work getting it ready for release. The devkit software
is in a very early stage and doesn't yet have true data to read.

In the meantime, consider checking out [the original paper](https://arxiv.org/pdf/2012.12437.pdf). If you'd like to, you
could also follow some of the authors' social media, e.g., [Andrei's](https://twitter.com/andreib) in order to be among
the first to hear of any updates!


## Getting Started

To preview some example very early stage data, check out the [tutorial notebook](pit30m/examples/tutorial_00_introduction.ipynb).
[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/pit30m/pit30m/blob/main/pit30m/examples/tutorial_00_introduction.ipynb)

More tutorials coming soon.

### Torch Data Loading

We provide some early stage log-based torch dataloaders. Visual-localization-specific ones are coming soon. To see an
example on how to use one of these dataloaders, have a look at `demo_dataloader` in `torch/dataset.py`.


## Features

 * Framework-agnostic multiprocessing-safe log reader objects
 * PyTorch dataloaders

### In-progress
 * Very efficient native S3 support through [AWS-authored PyTorch-optimized S3 DataPipes](https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/).
 * Support for non-S3 data sources, for users who wish to copy the dataset, or parts of it, to their own storage.
 * Tons of examples and tutorials. See `examples/` for more information.


## Citation

```bibtex
@inproceedings{martinez2020pit30m,
  title={Pit30m: A benchmark for global localization in the age of self-driving cars},
  author={Martinez, Julieta and Doubov, Sasha and Fan, Jack and Wang, Shenlong and M{\'a}ttyus, Gell{\'e}rt and Urtasun, Raquel and others},
  booktitle={{IROS}},
  pages={4477--4484},
  year={2020},
  organization={IEEE}
}
```