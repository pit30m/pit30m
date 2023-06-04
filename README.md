# Pit30M Development Kit

[![Python CI Status](https://github.com/pit30m/pit30m/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/pit30m/pit30m/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pit30m)](https://pypi.org/project/pit30m/)
[![Public on the AWS Open Data Registry](https://shields.io/badge/Open%20Data%20Registry-public-green?logo=amazonaws&style=flat)](#)

## Overview
This is the Python software development kit for the Pit30M benchmark for large-scale global localization. The devkit is currently in a pre-release state and many features are coming soon!

Consider checking out [the original paper](https://arxiv.org/abs/2012.12437). If you would like to, you could also follow some of the authors' social media, e.g., [Andrei's](https://twitter.com/andreib) in order to be among
the first to hear of any updates!

:warning: As of v0.0.1 the ground truth is not yet available as we are wrapping up the final steps for the dataset splits (train/test/val + query/database). The poses will be added once the next version of the devkit is available. Please see [the issue tracking v0.0.2](https://github.com/pit30m/pit30m/issues/20) for more information on this.

## Getting Started

The recommended way to interact with the dataset is with the `pip` package, which you can install with:

`pip install pit30m`

The devkit lets you efficiently access data on the fly. Here is a "hello world" command which renders a demo video from a random log segment. Note that it assumes `ffmpeg` is installed:

`python -m pit30m.cli multicam_demo --out-dir .`

To preview data more interactively, check out the [tutorial notebook](examples/tutorial_00_introduction.ipynb).
[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/pit30m/pit30m/blob/main/examples/tutorial_00_introduction.ipynb)

More tutorials coming soon.

### Torch Data Loading

We provide basic log-based PyTorch dataloaders. Visual-localization-specific ones are coming soon. To see an
example on how to use one of these dataloaders, have a look at `demo_dataloader` in `torch/dataset.py`.

An example command:

```
python -m pit30m.torch.dataset --root-uri s3://pit30m/ --logs 00682fa6-2183-4a0d-dcfe-bc38c448090f,021286dc-5fe5-445f-e5fa-f875f2eb3c57,1c915eda-c18a-46d5-e1ec-e4f624605ff0 --num-workers 16 --batch-size 64
```

## Features

 * Framework-agnostic multiprocessing-safe log reader objects
 * PyTorch dataloaders

### In-progress
 * More lightweight package with fewer dependencies.
 * Very efficient native S3 support through [AWS-authored PyTorch-optimized S3 DataPipes](https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/).
 * Support for non-S3 data sources, for users who wish to copy the dataset, or parts of it, to their own storage.
 * Tons of examples and tutorials. See `examples/` for more information.


## Development

Package development, testing, and releasing is performed with `poetry`. If you just want to use the `pit30m` package, you don't need to care about this section; just have a look at "Getting Started" above!

 1. [Install poetry](https://python-poetry.org/docs/)
 2. Setup/update your `dev` virtual environments with `poetry install --with=dev` in the project root
    - If you encounter strange keyring/credential errors, you may need `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --with=dev`
 3. Develop away
    - run commands like `poetry run python -m pit30m.cli`
 4. Test with `poetry run pytest`
    - Advanced command: `poetry run pytest -ff --new-first --quiet --color=yes --maxfail=3 -n 4`
    - This command will run tests, then wait for new changes and test them automatically. Test execution will run in parallel thanks to the `-n 4` argument.
    - The command lets you get feedback on whether your code change fixed or broke a particular test within seconds.
 5. Remember to run `poetry install` after pulling and/or updating dependencies.


Note that in the pre-release time, `torch` will be a "dev" dependency, since it's necessary for all tests to pass.

### Publishing

 1. [Configure poetry](https://www.digitalocean.com/community/tutorials/how-to-publish-python-packages-to-pypi-using-poetry-on-ubuntu-22-04) with a PyPI account which has access to edit the package. You need to make sure poetry is configured with your API key.
 2. `poetry publish --build`


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

## Additional Details

### Images

#### Compression
The images in the dataset are stored using lossy WebP compression at quality level 85. We picked this as a sweet spot between space- and network-bandwidth-saving (about 10x smaller than equivalent PNGs) and maintaining very good image quality for tasks such as SLAM, 3D reconstruction, and visual localization. The images were saved using `Pillow 9.2.0`.

The `s3://pit30m/raw` prefix contains lossless image data for a small subset of the logs present in the bucket root. This can be used as a reference by those curious in understanding which artifacts are induced by the lossy compression, and which are inherent in the raw data.

#### Known Issues

A fraction of the images in the dataset exhibit artifacts such as a strong purple tint or missing data (white images). An even smaller fraction of these purple images sometimes shows strong blocky compression artifacts. These represent a known (and, at this scale, difficult to avoid) problem; it was already present in the original raw logs from which we generated the public facing benchmark. Perfectly blank images can be detected quite reliably in a data loader or ETL script by checking whether `np.mean(img) > 250`.

On example of a log with many blank (whiteout) images is `8438b1ba-44e2-4456-f83b-207351a99865`.

### Ground Truth Sanitization

Poses belonging to test-query are not available and have been removed with zeroes / NaNs / blank submap IDs in the corresponding pose files and indexes. The long term plan is to use this held-out test query ground truth for a public leaderboard. More information will come in the second half of 2023. In the meantime, there should be a ton of data to iterate on using the publicly available train and val splits.