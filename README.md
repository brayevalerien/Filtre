# Filtre
An image dataset filtering script, made for automatic deduplication, face detection and image quality assertion.

Currently, Filtre performs the following (in order, some can be disables, see [usage section](#usage)):
1. Remove all images that are low resolution.
2. (OPTIONAL) filter images that contain an unwanted number of faces (see `--min_faces` and `--max_faces` arguments).
3. If multiple images are too similar (according to CLIP embeddings dot product), only keep the one with the highest pixel count.
4. Images are sorted by aesthetic score, meaning that the first images in the resulting directory will be the most aesthetic.

## Installation
Assuming you have [Git](https://git-scm.com/), [Python](https://www.python.org/) and [Anaconda](https://www.anaconda.com/download) installed, follow these steps to install Filtre. Note that Filtre won't work with Python 3.13 due to PyTorch.
1. Clone this repository: `git clone https://github.com/brayevalerien/Filtre` and then move into it by running `cd Filtre`.
2. (OPTIONAL BUT RECOMMANDED) create and activate a Conda environment by running `conda create -n filtre python=3.12 -y` and then `conda activate filtre`.
3. Install a torch version that matches your system by following the installation instruction on [PyTorch get started page](https://pytorch.org/get-started/locally/). Avoid CPU version otherwise similarity computation will take forever.
4. Install the other dependencies: `pip install -r requirements.txt`.

If you were able to go through these steps without any error, Filtre should be ready to be used.

## Usage
In any case, run `python filtre.py --help` to show the usage string:
```bash
$ python filtre.py --help
usage: Filtre [-h] --path PATH [--min_faces MIN_FACES] [--max_faces MAX_FACES] [--keep_similar KEEP_SIMILAR] [--ignore_aesthetic IGNORE_AESTHETIC]

An image dataset filtering script

options:
  -h, --help            show this help message and exit
  --path PATH           Path to the directory containing the images
  --min_faces MIN_FACES
                        Minimum number of faces that have to be in the images. This makes it possible to remove images without any faces by setting to 1.
  --max_faces MAX_FACES
                        Maximum number of faces that have to be in the images. This is useful to remove group photos for instance. Set to None to keep the number of faces unbounded.
  --keep_similar KEEP_SIMILAR
                        If enabled, the similarity filter (that removes images that look the same) is disabled.
  --ignore_aesthetic IGNORE_AESTHETIC
                        If enabled, the images won't be sorted by decreasing aesthetic score. This skips the aesthetic score prediction, which greatly reduces runtime, so use that if you don't need aesthetic storting.
```

Example usage:
```bash
python filtre.py --path ./dataset/ --min_faces 1 --max_faces 5 --keep_similar True
```
Running this command will filter the images in the `./dataset/` directory, removing all images that have between 1 and 5 faces in them (inclusive), and keeping similar images (skipping similarity filter). Images will be sorted by their aesthetic score, but none will be removed based on this score.

> [!WARNING]
> If the similarity filter is enabled, processing time grows in $\mathcal{O}(n^2)$ where $n$ is the number of images.
> For reference, it can take about 15 minutes for a 500 images dataset on a high end GPU.