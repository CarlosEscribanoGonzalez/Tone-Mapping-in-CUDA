## Overview
GPU implementation of the histogram-based stage of HDR tone mapping, written in CUDA. Given the _log-luminance_ of an image, it computes the cumulative distribution function (CDF) used to remap luminance values.

## Features

* Min and max luminance found with parallel reductions
* Luminance histogram computed on the GPU
* CDF obtained with a parallel exclusive scan

## Technologies

* C++
* CUDA

<p align="center">
  <img width="917" height="552" alt="memorial_raw_large" src="https://github.com/user-attachments/assets/295fc0ec-b7d0-4437-b549-96428130425b" />
</p>

## Disclaimer
Due to the small scope of this project and the tedious setup it requires (CUDA, OpenCV, NVIDIA GPU), I am only publishing the source code I developed myself. The remaining base code is not mine and is not included, so this repository is not compilable on its own.
