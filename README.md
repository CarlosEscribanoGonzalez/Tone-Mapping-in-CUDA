## Overview
GPU implementation of the histogram-based stage of HDR tone mapping, written in CUDA. Given the _log-luminance_ of an image, it computes the cumulative distribution function (CDF) used to remap luminance values.

## Features

* Min and max luminance found with parallel reductions
* Luminance histogram computed on the GPU
* CDF obtained with a parallel exclusive scan

## Authorship

I only developed the code in `funcHDR.cu`. The rest of the project (host code, image loading, reference implementation and build files) is not mine and is included only so the project can run.

## Technologies

* C++
* CUDA

## Requirements

* NVIDIA GPU with CUDA support
* CUDA Toolkit
