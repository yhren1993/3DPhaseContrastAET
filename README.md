# 3D Phase Contrast Atomic Electron Tomography
This repository contains the code for algorithm introduced in the following paper

[A Practical Reconstruction Method for Three-Dimensional Phase Contrast Atomic Electron Tomography](https://arxiv.org/abs/1807.03886)

[David Ren](http://scholar.google.com/citations?user=zTsT-cAAAAAJ&hl=en)\*, [Colin Ophus](https://foundry.lbl.gov/people/colin_ophus.html)\*, Michael Chen and [Laura Waller](https://www.laurawaller.com) (\* Authors contributed equally).


## Contents
1. [Usage](#usage)
2. [Data](#data)
3. [Updates](#updates)
4. [FAQ](#FAQ)

## Usage 
0. Install required dependencies [Arrayfire](https://github.com/arrayfire/arrayfire-python) and [contexttimer](https://pypi.org/project/contexttimer/).
1. Clone this repo: ```git clone https://github.com/yhren1993/3DPhaseContrastAET.git```
2. Copy measurement data into ```data/measurement/```

## Data
All simulated measurements, reconstructed volumes, and tracing results can be found in README under ```data/measurement/```,  ```data/reconstruction/```, and  ```data/atomtrace/```, respectively.

## Updates
10/22/2018:
1. Added first version of code to repo. Same as those used in the paper.

10/29/2018:
1. Added links to data location

01/25/2019:
1. Fixed an issue for using TV on CPU

## FAQ
#### What is the GPU configuration recommended?
In order to fit all of the computation on GPU, GPU memory of 11 GB (GTX 1080 Ti for example) is needed. Note that current Arrayfire does not support CUDA 10, which is required for a RTX GPU. 

