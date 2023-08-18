# Project Name

![Project Banner or Visual](./logo.png)

A planck PR3 compressed CMB TTTEEE likelihood and emulator-acceleraterated sampler. 

## Table of Contents
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)


## Usage

* The best way to get started with this compressed likelihood is by runing the example_notebook.ipynb provided. This is a self-contained application which makes use of the emcee sampler in run_mcmc_with_sampler.py to produce compressed Planck18 CMB TTTEEE + Lensing contours.

* If you are only interested in the MOPED compression vectors these are stored under ./data/moped_compression_vecs there are one set for the CMB lensing and a separate set for the CMB primary TTTEEE

* The compression vectors are stored as 2D numpy arrays which represent the compression vectors for each of the parameters stacked together in the same order as the name of the files. 

## Support

Feel free to reach out at areeves@phys.ethz.ch if you have any questions.

## Contributing

Please feel free to contribute to this project! It would be great to add in other compressed likelihoods!

---

© 2023 Alexander Reeves

