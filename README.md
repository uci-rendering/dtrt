# dtrt: Unbiased Differentiable Volumetric Renderer

![](https://shuangz.com/projects/diffrender-sa19/teaser.png)

dtrt is a physics-based differentiable renderer that can compute derivatives of rendering output with respect to arbitrary scene parameters. The renderer is capable of providing an unbiased estimation of image derivative in the existence of volumetric configuration (smoke, liquid etc.) and geometric discontinuities. The differentiable renderer can be used for solving inverse rendering problem through gradient descent optimization. For more details on the renderer and the theory behind, please refer to the paper: [A Differential Theory of Radiative Transfer](https://shuangz.com/projects/diffrender-sa19/), Cheng Zhang, Lifan Wu, Changxi Zheng, Ioannis Gkioulekas, Ravi Ramamoorthi, Shuang Zhao.

## Documentation

The renderer involves two components, C++ code `src` for computing the derivatives and python interface `pydtrt` for optimization using pytorch.


## Installation


## Dependencies

redner depends on a few libraries/systems, which are all included in the repository:
- [Python 3.6 or above](https://www.python.org)
- [pybind11](https://github.com/pybind/pybind11)
- [PyTorch 1.0 or above](https://pytorch.org)
- [OpenEXR](https://github.com/openexr/openexr)
- [Embree](https://embree.github.io)
- [OpenEXR Python](https://github.com/jamesbowman/openexrpython)
- A few other python packages: numpy, scikit-image


If you have any questions/comments/bug reports, feel free to open a github issue or e-mail to the author Cheng Zhang (zhangchengee@gmail.com)