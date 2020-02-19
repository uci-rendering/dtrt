# dtrt: An Unbiased Differentiable Volumetric Path Tracer

![](https://shuangz.com/projects/diffrender-sa19/teaser.png)

*dtrt* is a physics-based differentiable renderer computes derivative images with respect to arbitrary scene parameters (e.g., object geometries and material properties). This renderer is capable of producing unbiased derivative estimates with the presence of heterogeneous participating media. The estimated derivatives can be used for solving inverse rendering problems through gradient-based optimizations. For more details on our differential radiative transfer theory, please refer to the paper: [A Differential Theory of Radiative Transfer](https://shuangz.com/projects/diffrender-sa19/), Cheng Zhang, Lifan Wu, Changxi Zheng, Ioannis Gkioulekas, Ravi Ramamoorthi, Shuang Zhao.

## Installation
1. install all the dependencies with the provided script (You can also install them manually)
```
./install.sh
exec bash -l
```
2. install the `dtrt` & `pydtrt` library
```
mkdir build
cd build
cmake ..
sudo make install -j
```
3. Run example script to ensure the installation is successful
```
cd examples/glass_1
python3 optimize.py
```
To run any other example script, you need to change the global variable `nder` defined in file `include/config.h` so that the number of scene parameters in each example equals to `nder` value. For the number of parameters in each example scene, please refer to the [supplemental webpage](https://shuangz.com/projects/diffrender-sa19/supp_material/) of the paper

## Dependencies

redner depends on a few libraries/systems, which are all included in the repository:
- [Eigen3](http://eigen.tuxfamily.org)
- [Python 3.6 or above](https://www.python.org)
- [pybind11](https://github.com/pybind/pybind11)
- [PyTorch 1.0 or above](https://pytorch.org)
- [OpenEXR](https://github.com/openexr/openexr)
- [Embree](https://embree.github.io)
- [OpenEXR Python](https://github.com/jamesbowman/openexrpython)
- A few other python packages: numpy, scikit-image


## Documentation

The renderer involves two components, C++ code `src` for computing the derivatives and python interface `pydtrt` for optimization using pytorch.

(TBD)


If you have any questions/comments/bug reports, feel free to open a github issue or e-mail to the author Cheng Zhang (zhangchengee@gmail.com)
