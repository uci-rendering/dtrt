# dtrt: An Unbiased Differentiable Volumetric Path Tracer

![](https://shuangz.com/projects/diffrender-sa19/teaser.png)

**dtrt** is a physics-based differentiable renderer computes derivative images with respect to arbitrary scene parameters (e.g., object geometries and material properties). This renderer is capable of producing unbiased derivative estimates with the presence of heterogeneous participating media. The estimated derivatives can be used for solving inverse rendering problems through gradient-based optimizations. For more details on our differential radiative transfer theory, please refer to the paper: [A Differential Theory of Radiative Transfer](https://shuangz.com/projects/diffrender-sa19/), Cheng Zhang, Lifan Wu, Changxi Zheng, Ioannis Gkioulekas, Ravi Ramamoorthi, and Shuang Zhao.

## Compilation
**dtrt** has been tested mainly under Ubuntu 18.04 LTS and can be compiled as follows.

1. Install all the dependencies manually or using the provided script:
```
./install.sh
exec bash -l
```

2. Install the `dtrt` & `pydtrt` libraries:
```
mkdir build
cd build
cmake ..
sudo make install -j
```

3. To test if the compilation is successful, execute the example script:
```
cd examples/glass_1
python3 optimize.py
```

To run any other example script, the global variable `nder` defined in `include/config.h` needs to be changed to match the number of scene parameters with respect to which the derivatives are computed. Please refer to [this page](https://shuangz.com/projects/diffrender-sa19/supp_material/) for the number of parameters in each example scene.

## Dependencies

**dtrt** depends on a few libraries/systems that can be automatically installed using the accompanying `install.sh` script:
- [Eigen3](http://eigen.tuxfamily.org)
- [Python 3.6 or above](https://www.python.org)
- [pybind11](https://github.com/pybind/pybind11)
- [PyTorch 1.0 or above](https://pytorch.org)
- [OpenEXR](https://github.com/openexr/openexr)
- [Embree](https://embree.github.io)
- [OpenEXR Python](https://github.com/jamesbowman/openexrpython)
- A few other python packages: numpy, scikit-image


## Documentation

**dtrt** involves two major components: (i) C++ differentiable rendering code under `/src`; and (ii) python interface code under `/pydtrt` that allows the rendered results to be used by Python-based tools like PyTorch.

If you have any questions/comments/bug reports, please open a github issue or e-mail Cheng Zhang at zhangchengee@gmail.com.
