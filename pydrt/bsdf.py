import pydrt
import torch

class BSDF_diffuse:
    def __init__(self,
                 diffuse_reflectance):
        self.diffuse_reflectance = diffuse_reflectance
        self.type = 'diffuse'

class BSDF_null:
    def __init__(self):
        self.type = 'null'

class BSDF_Phong:
    def __init__(self,
                 diffuse_reflectance, specular_reflectance, exponent):
        self.diffuse_reflectance = diffuse_reflectance
        self.specular_reflectance = specular_reflectance
        self.exponent = exponent
        self.type = 'phong'

class BSDF_roughdielectric:
    def __init__(self, alpha, intIOR, extIOR):
        self.alpha = alpha
        self.intIOR = intIOR
        self.extIOR = extIOR
        self.type = 'roughdielectric'
