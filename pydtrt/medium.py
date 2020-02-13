import pydtrt
import torch

class Homogeneous:
    def __init__(self, sigma_t, albedo, phase_id):
        self.type = 'homogeneous'
        self.sigma_t = sigma_t
        self.albedo = albedo
        self.phase_id = phase_id

class Heterogeneous:
    def __init__(self, fn_density, albedo, to_world, scalar, phase_id):
        self.type = 'heterogeneous'
        self.fn_density = fn_density
        self.albedo = albedo
        self.to_world = to_world
        self.scalar = scalar
        self.phase_id = phase_id