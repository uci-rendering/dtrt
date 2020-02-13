import pydtrt
import torch

class Isotropic:
    def __init__(self):
    	self.type = 'isotropic'

class HG:
	def __init__(self, g):
		self.g = g
		self.type = 'hg'