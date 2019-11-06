import pydrt
import torch
import numpy as np
from drt import nder
import math
import scipy
import scipy.ndimage 



def downsample(input):
	if input.size(0) % 2 == 1:
		input = torch.cat((input, torch.unsqueeze(input[-1,:], 0)), dim=0)
	if input.size(1) % 2 == 1:
		input = torch.cat((input, torch.unsqueeze(input[:,-1], 1)), dim=1)
	return (input[0::2, 0::2, :] + input[1::2, 0::2, :] + input[0::2, 1::2, :] + input[1::2, 1::2, :]) * 0.25


	

class ADLossFunc(torch.autograd.Function):

	@staticmethod
	def forward(ctx, scene_manager, integrator, options, input, out_of_range = torch.tensor([0]*nder, dtype=torch.float),
																penalty_scale = torch.tensor([1]*nder, dtype=torch.float),
																pyramid_level = 1, 
																pyramid_scale = 4.0,
																index_iter = -1,
																clamping = 0):
		img = pydrt.render_scene(integrator, options, *(scene_manager.args))
		if index_iter > -1:
			torch.save(img, 'pt_iter%d.pt'%index_iter)
		ret = img[0, :, :, :]
		ctx.save_for_backward(img[1:, :, :,:], 
							  torch.tensor([pyramid_level], dtype=torch.int),
							  torch.tensor([pyramid_scale], dtype=torch.float),
							  out_of_range,
							  penalty_scale,
							  torch.tensor([clamping], dtype=torch.int))
		return ret

	@staticmethod
	def backward(ctx, grad_input):
		ret_list = [None, None, None]
		derivs, lvl, pyramid_scale, out_of_range, penalty_scale, clamping = ctx.saved_tensors
		lvl = int(min( math.log(derivs.size(1), 2)+1, math.log(derivs.size(2),2)+1, lvl))
		ret = torch.tensor([0]*nder, dtype=torch.float)
		grad_curr = []
		for ider in range(nder):
			grad_curr.append(derivs[ider, :, :, :])
		for i in range(lvl):
			for ider in range(nder):
				if abs(out_of_range[ider].item()) > 1e-4:
					print("param #%d is out of range..." % ider)
					ret[ider] = out_of_range[ider] * penalty_scale[ider]
				else:
					if clamping.data[0] == 0:
						ret[ider] += pow(pyramid_scale[0], i) * (grad_curr[ider] * grad_input).sum()
					else:
						clamped = grad_input.clone()
						clamped[grad_input > 2.0]  = 0.0
						clamped[grad_input < -2.0] = 0.0
						ret[ider] += pow(pyramid_scale[0], i) * (grad_curr[ider] * clamped).sum()
				if i < lvl - 1:
					grad_curr[ider] = downsample(grad_curr[ider])
			if i < lvl - 1:
				grad_input = downsample(grad_input)
		ret_list.append(ret)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		return tuple(ret_list)