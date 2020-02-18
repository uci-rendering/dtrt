import vredner
from pyvredner import SceneManager, SceneTransform
import pyvredner
import torch
import numpy as np

rnd_seed = 0
spp = 1024
sppe = 32768
sppse = 10
quite = False
max_bounces = 4
crop_rect = [ 0, 0, -1, -1 ]

scene, integrator, crop_rect = pyvredner.load_mitsuba('./cbox_side.xml')
options = vredner.RenderOptions(rnd_seed, 
                                spp, 
                                max_bounces,
                                crop_rect[0], 
                                crop_rect[1],
                                crop_rect[2],
                                crop_rect[3],
                                sppe,
                                sppse,
                                quite)
params = np.loadtxt('cvg_results/param.txt', delimiter=',')

args_init = pyvredner.serialize_scene(scene)
var_list = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, requires_grad=True);

T1 = SceneTransform("SHAPE_TRANSLATE", torch.tensor([100, 0, 0], dtype=torch.float), 3)
T2 = SceneTransform("SHAPE_TRANSLATE", torch.tensor([0, 100, 0], dtype=torch.float), 3)
T3 = SceneTransform("SHAPE_TRANSLATE", torch.tensor([0, 0, 100], dtype=torch.float), 3)
transform_list = []
transform_list.append([T1])
transform_list.append([T2])
transform_list.append([T3])
scene_manager = SceneManager(args_init, transform_list)
for i in range(np.size(params, 0)):
    var_list = torch.from_numpy(params[i, 1:4]).float()
    scene_manager.set_arguments(var_list)
    img = pyvredner.render_scene(integrator, options, *(scene_manager.args))
    img = img[0, :, :, :]
    pyvredner.imwrite(img, 'cvg_results/frames_side/iter_%d.png' % i)

scene_manager.set_arguments(torch.tensor([0.7, -0.7, -1.0]))
img = pyvredner.render_scene(integrator, options, *(scene_manager.args))
pyvredner.imwrite(img[0, :, :, :], 'cvg_results/target_side.exr')