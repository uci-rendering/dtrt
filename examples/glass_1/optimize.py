import dtrt
from pydtrt import SceneManager, SceneTransform
import pydtrt
import torch

rnd_seed = 0
spp = 1024
sppe = 32768
sppse = 10
quite = False
max_bounces = 4

scene, integrator = pydtrt.load_mitsuba('./scene.xml')
options = dtrt.RenderOptions(rnd_seed, spp, max_bounces, sppe, sppse, quite)
output_image = True
output_loss = True
output_param = True

args_init = pydtrt.serialize_scene(scene)
var_list = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float, requires_grad=True);

direc = 'results/'
T1 = SceneTransform("SHAPE_TRANSLATE", torch.tensor([1.0, 0.0, 0.0], dtype=torch.float), 0)
T2 = SceneTransform("SHAPE_TRANSLATE", torch.tensor([0.0, 1.0, 0.0], dtype=torch.float), 0)
T3 = SceneTransform("SHAPE_TRANSLATE", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float), 0)
T4 = SceneTransform("BSDF_VARY", "roughdielectric", 0, 0.1, 0.0)

transform_list = []
transform_list.append([T1])
transform_list.append([T2])
transform_list.append([T3])
transform_list.append([T4])
scene_manager = SceneManager(args_init, transform_list)

target_var = torch.tensor([-5.0, -3.0, -4.0, -2.5])
print("Target variable values: ", target_var)
scene_manager.set_arguments(target_var);
target = pydtrt.render_scene(integrator, options, *(scene_manager.args))
target = target[0, :, :, :]
pydtrt.imwrite(target, direc+'target.exr')

scene_manager.reset()
integrator = dtrt.VolPathTracerAD()
lossFunc = pydtrt.ADLossFunc.apply
optimizer = torch.optim.Adam([var_list], lr=2e-1)
grad_out_range = torch.tensor([0]*dtrt.nder, dtype=torch.float)
if output_loss:
    file1 = open(direc+'loss.txt', 'w')
if output_loss:
    file2 = open(direc+'param.txt', 'w')
for t in range(150):
    print('iteration:', t)
    optimizer.zero_grad()
    options.seed = t + 1
    img = lossFunc(scene_manager, integrator, options, var_list, grad_out_range, torch.tensor([10000.0]*dtrt.nder, dtype=torch.float), 10)
    if output_image:
        pydtrt.imwrite(img, direc+'iter_%d.exr'%t)
    loss = (img - target).pow(2).sum()
    loss.backward()
    optimizer.step()
    print("values: ", var_list)
    if output_loss:
        file1.write("%d, %.5f, %.5f\n" % (t, loss, (var_list-target_var).pow(2).sum()))
        file1.flush()
    if output_param:
        file2.write("%d, %.5f, %.5f, %.5f\n" % (t, var_list[0], var_list[1], var_list[2]))
        file2.flush()
    grad_out_range = scene_manager.set_arguments(var_list)
file1.close()
file2.close()