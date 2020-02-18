import dtrt
from pydtrt import SceneManager, SceneTransform
import pydtrt
import torch

rnd_seed = 0
spp = 512
sppe = 32768
sppse = 50
quite = False
max_bounces = 10

scene, integrator = pydtrt.load_mitsuba('./shadow.xml')
options = dtrt.RenderOptions(rnd_seed, spp, max_bounces, sppe, sppse, quite)
output_image = True
output_loss = True
output_param = True

args_init = pydtrt.serialize_scene(scene)
var_list = torch.tensor([0.0]*dtrt.nder, dtype=torch.float, requires_grad=True);

direc = 'results/'
T1     = SceneTransform("SHAPE_ROTATE", torch.tensor([0.0, 1.0, 0.0], dtype=torch.float), 2)
T1_vol = SceneTransform("MEDIUM_VARY", "heterogeneous", 0, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float),
                                                           torch.tensor([0.0, 1.0, 0.0], dtype=torch.float),
                                                           torch.tensor([0.0, 0.0, 0.0], dtype=torch.float),
                                                           0.0)

T2 = SceneTransform("MEDIUM_VARY", "heterogeneous", 0, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float),
                                                           torch.tensor([0.0, 0.0, 0.0], dtype=torch.float),
                                                           torch.tensor([0.0, 0.0, 0.0], dtype=torch.float),
                                                           5.0)

transform_list = []
transform_list.append([T1, T1_vol])
transform_list.append([T2])
scene_manager = SceneManager(args_init, transform_list)


integrator = dtrt.VolPathTracer()
init = pydtrt.render_scene(integrator, options, *(scene_manager.args))
pydtrt.imwrite(init[0, :, :, :], direc+'init.exr')
target_var = torch.tensor([1.6, 1.4])
print("Target variable values: ", target_var)
scene_manager.set_arguments(target_var);
target = pydtrt.render_scene(integrator, options, *(scene_manager.args))
target = target[0, :, :, :]
pydtrt.imwrite(target, direc+'target.exr')

integrator = dtrt.VolPathTracerAD()
scene_manager.reset()
lossFunc = pydtrt.ADLossFunc.apply
optimizer = torch.optim.Adam([var_list], lr=5e-2)
grad_out_range = torch.tensor([0]*dtrt.nder, dtype=torch.float)
if output_loss:
    file1 = open(direc+'loss.txt', 'w')
if output_param:
    file2 = open(direc+'param.txt', 'w')
for t in range(100):
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
        file2.write("%d" % t)
        for i in range(dtrt.nder):
          file2.write(", %.5f" % var_list[i])
        file2.write("\n")
        file2.flush()
    grad_out_range = scene_manager.set_arguments(var_list)
file1.close()
file2.close()