import torch
import xml.etree.ElementTree as etree
import numpy as np
import drt
from drt import nder, angleEps
import os
import pydrt
import pydrt.transform as transform

def parse_transform(node):
    ret = torch.eye(4)
    for child in node:
        if child.tag == 'matrix':
            value = torch.from_numpy(\
                np.reshape(\
                    np.fromstring(child.attrib['value'], dtype=np.float32, sep=' '),
                    (4, 4)))
            ret = value @ ret
        else:
            if child.tag == 'scale':
                x = 1.0; y = 1.0; z = 1.0
            else:
                x = 0.0; y = 0.0; z = 0.0
            if 'x' in child.attrib:
                x = float(child.attrib['x'])
            if 'y' in child.attrib:
                y = float(child.attrib['y'])
            if 'z' in child.attrib:
                z = float(child.attrib['z'])

            if child.tag == 'translate':
                value = transform.gen_translate_matrix(torch.tensor([x, y, z]))
                ret = value @ ret
            elif child.tag == 'scale':
                value = transform.gen_scale_matrix(torch.tensor([x, y, z]))
                ret = value @ ret
            elif child.tag == 'rotate':
                value = transform.gen_rotate_matrix([x, y, z], float(child.attrib['angle']))
                ret = value @ ret
    return ret

def parse_vector(str):
    v = np.fromstring(str, dtype=np.float32, sep=',')
    if v.shape[0] != 3:
        v = np.fromstring(str, dtype=np.float32, sep=' ')
    assert(v.ndim == 1)
    return torch.from_numpy(v)

def parse_camera(node, medium_dict):
    fov = torch.tensor([45.0])
    position = None
    look_at = None
    up = None
    clip_near = 1e-2
    resolution = [256, 256]
    crop_rect = [ 0, 0, -1, -1]
    med_id = -1
    for child in node:
        if 'name' in child.attrib:
            if child.attrib['name'] == 'fov':
                fov = torch.tensor([float(child.attrib['value'])])
            elif child.attrib['name'] == 'toWorld':
                has_lookat = False
                for grandchild in child:
                    if grandchild.tag.lower() == 'lookat':
                        has_lookat = True
                        position = parse_vector(grandchild.attrib['origin'])
                        look_at = parse_vector(grandchild.attrib['target'])
                        up = parse_vector(grandchild.attrib['up'])
                if not has_lookat:
                    print('Unsupported Mitsuba scene format: please use a look at transform')
                    assert(False)
        if child.tag == 'film':
            for grandchild in child:
                if 'name' in grandchild.attrib:
                    if grandchild.attrib['name'] == 'width':
                        resolution[0] = int(grandchild.attrib['value'])
                    elif grandchild.attrib['name'] == 'height':
                        resolution[1] = int(grandchild.attrib['value'])
                    elif grandchild.attrib['name'] == 'cropOffsetX':
                        crop_rect[0] = int(grandchild.attrib['value'])
                    elif grandchild.attrib['name'] == 'cropOffsetY':
                        crop_rect[1] = int(grandchild.attrib['value'])
                    elif grandchild.attrib['name'] == 'cropWidth':
                        crop_rect[2] = int(grandchild.attrib['value'])
                    elif grandchild.attrib['name'] == 'cropHeight':
                        crop_rect[3] = int(grandchild.attrib['value'])
        if child.tag == 'ref':
            med_id = medium_dict[child.attrib['id']]


    return pydrt.Camera(position     = position,
                            look_at      = look_at,
                            up           = up,
                            fov          = fov,
                            clip_near    = clip_near,
                            resolution   = resolution,
                            med_id       = med_id,
                            crop_rect    = crop_rect)

def parse_bsdf(node, two_sided = False):
    node_id = None
    if 'id' in node.attrib:
        node_id = node.attrib['id']
    if node.attrib['type'] == 'diffuse':
        diffuse_reflectance = torch.tensor([0.5, 0.5, 0.5])
        for child in node:
            if child.attrib['name'] == 'reflectance':
                diffuse_reflectance = parse_vector(child.attrib['value'])
        return (node_id, pydrt.BSDF_diffuse(diffuse_reflectance = diffuse_reflectance))
    elif node.attrib['type'] == 'null':
        return (node_id, pydrt.BSDF_null())
    elif node.attrib['type'] == 'phong':
        diffuse_reflectance = torch.tensor([0.5, 0.5, 0.5])
        specular_reflectance = torch.tensor([0.2, 0.2, 0.2])
        exponent = 30.0
        for child in node:
            if child.attrib['name'] == 'diffuseReflectance':
                diffuse_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'specularReflectance':
                specular_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'exponent':
                exponent = parse_vector(child.attrib['value'])
        return (node_id, pydrt.BSDF_Phong(diffuse_reflectance, specular_reflectance, exponent))
    elif node.attrib['type'] == 'roughdielectric':
        for child in node:
            if child.attrib['name'] == 'alpha':
                alpha = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'intIOR':
                intIOR = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'extIOR':
                extIOR = parse_vector(child.attrib['value'])
        return (node_id, pydrt.BSDF_roughdielectric(alpha, intIOR, extIOR))
    else:
        print('Unsupported bsdf type:', node.attrib['type'])
        assert(False)

def parse_phase(node):
    if node.attrib['type'] == 'isotropic':
        return pydrt.Isotropic()
    elif node.attrib['type'] == 'hg':
        for child in node:
            if child.attrib['name'] == 'g':
                g = parse_vector(child.attrib['value'])
        return pydrt.HG(g)

def parse_medium(node):
    node_id = None
    phase = None
    to_world = torch.eye(4)
    to_world0 = torch.eye(4)
    if 'id' in node.attrib:
        node_id = node.attrib['id']
    if node.attrib['type'] == 'homogeneous':
        for child in node:
            if child.tag == 'phase':
                phase = parse_phase(child)
            elif child.attrib['name'] == 'sigmaT':
                sigma_t = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'albedo':
                albedo = parse_vector(child.attrib['value'])
        if phase == None:
            phase = pydrt.Isotropic();
        return (node_id, pydrt.Homogeneous(sigma_t = sigma_t, albedo = albedo, phase_id = -1), phase)
    elif node.attrib['type'] == 'heterogeneous':
        scalar = 1.0
        for child in node:
            if child.tag == 'volume':
                if child.attrib['name'] == 'density':
                    for grandchild in child:
                        if 'name' in grandchild.attrib:
                            if grandchild.attrib['name'] == 'filename':
                                fn_density = grandchild.attrib['value']
                            elif grandchild.attrib['name'] == 'toWorld':
                                to_world = parse_transform(grandchild)
                elif child.attrib['name'] == 'albedo':
                    for grandchild in child:
                        if 'name' in grandchild.attrib:
                            if grandchild.attrib['name'] == 'value':
                                assert(child.attrib['type'] == 'constvolume')
                                albedo = parse_vector(grandchild.attrib['value'])
                            elif grandchild.attrib['name'] == "filename":
                                assert(child.attrib['type'] == 'gridvolume')
                                albedo = grandchild.attrib['value']
                            elif grandchild.attrib['name'] == 'toWorld':
                                to_world0 = parse_transform(grandchild)
            elif child.tag == 'phase':
                phase = parse_phase(child)
            elif 'name' in child.attrib:
                if child.attrib['name'] == 'scale':
                    scalar = parse_vector(child.attrib['value'])
        if phase == None:
            phase = pydrt.Isotropic();
        if isinstance(albedo, str):
            assert (torch.all(torch.lt(torch.abs(torch.add(to_world, -to_world0)), 1e-6)))
        return (node_id, pydrt.Heterogeneous(fn_density = fn_density, albedo = albedo, scalar = scalar, to_world = to_world.contiguous(), phase_id = -1), phase)
    else:
        print('Unsupported bsdf type:', node.attrib['type'])
        assert(False)

def parse_shape(node, bsdf_dict, med_dict, shape_id):
    if node.attrib['type'] != 'obj' and node.attrib['type'] != 'rectangle':
        print("Unsupported shape type:", node.attrib['type'])
        assert(False)
    to_world = torch.eye(4)
    bsdf_id = -1
    med_ext_id = med_int_id = -1
    light_intensity = None
    light_kappa = None

    filename = ''
    for child in node:
        if 'name' in child.attrib:
            if child.attrib['name'] == 'filename':
                assert(node.attrib['type'] == 'obj')
                filename = child.attrib['value']
            elif child.attrib['name'] == 'toWorld':
                to_world = parse_transform(child)
        if child.tag == 'ref':
            if 'name' in child.attrib:
                if child.attrib['name'] == 'interior':
                    med_int_id = med_dict[child.attrib['id']]
                elif child.attrib['name'] == 'exterior':
                    med_ext_id = med_dict[child.attrib['id']]
            else:
                bsdf_id = bsdf_dict[child.attrib['id']]
        elif child.tag == 'emitter':
            if child.attrib['type'] == 'area':
                for grandchild in child:
                    if grandchild.attrib['name'] == 'radiance':
                        light_intensity = parse_vector(grandchild.attrib['value'])
                        if light_intensity.shape[0] == 1:
                            light_intensity = torch.tensor(\
                                         [light_intensity[0],
                                          light_intensity[0],
                                          light_intensity[0]])
            elif child.attrib['type'] == 'areaEx':
                for grandchild in child:
                    if grandchild.attrib['name'] == 'radiance':
                        light_intensity = parse_vector(grandchild.attrib['value'])
                        if light_intensity.shape[0] == 1:
                            light_intensity = torch.tensor(\
                                         [light_intensity[0],
                                          light_intensity[0],
                                          light_intensity[0]])
                    elif grandchild.attrib['name'] == 'kappa':
                        light_kappa = float(grandchild.attrib['value'])

    if filename == '':
        assert(node.attrib['type'] == 'rectangle')
        indices = torch.tensor([[0, 2, 1], [1, 2, 3]], dtype = torch.int32)
        vertices = torch.tensor([[-1.0, -1.0, 0.0],
                                 [-1.0,  1.0, 0.0],
                                 [ 1.0, -1.0, 0.0],
                                 [ 1.0,  1.0, 0.0]])
        uvs = None
        normals = None
    else:
        mesh_list = pydrt.load_obj(filename)
        vertices = mesh_list[0].vertices.cpu()
        indices = mesh_list[0].indices.cpu()
        uvs = mesh_list[0].uvs
        normals = mesh_list[0].normals
        if uvs is not None:
            uvs = uvs.cpu()
        if normals is not None:
            normals = normals.cpu()

    # Transform the vertices and normals
    vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1)), dim = 1)
    vertices = vertices @ torch.transpose(to_world, 0, 1)
    vertices = vertices / vertices[:, 3:4]
    vertices = vertices[:, 0:3].contiguous()
    if normals is not None:
        normals = normals @ (torch.inverse(torch.transpose(to_world, 0, 1))[:3, :3])
        normals = normals.contiguous()
    assert(vertices is not None)
    assert(indices is not None)
    lgt = None
    if light_intensity is not None:
        if light_kappa is None:
            lgt = pydrt.AreaLight(shape_id, light_intensity)
        else:
            lgt = pydrt.AreaLightEx(shape_id, light_intensity, light_kappa)
    return pydrt.Shape(vertices, indices, uvs, normals, bsdf_id, med_ext_id, med_int_id), lgt

def parse_scene(node):
    cam = None
    resolution = None
    bsdfs = []
    bsdf_dict = {}
    shapes = []
    lights = []
    medium_dict = {}
    mediums = []
    phases = []
    for child in node:
        if child.tag == 'sensor':
            cam = parse_camera(child, medium_dict)
        elif child.tag == 'bsdf':
            node_id, bsdf = parse_bsdf(child)
            if node_id is not None:
                bsdf_dict[node_id] = len(bsdfs)
                bsdfs.append(bsdf)
        elif child.tag == 'medium':
            node_id, medium, phase = parse_medium(child)
            if node_id is not None:
                medium_dict[node_id] = len(mediums)
                mediums.append(medium)
                medium.phase_id = len(phases)
                phases.append(phase)
        elif child.tag == 'shape':
            shape, light = parse_shape(child, bsdf_dict, medium_dict, len(shapes))
            shapes.append(shape)
            if light is not None:
                lights.append(light)
        elif child.tag == 'integrator':
            if child.attrib['type'] == 'direct':
                integrator = drt.DirectIntegrator();
            elif child.attrib['type'] == 'path':
                integrator = drt.PathTracer();
            elif child.attrib['type'] == 'volpath_simple':
                integrator = drt.VolPathTracerSimple();
            elif child.attrib['type'] == 'volpath':
                integrator = drt.VolPathTracer();
            elif child.attrib['type'] == 'directAD':
                integrator = drt.DirectAD();
            elif child.attrib['type'] == 'pathAD':
                integrator = drt.PathTracerAD();
            elif child.attrib['type'] == 'volpathAD':
                integrator = drt.VolPathTracerAD();
            elif child.attrib['type'] == 'ptracer':
                integrator = drt.ParticleTracer();
            else:
                raise Exception("Integrator type [ %s ] not supported!" % child.attrib['type'])
            # print("Rendering using [ %s ] ..." % child.attrib['type'])
    return pydrt.Scene(cam, shapes, bsdfs, mediums, phases, lights), integrator

def load_mitsuba(filename):
    """
        Load from a Mitsuba scene file as PyTorch tensors.
    """
    print('# derivatives = %d, angle Eps = %.1e' % (nder, angleEps))
    tree = etree.parse(filename)
    root = tree.getroot()
    cwd = os.getcwd()
    os.chdir(os.path.dirname(filename))
    ret = parse_scene(root)
    os.chdir(cwd)
    return ret
