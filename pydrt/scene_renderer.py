import drt
import torch
import time

print_timing = True
def serialize_scene(scene):
    """
        Given a Pydrt scene, convert it to a linear list of argument,
        so that we can use it in PyTorch.
    """
    cam = scene.camera
    num_shapes = len(scene.shapes)
    num_bsdfs = len(scene.bsdfs)
    num_lights = len(scene.area_lights)
    num_medium = len(scene.mediums)
    num_phases = len(scene.phases)
    for light_id, light in enumerate(scene.area_lights):
        scene.shapes[light.shape_id].light_id = light_id
    args = []
    args.append(num_shapes)
    args.append(num_bsdfs)
    args.append(num_lights)
    args.append(num_medium)
    args.append(num_phases)

    args.append(cam.cam_to_world)
    args.append(cam.cam_to_ndc)
    args.append(cam.clip_near)
    args.append(cam.resolution)
    args.append(cam.med_id)
    args.append(None)
    args.append(cam.crop_rect)

    for shape in scene.shapes:
        args.append(shape.vertices)
        args.append(shape.indices)
        args.append(shape.uvs)
        args.append(shape.normals)
        args.append(shape.bsdf_id)
        args.append(shape.light_id)
        args.append(shape.med_ext_id)
        args.append(shape.med_int_id)
        args.append(None)
    for bsdf in scene.bsdfs:
        args.append(bsdf.type)
        if bsdf.type == 'diffuse':
            args.append(bsdf.diffuse_reflectance)
            args.append(None)
        elif bsdf.type == 'null':
            pass
        elif bsdf.type == 'phong':
            args.append(bsdf.diffuse_reflectance)
            args.append(bsdf.specular_reflectance)
            args.append(bsdf.exponent)
            args.append(None)
            args.append(None)
            args.append(None)
        elif bsdf.type == 'roughdielectric':
            args.append(bsdf.alpha)
            args.append(bsdf.intIOR)
            args.append(bsdf.extIOR)
            args.append(None)   
            args.append(None)
        else:
            raise

    for light in scene.area_lights:
        args.append(light.type)
        if light.type == 'area_light':
            args.append(light.shape_id)
            args.append(light.intensity)
            args.append(light.two_sided)
        elif light.type == 'area_lightEx':
            args.append(light.shape_id)
            args.append(light.intensity)
            args.append(light.kappa)
            args.append(None)
            args.append(None)

    for medium in scene.mediums:
        args.append(medium.type)
        if medium.type == 'homogeneous':
            args.append(medium.sigma_t)
            args.append(medium.albedo)
            args.append(medium.phase_id)
            args.append(None)
            args.append(None)
        elif medium.type == 'heterogeneous':
            args.append(medium.fn_density)
            args.append(medium.albedo)
            args.append(medium.to_world)
            args.append(medium.scalar)
            args.append(medium.phase_id)
            args.append(None)
            args.append(None)
            args.append(None)
            args.append(None)
    for phase in scene.phases:
        args.append(phase.type)
        if phase.type == 'isotropic':
            pass
        elif phase.type == 'hg':
            args.append(phase.g)
            args.append(None)
        else:
            raise
    args.append(None)
    return args

def render_scene(integrator, options, *args):
    num_shapes = args[0]
    num_bsdfs  = args[1]
    num_lights = args[2]
    num_medium = args[3]
    num_phases = args[4]

    cam_to_world = args[5]
    assert(cam_to_world.is_contiguous())
    cam_to_ndc   = args[6]
    clip_near    = args[7]
    resolution   = args[8]
    cam_med_id   = args[9]
    cam_vel      = args[10]
    rect         = args[11]
    if cam_vel is not None:
        assert(cam_vel.is_contiguous())

    camera = drt.Camera(resolution[0], resolution[1],
                            drt.float_ptr(cam_to_world.data_ptr()),
                            drt.float_ptr(cam_to_ndc.data_ptr()),
                            clip_near, cam_med_id, 
                            drt.float_ptr(cam_vel.data_ptr()) if cam_vel is not None else 0)
    camera.set_rect(rect[0], rect[1], rect[2], rect[3])
    
    current_index = 12
    shapes = []
    for i in range(num_shapes):
        vertices    = args[current_index]
        indices     = args[current_index + 1]
        uvs         = args[current_index + 2]
        normals     = args[current_index + 3]
        bsdf_id     = args[current_index + 4]
        light_id    = args[current_index + 5]
        med_ext_id  = args[current_index + 6]
        med_int_id  = args[current_index + 7]
        shape_vel   = args[current_index + 8]
        assert(vertices.is_contiguous())
        assert(indices.is_contiguous())
        if uvs is not None:
            assert(uvs.is_contiguous())
        if normals is not None:
            assert(normals.is_contiguous())
        if shape_vel is not None:
            assert( shape_vel.is_contiguous() )
        shapes.append(drt.Shape(\
            drt.float_ptr(vertices.data_ptr()),
            drt.int_ptr(indices.data_ptr()),
            drt.float_ptr(uvs.data_ptr() if uvs is not None else 0),
            drt.float_ptr(normals.data_ptr() if normals is not None else 0),
            int(vertices.shape[0]),
            int(indices.shape[0]),
            light_id, bsdf_id, med_int_id, med_ext_id, 
            drt.float_ptr(shape_vel.data_ptr() if shape_vel is not None else 0)))
        current_index += 9
    bsdfs = []   
    for i in range(num_bsdfs):
        if args[current_index] == 'null':
            bsdfs.append(drt.BSDF_null())
            current_index += 1
        elif args[current_index] == 'diffuse':
            diffuse_reflectance = args[current_index + 1]
            vec_reflectance     = drt.Spectrum3f(diffuse_reflectance[0], diffuse_reflectance[1], diffuse_reflectance[2])
            default             = args[current_index + 2] is None
            if default:
                bsdfs.append(drt.BSDF_diffuse(vec_reflectance))
            else:
                d_reflectance = args[current_index + 2]
                assert(d_reflectance.is_contiguous())
                bsdfs.append(drt.BSDF_diffuse(vec_reflectance, drt.float_ptr(d_reflectance.data_ptr())))
            current_index += 3
        elif args[current_index] == 'phong':
            diffuse_reflectance  = args[current_index + 1]
            specular_reflectance = args[current_index + 2]
            exponent = args[current_index + 3]
            vec_kd  = drt.Spectrum3f(diffuse_reflectance[0], diffuse_reflectance[1], diffuse_reflectance[2])
            vec_ks  = drt.Spectrum3f(specular_reflectance[0], specular_reflectance[1], specular_reflectance[2])
            default = args[current_index + 4] is None
            if default:
                bsdfs.append(drt.BSDF_Phong(vec_kd, vec_ks, exponent))
            else:
                d_diffuse = args[current_index + 4]
                assert( d_diffuse.is_contiguous() )
                d_specular = args[current_index + 5]
                assert( (d_specular is not None) and d_specular.is_contiguous() )                
                d_exponent = args[current_index + 6]
                assert( (d_exponent is not None) and d_exponent.is_contiguous() )
                bsdfs.append(drt.BSDF_Phong(vec_kd, vec_ks, exponent, 
                                                drt.float_ptr(d_diffuse.data_ptr()),
                                                drt.float_ptr(d_specular.data_ptr()), 
                                                drt.float_ptr(d_exponent.data_ptr())))
            current_index += 7
        elif args[current_index] == 'roughdielectric':
            alpha   = args[current_index + 1]
            intIOR  = args[current_index + 2]
            extIOR  = args[current_index + 3]
            default = args[current_index + 4] is None
            if default:
                bsdfs.append(drt.BSDF_roughdielectric(alpha, intIOR, extIOR))
            else:
                d_alpha = args[current_index + 4]
                assert( d_alpha.is_contiguous() )
                d_eta = args[current_index + 5]
                assert( (d_eta is not None) and d_eta.is_contiguous())
                bsdfs.append(drt.BSDF_roughdielectric(alpha, intIOR, extIOR, drt.float_ptr(d_alpha.data_ptr()), drt.float_ptr(d_eta.data_ptr())))
            current_index += 6
        else:
            raise
    area_lights = []
    for i in range(num_lights):
        if args[current_index] == 'area_light':
            shape_id    = args[current_index + 1]
            intensity   = args[current_index + 2]
            two_sided   = args[current_index + 3]
            area_lights.append(drt.AreaLight(shape_id, drt.float_ptr(intensity.data_ptr()), two_sided))
            current_index += 4
        else:
            shape_id    = args[current_index + 1]
            intensity   = drt.Spectrum3f(args[current_index + 2][0], args[current_index + 2][1], args[current_index + 2][2])
            kappa       = args[current_index + 3]
            default     = args[current_index + 4] is None
            if default:
                area_lights.append(drt.AreaLightEx(shape_id, intensity, kappa))
            else:
                d_intensity = args[current_index + 4]
                assert( d_intensity.is_contiguous() )
                d_kappa = args[current_index + 5]
                assert( (d_kappa is not None) and d_kappa.is_contiguous() )
                area_lights.append(drt.AreaLightEx(shape_id, intensity, kappa, drt.float_ptr(d_intensity.data_ptr()), drt.float_ptr(d_kappa.data_ptr())))
            current_index += 6
    
    mediums = []
    for i in range(num_medium):
        if args[current_index] == 'homogeneous':
            sigma_t     = args[current_index + 1]
            albedo      = args[current_index + 2]
            vec_albedo  = drt.Spectrum3f(albedo[0], albedo[1], albedo[2])
            phase_id    = args[current_index + 3]
            default     = args[current_index + 4] is None
            if default:
                mediums.append(drt.Homogeneous(sigma_t, vec_albedo, phase_id))
            else:
                d_sigmaT = args[current_index + 4]
                assert( d_sigmaT.is_contiguous() )
                d_albedo = args[current_index + 5]
                assert( (d_albedo is not None) and d_albedo.is_contiguous() )
                mediums.append(drt.Homogeneous(sigma_t, vec_albedo, phase_id, drt.float_ptr(d_sigmaT.data_ptr()), 
                                                                                  drt.float_ptr(d_albedo.data_ptr())))
            current_index += 6
        elif args[current_index] == 'heterogeneous':
            fn_density  = args[current_index + 1]
            albedo      = args[current_index + 2]
            vol_albedo  = isinstance(albedo, str)
            to_world    = args[current_index + 3]
            scalar      = args[current_index + 4]
            phase_id    = args[current_index + 5]
            default     = args[current_index + 6] is None
            if default:
                if vol_albedo:
                    mediums.append(drt.Heterogeneous(fn_density, drt.float_ptr(to_world.data_ptr()), float(scalar), albedo, phase_id))
                else:
                    vec_albedo  = drt.Spectrum3f(albedo[0], albedo[1], albedo[2])
                    mediums.append(drt.Heterogeneous(fn_density, drt.float_ptr(to_world.data_ptr()), float(scalar), vec_albedo, phase_id))
            else:
                d_albedo        = args[current_index + 6]
                assert( d_albedo.is_contiguous() )
                d_scalar        = args[current_index + 7]
                assert( (d_scalar is not None) and d_scalar.is_contiguous() )
                vec_translate   = args[current_index + 8]
                assert( (vec_translate is not None) and vec_translate.is_contiguous() )
                vec_rotate      = args[current_index + 9]
                assert( (vec_rotate is not None) and vec_rotate.is_contiguous() )
                if vol_albedo:
                    mediums.append(drt.Heterogeneous(fn_density, drt.float_ptr(to_world.data_ptr()), float(scalar), albedo, phase_id,
                                                     drt.float_ptr(vec_translate.data_ptr()),
                                                     drt.float_ptr(vec_rotate.data_ptr()),
                                                     drt.float_ptr(d_scalar.data_ptr())))                    
                else:
                    vec_albedo  = drt.Spectrum3f(albedo[0], albedo[1], albedo[2])
                    mediums.append(drt.Heterogeneous(fn_density, drt.float_ptr(to_world.data_ptr()), float(scalar), vec_albedo, phase_id,
                                                     drt.float_ptr(vec_translate.data_ptr()),
                                                     drt.float_ptr(vec_rotate.data_ptr()),
                                                     drt.float_ptr(d_scalar.data_ptr()),
                                                     drt.float_ptr(d_albedo.data_ptr())))
            current_index += 10
    phases = []
    for i in range(num_phases):
        if args[current_index] == 'isotropic':
            phases.append(drt.Isotropic())
            current_index += 1
        elif args[current_index] == 'hg':
            g = args[current_index + 1]
            default  = args[current_index + 2] is None
            if default:
                phases.append(drt.HG(g));
            else:
                d_g  = args[current_index + 2]
                assert( d_g.is_contiguous() )
                phases.append(drt.HG(g, drt.float_ptr(d_g.data_ptr())));
            current_index += 3
        else:
            raise

    scene = drt.Scene(camera, shapes, bsdfs, area_lights, phases, mediums)
    if args[current_index] is not None:
        print(args[current_index])
        scene.initEdges(drt.float_ptr(args[current_index].data_ptr()))

    if rect[2] == -1 or rect[3] == -1:
        rendered_image = torch.zeros(drt.nder + 1, resolution[1], resolution[0], 3)
    else:
        rendered_image = torch.zeros(drt.nder + 1, rect[3], rect[2], 3)
    start = time.time()
    integrator.render(scene, options, drt.float_ptr(rendered_image.data_ptr()))
    time_elapsed = time.time() - start
    if print_timing:
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Total rendering time: {:0>2}h {:0>2}m {:0>2.2f}s".format(int(hours),int(minutes),seconds))
    return rendered_image
