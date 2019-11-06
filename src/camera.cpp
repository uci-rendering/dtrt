#include "camera.h"
#include "ray.h"
#include "shape.h"
#include "camera.h"
#include "scene.h"
#include <math.h>

Camera::Camera(int width, int height, ptr<float> cam_to_world, ptr<float> cam_to_ndc, float clip_near, int med_id, ptr<float> velocities) {
    this->width = width;
    this->height = height;
    this->clip_near = clip_near;
    this->med_id = med_id;
    this->cam_to_world << cam_to_world[0],  cam_to_world[1],  cam_to_world[2],  cam_to_world[3],
                          cam_to_world[4],  cam_to_world[5],  cam_to_world[6],  cam_to_world[7],
                          cam_to_world[8],  cam_to_world[9],  cam_to_world[10], cam_to_world[11],
                          cam_to_world[12], cam_to_world[13], cam_to_world[14], cam_to_world[15];
    this->world_to_cam = this->cam_to_world.inverse();
    this->cam_to_ndc << cam_to_ndc[0], cam_to_ndc[1], cam_to_ndc[2],
                        cam_to_ndc[3], cam_to_ndc[4], cam_to_ndc[5],
                        cam_to_ndc[6], cam_to_ndc[7], cam_to_ndc[8];
    assert(this->cam_to_ndc(0,0) == this->cam_to_ndc(1,1));
    this->ndc_to_cam = this->cam_to_ndc.inverse();

    cpos.val     = this->cam_to_world.block<3,1>(0,3);
    cframe.s.val = this->cam_to_world.block<3,1>(0,0);
    cframe.t.val = this->cam_to_world.block<3,1>(0,1);
    cframe.n.val = this->cam_to_world.block<3,1>(0,2);

    if (!velocities.is_null()) {
        int nrows = 3;
        int ncols = nder;
        Eigen::MatrixXf dx(nrows, ncols);
        dx = Eigen::Map<Eigen::MatrixXf>(velocities.get(), dx.rows(), dx.cols());
        Eigen::MatrixXf dw(nrows, ncols);
        dw = Eigen::Map<Eigen::MatrixXf>(velocities.get() + 3*nder, dw.rows(), dw.cols());
        initVelocities(dx.cast<Float>(), dw.cast<Float>());
    }
}

Ray Camera::samplePrimaryRay(int pixel_x, int pixel_y, const Vector2 &rnd2) const {
	Vector2 screen_pos = Vector2( (pixel_x + rnd2.x())/Float(width),
                                  (pixel_y + rnd2.y())/Float(height));
    auto aspect_ratio = Float(width) / Float(height);
    auto ndc = Vector3((screen_pos(0) - 0.5f) * 2.f,
                       (screen_pos(1) - 0.5f) * (-2.f) / aspect_ratio,
                        Float(1));
    Vector3 dir = ndc_to_cam * ndc;
    dir.normalize();
    return Ray{xfm_point(cam_to_world, Vector3::Zero()), xfm_vector(cam_to_world, dir)};
}

Ray Camera::samplePrimaryRay(Float x, Float y) const {
    x /= width;
    y /= height;
    auto aspect_ratio = Float(width) / Float(height);
    auto ndc = Vector3((x - 0.5f)*2.f, (y - 0.5f)*(-2.f)/aspect_ratio, 1.0f);
    Vector3 dir = ndc_to_cam * ndc;
    dir.normalize();
    return Ray{xfm_point(cam_to_world, Vector3::Zero()), xfm_vector(cam_to_world, dir)};
}

RayAD Camera::samplePrimaryRayAD(int pixel_x, int pixel_y, const Vector2 &rnd2) const {
    Vector2 screen_pos = Vector2( (pixel_x + rnd2.x())/Float(width),
                                  (pixel_y + rnd2.y())/Float(height));
    auto aspect_ratio = Float(width) / Float(height);
    auto ndc = Vector3((screen_pos(0) - 0.5f) * 2.f,
                       (screen_pos(1) - 0.5f) * (-2.f) / aspect_ratio,
                        Float(1));
    Vector3 dir = ndc_to_cam * ndc;
    dir.normalize();

    return RayAD(cpos, cframe.toWorld(VectorAD(dir)));
}

RayAD Camera::samplePrimaryRayAD(Float x, Float y) const {
    x /= width;
    y /= height;
    auto aspect_ratio = Float(width) / Float(height);
    auto ndc = Vector3((x - 0.5f)*2.f, (y - 0.5f)*(-2.f)/aspect_ratio, 1.0f);
    Vector3 dir = ndc_to_cam * ndc;
    dir.normalize();
    return RayAD(cpos, cframe.toWorld(VectorAD(dir)));
}

void Camera::zeroVelocities() {
    cframe.s.zeroGrad();
    cframe.t.zeroGrad();
    cframe.n.zeroGrad();
    cpos.zeroGrad();
}

void Camera::initVelocities(const Eigen::Matrix<Float, 3, -1> &dx) {
    assert(dx.cols() == nder);
    for (int i = 0; i < nder; i++)
        cpos.grad(i) = dx.col(i);
}

void Camera::initVelocities(const Eigen::Matrix<Float, 3, 1> &dx, int der_index) {
    assert(der_index >= 0 && der_index < nder);
    cpos.grad(der_index) = dx;
}

void Camera::initVelocities(const Eigen::Matrix<Float, 3, -1> &dx, const Eigen::Matrix<Float, 3, -1> &dw) {
    assert(dx.cols() == nder && dw.cols() == nder);
    initVelocities(dx);
    for (int i = 0; i < nder; i++) {
        cframe.s.grad(i) = dw.col(i).cross(cframe.s.val);
        cframe.t.grad(i) = dw.col(i).cross(cframe.t.val);
        cframe.n.grad(i) = dw.col(i).cross(cframe.n.val);
    }
}

void Camera::initVelocities(const Eigen::Matrix<Float, 3, 1> &dx, const Eigen::Matrix<Float, 3, 1> &dw, int der_index) {
    assert(der_index >= 0 && der_index < nder);
    initVelocities(dx, der_index);
    cframe.s.grad(der_index) = dw.cross(cframe.s.val);
    cframe.t.grad(der_index) = dw.cross(cframe.t.val);
    cframe.n.grad(der_index) = dw.cross(cframe.n.val);
}

void Camera::advance(Float stepSize, int derId) {
    assert(derId >= 0 && derId < nder);
    cpos.advance(stepSize, derId);
    cframe.s.advance(stepSize, derId);
    cframe.t.advance(stepSize, derId);
    cframe.n.advance(stepSize, derId);
}

Float Camera::sampleDirect(const Vector& p, Vector2& pixel_uv, Vector& dir) const {
    Vector refP = xfm_point(world_to_cam, p);
    if (refP.z() < clip_near)
        return 0.0;
    auto fov_factor = cam_to_ndc(0, 0);
    auto aspect_ratio = Float(width) / Float(height);
    Float inv_area = 0.25 * fov_factor*fov_factor*aspect_ratio;

    int xmin = 0, xmax = width;
    int ymin = 0, ymax = height;

    if (rect.isValid()) {
        inv_area *= (Float)width/(Float)rect.crop_width *  (Float)height/rect.crop_height;
        xmin = rect.offset_x; xmax = rect.offset_x + rect.crop_width;
        ymin = rect.offset_y; ymax = rect.offset_y + rect.crop_height;
    }
     

    Vector pos_camera = cam_to_ndc * refP;
    pos_camera.x() /= pos_camera.z();
    pos_camera.y() /= pos_camera.z();
    Vector2 screen_pos = Vector2( (pos_camera.x() * 0.5f + 0.5f) * width,
                                  (-pos_camera.y() * 0.5f * aspect_ratio + 0.5f) * height);
    if (screen_pos.x() >= xmin && screen_pos.x() <=xmax &&
        screen_pos.y() >= ymin && screen_pos.y() <=ymax)
    {
        pixel_uv.x() = screen_pos.x() - xmin;
        pixel_uv.y() = screen_pos.y() - ymin;
        Float dist = refP.norm(), inv_dist = 1.0f/dist;
        refP *= inv_dist;
        Float inv_cosTheta = 1.0f/refP.z();
        dir = (cpos.val - p) * inv_dist;
        Float inv_pixel_area = inv_area * (xmax-xmin) * (ymax-ymin);
        return inv_dist * inv_dist * inv_cosTheta * inv_cosTheta * inv_cosTheta * inv_pixel_area;
    } else {
        return 0.0;
    }
}