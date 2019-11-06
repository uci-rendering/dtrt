#pragma once
#ifndef CAMERA_H__
#define CAMERA_H__

#include "ptr.h"
#include "fwd.h"
#include "frameAD.h"
#include "pmf.h"
#include "utils.h"

struct Ray;
struct Shape;
struct Scene;

struct CropRectangle {
    int offset_x, offset_y;
    int crop_width, crop_height;
    CropRectangle(): offset_x(0), offset_y(0), crop_width(-1), crop_height(-1) {}
    CropRectangle(int ox, int oy, int width, int height): 
                 offset_x(ox), offset_y(oy), crop_width(width), crop_height(height) {}
    bool isValid() const { return crop_width > 0 && crop_height > 0;}
};

struct Camera {
	Camera() {}
    Camera(int width, int height, ptr<float> cam_to_world, ptr<float> cam_to_ndc, float clip_near, int med_id, ptr<float> velocities = ptr<float>(nullptr));
    inline void setCropRect(int offset_x, int offset_y, int crop_width, int crop_height) { rect = CropRectangle(offset_x, offset_y, crop_width, crop_height); }
    inline int getNumPixels() const { return width*height; }
    inline int getMedID() const { return med_id; }
    
    void zeroVelocities();
    void initVelocities(const Eigen::Matrix<Float, 3, -1> &dx);
    void initVelocities(const Eigen::Matrix<Float, 3, 1> &dx, int der_index);
    void initVelocities(const Eigen::Matrix<Float, 3, -1> &dx, const Eigen::Matrix<Float, 3, -1> &dw);
    void initVelocities(const Eigen::Matrix<Float, 3, 1> &dx, const Eigen::Matrix<Float, 3, 1> &dw, int der_index);
    void advance(Float stepSize, int derId);

    Ray samplePrimaryRay(int pixel_x, int pixel_y, const Vector2 &rnd2) const;
    RayAD samplePrimaryRayAD(int pixel_x, int pixel_y, const Vector2 &rnd2) const;
    Ray samplePrimaryRay(Float x, Float y) const;
    RayAD samplePrimaryRayAD(Float x, Float y) const;
    Float sampleDirect(const Vector& p, Vector2& pixel_uv, Vector& dir) const;

	int width, height;
    CropRectangle rect;
    Matrix4x4 cam_to_world;
    Matrix4x4 world_to_cam;
    Matrix3x3 ndc_to_cam;
    Matrix3x3 cam_to_ndc;
    VectorAD cpos;
    FrameAD cframe;

    float clip_near;
    int med_id;
};

#endif