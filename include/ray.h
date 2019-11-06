#pragma once
#ifndef RAY_H__
#define RAY_H__


#include <limits>
#include "fwd.h"

struct Ray {
    inline Ray() {}
    inline Ray(const Vector &org, const Vector& dir) : org(org), dir(dir) {}
    inline Ray(const Ray &ray) : org(ray.org), dir(ray.dir) {}

    inline Vector operator() (Float t) const { return org + t*dir; }
    inline Ray flipped() const { return Ray(org, -dir); }

    Vector org, dir;
};

#endif
