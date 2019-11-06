#pragma once
#ifndef FRAME_H__
#define FRAME_H__

#include <cmath>
#include "utils.h"

struct Frame {
    Vector s, t, n;

    /// Default constructor -- performs no initialization!
    inline Frame() { }

    /// Construct a frame from the given orthonormal vectors
    inline Frame(const Vector &x, const Vector &y, const Vector &z)
     : s(x), t(y), n(z) {
    }

    /// Copy constructor
    inline Frame(const Frame &frame) : s(frame.s), t(frame.t), n(frame.n) {}

    /// Construct a new coordinate frame from a single vector
    inline Frame(const Vector &n) : n(n) {
        coordinateSystem(n, s, t);
    }
    /// Convert from world coordinates to local coordinates
    inline Vector toLocal(const Vector &v) const {
        return Vector(v.dot(s),v.dot(t),v.dot(n)
        );
    }

    /// Convert from local coordinates to world coordinates
    inline Vector toWorld(const Vector &v) const {
        return s * v.x() + t * v.y() + n * v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared cosine of the angle between the normal and v */
    inline static Float cosTheta2(const Vector &v) {
        return v.z() * v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the cosine of the angle between the normal and v */
    inline static Float cosTheta(const Vector &v) {
        return v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared sine of the angle between the normal and v */
    inline static Float sinTheta2(const Vector &v) {
        return 1.0f - v.z() * v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the sine of the angle between the normal and v */
    inline static Float sinTheta(const Vector &v) {
        Float temp = sinTheta2(v);
        return temp < Epsilon ? 0.0f : std::sqrt(temp);
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the tangent of the angle between the normal and v */
    inline static Float tanTheta(const Vector &v) {
        Float temp = 1.0f - v.z() * v.z();
        return temp < Epsilon ? 0.0f : std::sqrt(temp)/v.z();
    }

    /// Return a string representation of this frame
    inline std::string toString() const {
        std::ostringstream oss;
        oss << "Frame[" << std::endl
            << "  s = " << "(" << s.x() << "," << s.y() << ","<< s.z() << ")\n"
            << "  t = " << "(" << t.x() << "," << t.y() << ","<< t.z() << ")\n"
            << "  n = " << "(" << n.x() << "," << n.y() << ","<< n.z() << ")\n"
            << "]" << std::endl;
        return oss.str();
    }
};

#endif