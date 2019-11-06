#pragma once
#ifndef FRAME_AD_H__
#define FRAME_AD_H__

#include <cmath>
#include "utils.h"
#include "frame.h"

struct FrameAD {
    VectorAD s, t, n;

    /// Default constructor -- performs no initialization!
    inline FrameAD() { }

    /// Construct a frame from the given orthonormal vectors
    inline FrameAD(const VectorAD &x, const VectorAD &y, const VectorAD &z) : s(x), t(y), n(z) {}

    /// Copy constructor
    inline FrameAD(const FrameAD &frame) : s(frame.s), t(frame.t), n(frame.n) {}

    /// Construct a new coordinate frame from a single vector
    inline FrameAD(const VectorAD &n) : n(n) {
        coordinateSystemAD(n, s, t);
    }

    /// Convert from world coordinates to local coordinates
    inline VectorAD toLocal(const VectorAD &v) const {
        return VectorAD(v.dot(s), v.dot(t), v.dot(n));
    }

    /// Convert from local coordinates to world coordinates
    inline VectorAD toWorld(const VectorAD &v) const {
        return s*v.x() + t*v.y() + n*v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared cosine of the angle between the normal and v */
    inline static FloatAD cosTheta2(const VectorAD &v) {
        return v.z()*v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the cosine of the angle between the normal and v */
    inline static FloatAD cosTheta(const VectorAD &v) {
        return v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared sine of the angle between the normal and v */
    inline static FloatAD sinTheta2(const VectorAD &v) {
        return 1.0f - v.z()*v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the sine of the angle between the normal and v */
    inline static FloatAD sinTheta(const VectorAD &v) {
        FloatAD temp = sinTheta2(v);
        return temp < Epsilon ? FloatAD(0.0f) : temp.sqrt();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the tangent of the angle between the normal and v */
    inline static FloatAD tanTheta(const VectorAD &v) {
        FloatAD temp = 1.0f - v.z()*v.z();
        return temp < Epsilon ? FloatAD(0.0f) : temp.sqrt()/v.z();
    }

    inline Frame toFrame() const {
        return Frame(s.val, t.val, n.val);
    }

    /// Return a string representation of this frame
    inline std::string toString() const {
        std::ostringstream oss;
        oss << "FrameAD[" << std::endl
            << "  s = " << s << "\n"
            << "  t = " << t << "\n"
            << "  n = " << n << "\n"
            << "]" << std::endl;
        return oss.str();
    }
};

#endif //FRAME_AD_H__
