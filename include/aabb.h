#pragma once
#ifndef AABB_H__
#define AABB_H__

#include "fwd.h"
#include "ray.h"
#include "utils.h"


struct AABB {
    inline AABB() { reset(); }

    inline AABB(const Vector &p) : min(p), max(p) {}

    /// Create a bounding box from two positions
    inline AABB(const Vector &min, const Vector &max) : min(min), max(max) {}

    /// Copy constructor
    inline AABB(const AABB &aabb) : min(aabb.min), max(aabb.max) {}

    /// Equality test
    inline bool operator==(const AABB &aabb) const {
        return (min - aabb.min).isZero(Epsilon) && (max - aabb.max).isZero(Epsilon);
    }

    /// Inequality test
    inline bool operator!=(const AABB &aabb) const {
        return !(min - aabb.min).isZero(Epsilon) || !(max - aabb.max).isZero(Epsilon);
    }

    /// Clip to another bounding box
    inline void clip(const AABB &aabb) {
        for (int i = 0; i < 3; ++i) {
            min[i] = std::max(min[i], aabb.min[i]);
            max[i] = std::min(max[i], aabb.max[i]);
        }
    }

    /**
     * \brief Mark the bounding box as invalid.
     *
     * This operation sets the components of the minimum
     * and maximum position to \f$\infty\f$ and \f$-\infty\f$,
     * respectively.
     */
    inline void reset() {
        min.fill( std::numeric_limits<Float>::infinity());
        max.fill(-std::numeric_limits<Float>::infinity());
    }

    inline Vector getCenter() const { return (max + min)*0.5f; }

    /// Return the position of one of the corners
    inline Vector getCorner(int index) const {
        Vector result;
        for ( int d = 0; d < 3; ++d )
            result[d] = (index & (1 << d)) ? max[d] : min[d];
        return result;
    }

    /// Check whether a point lies on or inside the bounding box
    inline bool contains(const Vector &p) const {
        for ( int i = 0; i < 3; ++i )
            if (p[i] < min[i] - Epsilon || p[i] > max[i] + Epsilon)
                return false;
        return true;
    }

    /// Expand the bounding box to contain another point
    inline void expandBy(const Vector &p) {
        for ( int i = 0; i < 3; ++i ) {
            min[i] = std::min(min[i], p[i]);
            max[i] = std::max(max[i], p[i]);
        }
    }

    /// Expand the bounding box to contain another bounding box
    inline void expandBy(const AABB &aabb) {
        for ( int i = 0; i < 3; ++i) {
            min[i] = std::min(min[i], aabb.min[i]);
            max[i] = std::max(max[i], aabb.max[i]);
        }
    }

    /// Calculate the squared point-AABB distance
    inline Float squaredDistanceTo(const Vector &p) const {
        Float result = 0;
        for ( int i = 0; i < 3; ++i ) {
            Float value = 0;
            if (p[i] < min[i])
                value = min[i] - p[i];
            else if (p[i] > max[i])
                value = p[i] - max[i];
            result += value*value;
        }
        return result;
    }

    /// Calculate the point-AABB distance
    inline Float distanceTo(const Vector &p) const {
        return std::sqrt(squaredDistanceTo(p));
    }

    /// Calculate the minimum squared AABB-AABB distance
    inline Float squaredDistanceTo(const AABB &aabb) const {
        Float result = 0.0f;

        for ( int i = 0; i < 3; ++i ) {
            Float value = 0.0f;
            if (aabb.max[i] < min[i])
                value = min[i] - aabb.max[i];
            else if (aabb.min[i] > max[i])
                value = aabb.min[i] - max[i];
            result += value*value;
        }
        return result;
    }

    /// Calculate the minimum AABB-AABB distance
    inline Float distanceTo(const AABB &aabb) const {
        return std::sqrt(squaredDistanceTo(aabb));
    }

    /// Return whether this bounding box is valid
    inline bool isValid() const {
        for (int i = 0; i < 3; ++i)
            if (max[i] < min[i]) return false;
        return true;
    }

    /**
     * \brief Return whether or not this bounding box
     * covers anything at all.
     *
     * A bounding box which only covers a single point
     * is considered nonempty.
     */
    inline bool isEmpty() const {
        for (int i = 0; i < 3; ++i) {
            if (max[i] > min[i]) return false;
        }
        return true;
    }

    /// Return the axis index with the largest associated side length
    inline int getLargestAxis() const {
        Vector d = max - min;
        int largest = 0;

        for (int i = 1; i < 3; ++i )
            if (d[i] > d[largest]) largest = i;
        return largest;
    }

    /// Return the axis index with the shortest associated side length
    inline int getShortestAxis() const {
        Vector d = max - min;
        int shortest = 0;

        for ( int i = 1; i < 3; ++i )
            if (d[i] < d[shortest]) shortest = i;
        return shortest;
    }

    /**
     * \brief Calculate the bounding box extents
     * \return max-min
     */
    inline Vector getExtents() const {
        return max - min;
    }

    /** \brief Calculate the near and far ray-AABB intersection
     * points (if they exist).
     *
     * The parameters \c nearT and \c farT are used to return the
     * ray distances to the intersections (including negative distances).
     * Any previously contained value is overwritten, even if there was
     * no intersection.
     *
     * \remark In the Python bindings, this function returns the
     * \c nearT and \c farT values as a tuple (or \c None, when no
     * intersection was found)
     */
    inline bool rayIntersect(const Ray &ray, Float &nearT, Float &farT) const {
        nearT = -std::numeric_limits<Float>::infinity();
        farT  = std::numeric_limits<Float>::infinity();

        /* For each pair of AABB planes */
        for ( int i = 0; i < 3; i++ ) {
            const Float origin = ray.org[i];
            const Float minVal = min[i], maxVal = max[i];

            if (ray.dir[i] == 0) {
                /* The ray is parallel to the planes */
                if (origin < minVal || origin > maxVal)
                    return false;
            } else {
                /* Calculate intersection distances */
                Float t1 = (minVal - origin)/ray.dir[i];
                Float t2 = (maxVal - origin)/ray.dir[i];

                if (t1 > t2)
                    std::swap(t1, t2);

                nearT = std::max(t1, nearT);
                farT = std::min(t2, farT);

                if (!(nearT <= farT))
                    return false;
            }
        }

        return true;
    }

    inline Float surfaceArea() const {
        Vector d = max - min;
        return 2 * (d.x() * d.y() + d.x() * d.z() + d.y() * d.z());
    }

    bool sphereIntersect(const Vector& center, Float radius) {
        Float d_min = 0.0;
        Float r2 = radius * radius;
        for(int i = 0; i < 3; i++) {
            if (center[i] < min[i]) {
                d_min += square(center[i] - min[i]);
            } else if (center[i] > max[i]) {
                d_min += square(center[i] - max[i]);
            }
            if (d_min <= r2) {
                return true;
            }
        }
        return false;
    }
    
    /// Return a string representation of the bounding box
    std::string toString() const {
        std::ostringstream oss;
        oss << "AABB[";
        if (!isValid()) {
            oss << "invalid";
        } else {
            oss << "min=" << min << ", max=" << max;
        }
        oss << "]";
        return oss.str();
    }

    Vector min, max;
};


#endif //AABB_H__
