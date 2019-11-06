#pragma once
#ifndef GRID_VOLUME_H__
#define GRID_VOLUME_H__

#include "fwd.h"
#include "aabb.h"
#include <Eigen/Geometry>
#include <stdio.h>
#include <string>
#include <vector>

#define GRIDVOL_ROTATE_AROUND_CENTER


struct GridVolume {
    inline GridVolume() { clear(); }

    inline GridVolume(const std::string &fname, const Matrix4x4 &volumeToWorld = Matrix4x4::Identity()) {
        init(fname, volumeToWorld);
    }

    inline GridVolume(const GridVolume &vol)
        : m_isValid(vol.m_isValid), m_res(vol.m_res), m_channels(vol.m_channels)
        , m_min(vol.m_min), m_max(vol.m_max), m_extent(vol.m_extent)
        , m_center(vol.m_center), m_aabb(vol.m_aabb)
        , m_data(vol.m_data), m_maxFloatValue(vol.m_maxFloatValue)
        , m_volumeToWorld(vol.m_volumeToWorld), m_worldToVolume(vol.m_worldToVolume)
        , m_transVelocities(vol.m_transVelocities), m_rotVelocities(vol.m_rotVelocities)
    {}


    inline void clear() {
        m_isValid = false;
        m_volumeToWorld = m_worldToVolume = Matrix4x4::Identity();
        m_transVelocities.setZero();
        m_rotVelocities.setZero();
        m_res.setZero();
        m_channels = 0;
        m_maxFloatValue = 0.0f;
        m_min.fill(std::numeric_limits<Float>::infinity());
        m_max.fill(-std::numeric_limits<Float>::infinity());
        m_extent.setZero();
        m_center.setZero();
        m_aabb.reset();
        m_data.clear();
    }


    inline void init(const std::string &fname, const Matrix4x4 &volumeToWorld = Matrix4x4::Identity()) {
        if ( !(volumeToWorld.row(3) - Eigen::Matrix<Float, 1, 4>(0.0f, 0.0f, 0.0f, 1.0f)).isZero(Epsilon) ) {
            std::cerr << "Invalid to-world transformation:\n" << volumeToWorld << std::endl;
            assert(false);
        }

        m_volumeToWorld = volumeToWorld;
        m_worldToVolume = volumeToWorld.inverse();
        m_transVelocities.setZero();
        m_rotVelocities.setZero();
        loadFromFile(fname);
        configure();
        m_isValid = true;
    }


    inline void initVelocities(const Eigen::Matrix<Float, 3, nder> &transVelocities,
                               const Eigen::Matrix<Float, 3, nder> &rotVelocities)
    {
        m_transVelocities = transVelocities;
        m_rotVelocities = rotVelocities;
    }


    inline void advance(Float delta, int derId = 0) {
#ifdef GRIDVOL_ROTATE_AROUND_CENTER
        auto t = Eigen::Translation<Float, 3>(m_center + delta*m_transVelocities.col(derId))*
                 Eigen::AngleAxis<Float>(delta, m_rotVelocities.col(derId))*
                 Eigen::Translation<Float, 3>(-m_center);
#else
        auto t = Eigen::Translation<Float, 3>(delta*m_transVelocities.col(derId))*
                 Eigen::AngleAxis<Float>(delta, m_rotVelocities.col(derId));
#endif
        m_volumeToWorld = t.matrix()*m_volumeToWorld;
        m_worldToVolume = m_volumeToWorld.inverse();
        configure();
    }


    void loadFromFile(const std::string &filename) {
        FILE *fin = fopen(filename.c_str(), "rb");
        assert(fin != nullptr);
        size_t tmp;

        char header[3];
        tmp = fread(header, sizeof(char), 3, fin); assert(tmp == 3);
        if ( strncmp(header, "VOL", 3) != 0 ) {
            std::cerr << "Encountered an invalid volume data file (incorrect header identifier)" << std::endl;
            assert(false);
        }

        uint8_t version;
        tmp = fread(&version, sizeof(uint8_t), 1, fin); assert(tmp == 1);
        if ( version != 3 ) {
            std::cerr << "Encountered an invalid volume data file (incorrect file version)" << std::endl;
            assert(false);
        }

        int type;
        tmp = fread(&type, sizeof(int), 1, fin); assert(tmp == 1);
        tmp = fread(&m_res, sizeof(int), 3, fin); assert(tmp == 3);
        tmp = fread(&m_channels, sizeof(int), 1, fin); assert(tmp == 1);

        if ( type != 1 || (m_channels != 1 && m_channels != 3) ) {
            std::cerr << "Encountered a volume data file of unknown type (type=" << type << ", channels=" << m_channels << ")!" << std::endl;
            assert(false);
        }

        Vector3f pmin, pmax;
        tmp = fread(&pmin, sizeof(float), 3, fin); assert(tmp == 3); m_min = pmin.cast<Float>();
        tmp = fread(&pmax, sizeof(float), 3, fin); assert(tmp == 3); m_max = pmax.cast<Float>();
        m_extent = m_max - m_min;

        size_t elemCount = static_cast<size_t>(m_res.prod())*m_channels;
        m_data.resize(elemCount);
        tmp = fread(&m_data[0], sizeof(float), elemCount, fin);
        assert(tmp == elemCount);
        fclose(fin);

        m_maxFloatValue = 0.0f;
        for ( size_t i = 0; i < elemCount; ++i )
            if ( (m_data[i] = std::max(m_data[i], 0.0f)) > m_maxFloatValue )
                m_maxFloatValue = m_data[i];
    }


    inline void configure() {
        m_center = volumeToWorld((m_min + m_max)*0.5f);
        AABB dataAABB(m_min, m_max);
        m_aabb.reset();
        for ( int i = 0; i < 8; ++i )
            m_aabb.expandBy(volumeToWorld(dataAABB.getCorner(i)));
    }


    inline Float lookupFloat(const Vector &_p) const {
        assert(m_channels == 1);
        const Vector p = volumeToGrid(worldToVolume(_p));

        int x1 = static_cast<int>(std::floor(p.x())),
            y1 = static_cast<int>(std::floor(p.y())),
            z1 = static_cast<int>(std::floor(p.z()));
        int x2 = x1 + 1, y2 = y1 + 1, z2 = z1 + 1;
        if ( x1 < 0 || x2 >= m_res.x() || y1 < 0 || y2 >= m_res.y() || z1 < 0 || z2 >= m_res.z() )
            return 0.0f;

        const Float fx = p.x() - x1, fy = p.y() - y1, fz = p.z() - z1;
        const Float _fx = 1.0f - fx, _fy = 1.0f - fy, _fz = 1.0f - fz;

        const Float d000 = m_data[(z1*m_res.y() + y1)*m_res.x() + x1],
                    d001 = m_data[(z1*m_res.y() + y1)*m_res.x() + x2],
                    d010 = m_data[(z1*m_res.y() + y2)*m_res.x() + x1],
                    d011 = m_data[(z1*m_res.y() + y2)*m_res.x() + x2],
                    d100 = m_data[(z2*m_res.y() + y1)*m_res.x() + x1],
                    d101 = m_data[(z2*m_res.y() + y1)*m_res.x() + x2],
                    d110 = m_data[(z2*m_res.y() + y2)*m_res.x() + x1],
                    d111 = m_data[(z2*m_res.y() + y2)*m_res.x() + x2];

        return ((d000*_fx + d001*fx)*_fy +
                (d010*_fx + d011*fx)*fy)*_fz +
               ((d100*_fx + d101*fx)*_fy +
                (d110*_fx + d111*fx)*fy)*fz;
    }


    inline FloatAD lookupFloatAD(const VectorAD &_p) const {
        assert(m_channels == 1);
        const VectorAD p = volumeToGridAD(worldToVolumeAD(_p));

        int x1 = static_cast<int>(std::floor(p.x().val)),
            y1 = static_cast<int>(std::floor(p.y().val)),
            z1 = static_cast<int>(std::floor(p.z().val));
        int x2 = x1 + 1, y2 = y1 + 1, z2 = z1 + 1;
        if ( x1 < 0 || x2 >= m_res.x() || y1 < 0 || y2 >= m_res.y() || z1 < 0 || z2 >= m_res.z() )
            return FloatAD();

        const FloatAD fx = p.x() - x1, fy = p.y() - y1, fz = p.z() - z1;
        const FloatAD _fx = 1.0f - fx, _fy = 1.0f - fy, _fz = 1.0f - fz;

        const Float d000 = m_data[(z1*m_res.y() + y1)*m_res.x() + x1],
                    d001 = m_data[(z1*m_res.y() + y1)*m_res.x() + x2],
                    d010 = m_data[(z1*m_res.y() + y2)*m_res.x() + x1],
                    d011 = m_data[(z1*m_res.y() + y2)*m_res.x() + x2],
                    d100 = m_data[(z2*m_res.y() + y1)*m_res.x() + x1],
                    d101 = m_data[(z2*m_res.y() + y1)*m_res.x() + x2],
                    d110 = m_data[(z2*m_res.y() + y2)*m_res.x() + x1],
                    d111 = m_data[(z2*m_res.y() + y2)*m_res.x() + x2];

        return ((d000*_fx + d001*fx)*_fy +
                (d010*_fx + d011*fx)*fy)*_fz +
               ((d100*_fx + d101*fx)*_fy +
                (d110*_fx + d111*fx)*fy)*fz;
    }


    inline Spectrum lookupSpectrum(const Vector &_p) const {
        assert(m_channels == 3);
        const Vector p = volumeToGrid(worldToVolume(_p));
        Eigen::Map<const Eigen::Array<float, 3, -1> > data(&m_data[0], 3, m_data.size()/3);

        int x1 = static_cast<int>(std::floor(p.x())),
            y1 = static_cast<int>(std::floor(p.y())),
            z1 = static_cast<int>(std::floor(p.z()));
        int x2 = x1 + 1, y2 = y1 + 1, z2 = z1 + 1;
        if ( x1 < 0 || x2 >= m_res.x() || y1 < 0 || y2 >= m_res.y() || z1 < 0 || z2 >= m_res.z() )
            return Spectrum::Zero();

        const Float fx = p.x() - x1, fy = p.y() - y1, fz = p.z() - z1;
        const Float _fx = 1.0f - fx, _fy = 1.0f - fy, _fz = 1.0f - fz;

        const auto d000 = data.col((z1*m_res.y() + y1)*m_res.x() + x1).cast<Float>(),
                   d001 = data.col((z1*m_res.y() + y1)*m_res.x() + x2).cast<Float>(),
                   d010 = data.col((z1*m_res.y() + y2)*m_res.x() + x1).cast<Float>(),
                   d011 = data.col((z1*m_res.y() + y2)*m_res.x() + x2).cast<Float>(),
                   d100 = data.col((z2*m_res.y() + y1)*m_res.x() + x1).cast<Float>(),
                   d101 = data.col((z2*m_res.y() + y1)*m_res.x() + x2).cast<Float>(),
                   d110 = data.col((z2*m_res.y() + y2)*m_res.x() + x1).cast<Float>(),
                   d111 = data.col((z2*m_res.y() + y2)*m_res.x() + x2).cast<Float>();

        return ((d000*_fx + d001*fx)*_fy +
                (d010*_fx + d011*fx)*fy)*_fz +
               ((d100*_fx + d101*fx)*_fy +
                (d110*_fx + d111*fx)*fy)*fz;
    }


    inline SpectrumAD lookupSpectrumAD(const VectorAD &_p) const {
        assert(m_channels == 3);
        const VectorAD p = volumeToGridAD(worldToVolumeAD(_p));
        Eigen::Map<const Eigen::Array<float, 3, -1> > data(&m_data[0], 3, m_data.size()/3);

        int x1 = static_cast<int>(std::floor(p.x().val)),
            y1 = static_cast<int>(std::floor(p.y().val)),
            z1 = static_cast<int>(std::floor(p.z().val));
        int x2 = x1 + 1, y2 = y1 + 1, z2 = z1 + 1;
        if ( x1 < 0 || x2 >= m_res.x() || y1 < 0 || y2 >= m_res.y() || z1 < 0 || z2 >= m_res.z() )
            return SpectrumAD();

        const FloatAD fx = p.x() - x1, fy = p.y() - y1, fz = p.z() - z1;
        const FloatAD _fx = 1.0f - fx, _fy = 1.0f - fy, _fz = 1.0f - fz;

        const SpectrumAD d000(data.col((z1*m_res.y() + y1)*m_res.x() + x1).cast<Float>().transpose()),
                         d001(data.col((z1*m_res.y() + y1)*m_res.x() + x2).cast<Float>().transpose()),
                         d010(data.col((z1*m_res.y() + y2)*m_res.x() + x1).cast<Float>().transpose()),
                         d011(data.col((z1*m_res.y() + y2)*m_res.x() + x2).cast<Float>().transpose()),
                         d100(data.col((z2*m_res.y() + y1)*m_res.x() + x1).cast<Float>().transpose()),
                         d101(data.col((z2*m_res.y() + y1)*m_res.x() + x2).cast<Float>().transpose()),
                         d110(data.col((z2*m_res.y() + y2)*m_res.x() + x1).cast<Float>().transpose()),
                         d111(data.col((z2*m_res.y() + y2)*m_res.x() + x2).cast<Float>().transpose());

        return ((d000*_fx + d001*fx)*_fy +
                (d010*_fx + d011*fx)*fy)*_fz +
               ((d100*_fx + d101*fx)*_fy +
                (d110*_fx + d111*fx)*fy)*fz;
    }


    inline std::string toString() const {
        std::ostringstream oss;
        oss << "GridVolume[" << std::endl
            << "  res = [" << m_res.transpose() << "]," << std::endl
            << "  channels = " << m_channels << "," << std::endl
            << "  aabb = [" << m_min.transpose() << "], [" << m_max.transpose() << "]" << std::endl
            << "]";
        return oss.str();
    }


    inline Vector volumeToWorld(const Vector &p) const {
        return m_volumeToWorld.block<3, 3>(0, 0)*p + m_volumeToWorld.block<3, 1>(0, 3);
    }


    inline VectorAD volumeToWorldAD(const VectorAD &_p) const {
        VectorAD p(volumeToWorld(_p.val));
        for ( int i = 0; i < nder; ++i ) {
#ifdef GRIDVOL_ROTATE_AROUND_CENTER
            Vector q = p.val - m_center;
#else
            Vector q = p.val;
#endif
            p.grad(i) = m_volumeToWorld.block<3, 3>(0, 0)*_p.grad(i) + m_rotVelocities.col(i).cross(q) + m_transVelocities.col(i);
        }
        return p;
    }


    inline Vector worldToVolume(const Vector &p) const {
        return m_worldToVolume.block<3, 3>(0, 0)*p + m_worldToVolume.block<3, 1>(0, 3);
    }


    inline VectorAD worldToVolumeAD(const VectorAD &_p) const {
        VectorAD p(worldToVolume(_p.val));
        for ( int i = 0; i < nder; ++i ) {
#ifdef GRIDVOL_ROTATE_AROUND_CENTER
            Vector q = _p.val - m_center;
#else
            Vector q = _p.val;
#endif
            p.grad(i) = m_worldToVolume.block<3, 3>(0, 0)*(_p.grad(i) - m_rotVelocities.col(i).cross(q) - m_transVelocities.col(i));
        }
        return p;
    }


    inline Vector volumeToGrid(const Vector &p) const {
        return (p - m_min).cwiseProduct((m_res - Vector3i::Ones()).cast<Float>()).cwiseQuotient(m_extent);
    }


    inline VectorAD volumeToGridAD(const VectorAD &p) const {
        VectorAD ret;
        ret.x() = (p.x() - m_min.x())*(static_cast<Float>(m_res.x() - 1)/m_extent.x());
        ret.y() = (p.y() - m_min.y())*(static_cast<Float>(m_res.y() - 1)/m_extent.y());
        ret.z() = (p.z() - m_min.z())*(static_cast<Float>(m_res.z() - 1)/m_extent.z());
        return ret;
    }


    inline bool isValid() const { return m_isValid; }
    inline Float getMaximumFloatValue() const { return m_maxFloatValue; }
    inline const AABB &getAABB() const { return m_aabb; }


    bool m_isValid;

    Vector3i m_res;
    int m_channels;

    // In volume space
    Vector m_min, m_max, m_extent;

    // In world space
    Vector m_center;
    AABB m_aabb;

    std::vector<float> m_data;
    Float m_maxFloatValue;

    Matrix4x4 m_volumeToWorld, m_worldToVolume;
    Eigen::Matrix<Float, 3, nder> m_transVelocities, m_rotVelocities;
};


#endif //GRID_VOLUME_H__
