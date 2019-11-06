#include "utils.h"
#include "math_func.h"
#include <iostream>
#include <Eigen/Geometry>


Vector3 xfm_point(const Matrix4x4 &T, const Vector3& pos) {
    Vector4 pos_homo(pos.x(), pos.y(), pos.z(), 1.0f);
    pos_homo = T * pos_homo;
    return pos_homo.head(3) / pos_homo.w();
}

Vector3 xfm_vector(const Matrix4x4 &T, const Vector3& vec) {
    return T.block(0,0,3,3) * vec;
}


Vector2 squareToUniformDiskConcentric(const Vector2 &sample) {
    Float r1 = 2.0f*sample.x() - 1.0f;
    Float r2 = 2.0f*sample.y() - 1.0f;
    Float phi, r;
    if (r1 == 0 && r2 == 0) {
        r = phi = 0;
    } else if (r1*r1 > r2*r2) {
        r = r1;
        phi = (M_PI/4.0f) * (r2/r1);
    } else {
        r = r2;
        phi = (M_PI/2.0f) - (r1/r2) * (M_PI/4.0f);
    }
    Float cosPhi = sin(phi);
    Float sinPhi = cos(phi);
    return Vector2(r * cosPhi, r * sinPhi);
}

Vector squareToCosineHemisphere(const Vector2 &sample) {
    Vector2 p = squareToUniformDiskConcentric(sample);
    Float z = std::sqrt(std::max((Float)0.0, 1.0f - p.x()*p.x() - p.y()*p.y()));
    return Vector(p.x(), p.y(), z);
}

Vector squareToUniformSphere(const Vector2 &sample) {
    Float z = 1.0f - 2.0f * sample.y();
    Float r = sqrt(1.0f - z*z);
    Float sinPhi = sin(2.0f * M_PI * sample.x());
    Float cosPhi = cos(2.0f * M_PI * sample.x());
    return Vector(r * cosPhi, r * sinPhi, z);
}

Float squareToCosineHemispherePdf(const Vector &d) {
    return INV_PI * d.z();
}

void coordinateSystem(const Vector &_n, Vector &s, Vector &t) {
    static const Matrix3x3 randRot(Eigen::AngleAxis<Float>(0.1f, Vector(0.1f, 0.2f, 0.3f).normalized()).toRotationMatrix());
    static const Matrix3x3 randRotInv = randRot.transpose();
    Vector n = randRot*_n;
    if (std::abs(n.x()) > std::abs(n.y())) {
        Float invLen = 1.0f / std::sqrt( n.x()*n.x() +  n.z()*n.z() );
        t = Vector(n.z()*invLen, 0.0f, -n.x()*invLen);
    } else {
        Float invLen = 1.0f / std::sqrt( n.y()*n.y() +  n.z()*n.z() );
        t = Vector(0.0, n.z()*invLen, -n.y()*invLen);
    }
    s = t.cross(n);
    s = randRotInv*s; t = randRotInv*t;
}

void coordinateSystemAD(const VectorAD &_n, VectorAD &s, VectorAD &t) {
    static const Matrix3x3AD randRot(Eigen::AngleAxis<Float>(0.1f, Vector(0.1f, 0.2f, 0.3f).normalized()).toRotationMatrix());
    static const Matrix3x3AD randRotInv = randRot.transpose();
    VectorAD n = randRot*_n;
    if (std::abs(n.x().val) > std::abs(n.y().val)) {
        FloatAD invLen = static_cast<Float>(1.0)/(n.x()*n.x() + n.z()*n.z()).sqrt();
        t = VectorAD(n.z()*invLen, 0.0f, -n.x()*invLen);
    } else {
        FloatAD invLen = static_cast<Float>(1.0)/(n.y()*n.y() + n.z()*n.z()).sqrt();
        t = VectorAD(0.0f, n.z()*invLen, -n.y()*invLen);
    }
    s = t.cross(n);
    s = randRotInv*s; t = randRotInv*t;
}

Float luminance(const Vector &v) {
    return 0.212671f * v(0) +
           0.715160f * v(1) +
           0.072169f * v(2);
}

FloatAD luminanceAD(const VectorAD &v) {
    return 0.212671f * v(0) +
           0.715160f * v(1) +
           0.072169f * v(2);
}

bool isPixelValueValid(const Spectrum3f &val) {
    for (int i = 0; i < 3; i++) {
        if (val(i) < -1e-6 ||  std::isinf(val(i)) || std::isnan(val(i))) {
            std::cout << "[invalid pixel value] " << val << std::endl;
            return false;
        }
    }
    return true;
}

void progressIndicator(Float progress) {
    int barWidth = 70;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

Float fresnelDielectricExt(Float cosThetaI_, Float &cosThetaT_, Float eta) {
    if (std::abs(eta - 1.0f) < Epsilon) {
        cosThetaT_ = -cosThetaI_;
        return 0.0f;
    }
    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    Float scale = (cosThetaI_ > 0.0f) ? 1.0f/eta : eta,
          cosThetaTSqr = 1.0f - (1.0f - cosThetaI_*cosThetaI_)*(scale*scale);

    /* Check for total internal reflection */
    if (cosThetaTSqr < Epsilon) {
        cosThetaT_ = 0.0f;
        return 1.0f;
    }

    /* Find the absolute cosines of the incident/transmitted rays */
    Float cosThetaI = std::abs(cosThetaI_);
    Float cosThetaT = std::sqrt(cosThetaTSqr);

    Float Rs = (cosThetaI - eta * cosThetaT)
             / (cosThetaI + eta * cosThetaT);
    Float Rp = (eta * cosThetaI - cosThetaT)
             / (eta * cosThetaI + cosThetaT);

    cosThetaT_ = (cosThetaI_ > 0.0f) ? -cosThetaT : cosThetaT;

    /* No polarization -- return the unpolarized reflectance */
    return 0.5f*(Rs*Rs + Rp*Rp);
}

FloatAD fresnelDielectricExtAD(const FloatAD &cosThetaI_, FloatAD &cosThetaT_, const FloatAD &eta) {
    if (std::abs(eta.val - 1.0f) < Epsilon) {
        cosThetaT_ = -cosThetaI_;
        return FloatAD();
    }

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    FloatAD scale;
    if (cosThetaI_ > 0.0f)
        scale = 1.0f/eta;
    else
        scale = eta;
    FloatAD cosThetaTSqr = 1.0f - (1.0f - cosThetaI_.square())*scale.square();

    /* Check for total internal reflection */
    if (cosThetaTSqr < Epsilon) {
        cosThetaT_.zero();
        return FloatAD(1.0f);
    }

    /* Find the absolute cosines of the incident/transmitted rays */
    FloatAD cosThetaI = cosThetaI_.abs();
    FloatAD cosThetaT = cosThetaTSqr.sqrt();

    FloatAD Rs = (cosThetaI - eta * cosThetaT)
               / (cosThetaI + eta * cosThetaT);
    FloatAD Rp = (eta * cosThetaI - cosThetaT)
               / (eta * cosThetaI + cosThetaT);

    cosThetaT_ = (cosThetaI_ > 0.0f) ? -cosThetaT : cosThetaT;

    /* No polarization -- return the unpolarized reflectance */
    return 0.5f*(Rs*Rs + Rp*Rp);
}

Vector refract(const Vector &wi, const Vector &n, Float eta, Float cosThetaT) {
    if (cosThetaT < 0) eta = 1.0f/eta;
    return n*(wi.dot(n)*eta + cosThetaT) - wi*eta;
}

Vector refract(const Vector &wi, const Vector &n, Float eta) {
    assert(std::abs(eta - 1.0) > Epsilon);

    Float cosThetaI = wi.dot(n);
    if (cosThetaI > 0) eta = 1.0f/eta;

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    Float cosThetaTSqr = 1.0f - (1.0f - cosThetaI*cosThetaI)*(eta*eta);

    /* Check for total internal reflection */
    if (cosThetaTSqr < Epsilon) return Vector::Zero();

    return n*(cosThetaI*eta - math::signum(cosThetaI)*std::sqrt(cosThetaTSqr)) - wi*eta;
}

Vector refract(const Vector &wi, const Vector &n, Float eta, Float &cosThetaT, Float &F) {
    Float cosThetaI = wi.dot(n);
    F = fresnelDielectricExt(cosThetaI, cosThetaT, eta);

    if (std::abs(F - 1.0) < Epsilon) /* Total internal reflection */
        return Vector::Zero();

    if (cosThetaT < 0) eta = 1.0f/eta;
    return n*(eta*cosThetaI + cosThetaT) - wi*eta;
}

Array rayIntersectTriangle(const Vector &v0, const Vector &v1, const Vector &v2, const Ray &ray) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pvec = ray.dir.cross(e2);
    auto divisor = pvec.dot(e1);
    // Hack
    if (std::abs(divisor) < Epsilon)
        divisor = (divisor > 0) ? Epsilon : -Epsilon;
    auto s = ray.org - v0;
    auto dot_s_pvec = s.dot(pvec);
    auto u = dot_s_pvec / divisor;
    auto qvec = s.cross(e1);
    auto dot_dir_qvec = ray.dir.dot(qvec);
    auto v = dot_dir_qvec / divisor;
    auto dot_e2_qvec = e2.dot(qvec);
    auto t = dot_e2_qvec / divisor;
    return Vector(u, v, t);
}

ArrayAD rayIntersectTriangleAD(const VectorAD &v0, const VectorAD &v1, const VectorAD &v2, const RayAD &ray) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pvec = ray.dir.cross(e2);
    auto divisor = pvec.dot(e1);
    // Hack
    if (std::abs(divisor.val) < Epsilon)
        divisor = (divisor > 0) ? Epsilon : -Epsilon;
    auto s = ray.org - v0;
    auto dot_s_pvec = s.dot(pvec);
    auto u = dot_s_pvec / divisor;
    auto qvec = s.cross(e1);
    auto dot_dir_qvec = ray.dir.dot(qvec);
    auto v = dot_dir_qvec / divisor;
    auto dot_e2_qvec = e2.dot(qvec);
    auto t = dot_e2_qvec / divisor;
    return ArrayAD(u, v, t);
}

Float computeIntersectionInTri(const Vector& a, const Vector& b0, const Vector& c0, const Vector& b1, const Vector& c1, Float t0) {
    // b0 = a + coeff_b * (b1-a) + 0 * (c1-a)
    Float coeff_b = (b0 - a).norm()/(b1 - a).norm();
    // c0 = a +    0 * (b1-a)    + coeff_c * (c1-a)
    Float coeff_c = (c0 - a).norm()/(c1 - a).norm();
    // p0 = b0 + t0 * (c0 - b0)
    Vector2 barycentric((1 - t0)*coeff_b, t0*coeff_c);
    barycentric /= (barycentric.x()+barycentric.y());
    // p = a + bary.x * (b1-a) + bary.y * (c1-a)
    return barycentric.y();
}