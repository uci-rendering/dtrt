#include "edge.h"
#include "shape.h"
#include "intersection.h"
#include "sampler.h"
#include "utils.h"
#include "stats.h"
#include "camera.h"
#include <omp.h>
#include <algorithm>

Float rayIntersectTri(const Vector &v0, const Vector &v1, const Vector &v2, const Ray &ray) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pvec = ray.dir.cross(e2);
    auto divisor = pvec.dot(e1);
    if (std::abs(divisor) < Epsilon)
    	return std::numeric_limits<Float>::infinity();
    auto s = ray.org - v0;
    auto dot_s_pvec = s.dot(pvec);
    auto u = dot_s_pvec / divisor;
    if (u < -Epsilon || u > 1.0+Epsilon)
    	return std::numeric_limits<Float>::infinity();
    auto qvec = s.cross(e1);
    auto dot_dir_qvec = ray.dir.dot(qvec);
    auto v = dot_dir_qvec / divisor;
    if (v < -Epsilon || u + v > 1.0+Epsilon)
    	return std::numeric_limits<Float>::infinity();    
    auto dot_e2_qvec = e2.dot(qvec);
    auto t = dot_e2_qvec / divisor;
    if ( t > Epsilon)
    	return t;
    else
    	return std::numeric_limits<Float>::infinity();
}

// Triangle emitter(Moving) + Triangle Blocker(Moving)
struct Prob0_1
{
	Prob0_1();
	Float eval(const Vector &dir);
	void main();

	Vector a0, b0, c0, a1, b1, c1;
	Shape shape;
};

Prob0_1::Prob0_1() {
	int nvertices = 6;
	int nfaces = 2;
	float vtxPositions[nvertices*3] = {
		// Emitter
        -1.0f, -1.0f, 1.0f,
         1.0f, -1.0f, 1.0f,
         0.0f,  1.0f, 1.0f,
        // Blocker
        -0.5f, -1.0f, 0.5f,
         1.5f, -1.0f, 0.5f,
         0.5f,  1.0f, 0.5f
    };

    int vtxIndices[nfaces*3] = {
        0, 1, 2,
        3, 4, 5
    };
	shape = Shape(vtxPositions, vtxIndices, nullptr, nullptr, nvertices, nfaces, -1, -1, -1, -1);
	Eigen::Matrix<Float, 3, 6> vtxVelocities;
	vtxVelocities << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
					 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
					 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f;
	shape.initVelocities(vtxVelocities, 0);

	a0 = shape.getVertex(0);
	b0 = shape.getVertex(1);
	c0 = shape.getVertex(2);	
	a1 = shape.getVertex(3);
	b1 = shape.getVertex(4);
	c1 = shape.getVertex(5);
}

Float Prob0_1::eval(const Vector &dir) {
	Ray ray(Vector::Zero(), dir);
	Float dist = rayIntersectTri(a0, b0, c0, ray);
	if (dist < std::numeric_limits<Float>::infinity())
		return rayIntersectTri(a1, b1, c1, ray) < dist ? 0.0 : 1.0;
	else
		return 0.0;
}

void Prob0_1::main() {
	Float delta = 1e-3;
    Vector a2 = a1 + shape.getVertexAD(3).grad()*delta,
           b2 = b1 + shape.getVertexAD(4).grad()*delta,
           c2 = c1 + shape.getVertexAD(5).grad()*delta;

    Float area = shape.getArea(0);
	long long N = 100000000LL;
    int nworker = omp_get_num_procs();
    std::vector<Statistics> stats(nworker);
    std::vector<RndSampler> samplers;
    for (int i = 0; i < nworker; i++)
    	samplers.push_back(RndSampler(0, i));
#pragma omp parallel for num_threads(nworker)
	for (long long omp_i = 0; omp_i < N; omp_i++) {
		const int tid = omp_get_thread_num();
		auto &stat = stats[tid];
		Vector pos, norm;
		// Sample on Emitter
		shape.samplePosition(0, samplers[tid].next2D(), pos, norm);
        Float distSqr = pos.squaredNorm(), dist = std::sqrt(distSqr);
        pos /= dist;
        Float v1 = area * std::abs(norm.dot(pos)) / distSqr;
        Float v2 = v1;
        Ray ray(Vector::Zero(), pos);
        if ( rayIntersectTri( a1, b1, c1, ray ) < dist )
            v1 = 0.0;
        if ( rayIntersectTri( a2, b2, c2, ray ) < dist )
            v2 = 0.0;
        stat.push((v2 - v1)/delta);
	}
    for ( int i = 1; i < nworker; ++i )
        stats[0].push(stats[i]);
    printf("[Test 1] FD  : %.2le +- %.2le\n", stats[0].getMean(), stats[0].getCI());

	// Edeg sampling
	N = 10000000LL;
	delta = 1e-4;
	VectorAD x;
	x.val = Vector::Zero();
	x.grad() = Vector::Zero();
	for (int i = 0; i < nworker; ++i)
		stats[i].reset();
#pragma omp parallel for num_threads(nworker)
	for (long long omp_i = 0; omp_i < N; omp_i++) {
		const int tid = omp_get_thread_num();
		auto &stat = stats[tid];
		Float pdf = 1.0;
		Float t;
		const Edge& edge = shape.sampleEdge(samplers[tid].next1D(), t, pdf);
		auto v0 = shape.getVertexAD(edge.v0);
		auto v1 = shape.getVertexAD(edge.v1);
		Vector e = (v1.val-v0.val).normalized();			

		if (v0.grad().isZero() && v1.grad().isZero()) {
			stat.push(0.0);
		} else {		
			auto y = v0 + t*(v1 - v0);
			Vector n = v0.val.cross(v1.val).normalized();
			Vector v = (y - x).normalized().grad();
			Float dist = y.val.norm();
			y = y/dist;
			Float deltaFunc = eval(y.val-delta*v) - eval(y.val+delta*v);
			Float cosTheta = y.val.dot(e),
				  sinTheta = std::sqrt(1.0-cosTheta*cosTheta);
			Float val = deltaFunc*std::abs(n.dot(v))*sinTheta / (pdf*dist);
			stat.push(val);
		}
	}

    for ( int i = 1; i < nworker; ++i )
        stats[0].push(stats[i]);
    printf("[Test 1] Edge : %.2le +- %.2le\n", stats[0].getMean(), stats[0].getCI());
}

void projectPointToCamera(VectorAD v, const Camera& camera, Vector2AD& p) {
    v = (v - camera.cpos)/(v - camera.cpos).dot(camera.cframe.n);
    v = camera.cframe.toLocal(v);
    auto fov_factor = camera.cam_to_ndc(0,0);
    auto aspect_ratio = Float(camera.width) / Float(camera.height);
    p.x() = v.x() * fov_factor * 0.5f + 0.5;
    p.y() = v.y() * fov_factor * (-0.5f) * aspect_ratio + 0.5f;
}

bool insideTriangle(const Vector2& p, const Vector2AD& a, const Vector2AD& b, const Vector2AD& c) {
	Vector2 p_c = p - c.val;
	Vector2 b_c = b.val - c.val;
	Vector2 a_c = a.val - c.val;
	Float u = (b_c.y()*p_c.x()-b_c.x()*p_c.y())/(b_c.y()*a_c.x()-b_c.x()*a_c.y());
	Float v = (-a_c.y()*p_c.x()+a_c.x()*p_c.y())/(b_c.y()*a_c.x()-b_c.x()*a_c.y());
	return (u >= 0.0 && u <= 1.0) && (v >= 0.0 && v <= 1.0) && (u+v >= 0.0 && u+v <= 1.0);
}

void edgeAxisIntersect(const Vector2AD& a, const Vector2AD& b, const Vector2AD& c, bool is_x, Float th, Float th_low, Float th_high, std::vector<Vector2AD>& vsets) {
	FloatAD a_valAD = is_x ? a.x() : a.y();
	FloatAD b_valAD = is_x ? b.x() : b.y();
	FloatAD c_valAD = is_x ? c.x() : c.y();
	FloatAD thAD;
	thAD.val = th;

	FloatAD tAD = (thAD-a_valAD)/(b_valAD-a_valAD);
	if (tAD.val >= 0.0 && tAD.val <= 1.0) {
		Vector2AD p = a + tAD * (b-a);
		Float val = is_x ? p.val.y() : p.val.x();
		if (val > th_low && val < th_high) {
			vsets.push_back(p);
		}
	}
	tAD = (thAD-a_valAD)/(c_valAD-a_valAD);
	if (tAD.val >= 0.0 && tAD.val <= 1.0) {
		Vector2AD p = a + tAD*(c-a);
		Float val = is_x ? p.val.y() : p.val.x();
		if (val > th_low && val < th_high) {		
			vsets.push_back(p);
		}
	}
	tAD = (thAD-b_valAD)/(c_valAD-b_valAD);
	if (tAD.val >= 0.0 && tAD.val <= 1.0) {
		Vector2AD p = b + tAD*(c-b);
		Float val = is_x ? p.val.y() : p.val.x();
		if (val > th_low && val < th_high) {			
			vsets.push_back(p);			
		}
	}
}

void findAllValidVertices(const Vector2AD& a, const Vector2AD& b, const Vector2AD& c, 
						  Float x_min, Float x_max, Float y_min, Float y_max,
						  std::vector<Vector2AD>& vsets) {

	if (a.val.x() > x_min && a.val.x() < x_max && a.val.y() > y_min && a.val.y() < y_max)
		vsets.push_back(a);
	if (b.val.x() > x_min && b.val.x() < x_max && b.val.y() > y_min && b.val.y() < y_max)
		vsets.push_back(b);
	if (c.val.x() > x_min && c.val.x() < x_max && c.val.y() > y_min && c.val.y() < y_max)
		vsets.push_back(c);
	if (vsets.size() == 3) return;
	// check if 4 points
	Vector2 v_00(x_min, y_min), v_10(x_max, y_min), v_01(x_min, y_max), v_11(x_max, y_max);
	Vector2AD v_corner;
	if (insideTriangle(v_00, a, b, c)) {
		v_corner.val = v_00;
		vsets.push_back(v_corner);
	}
	if (insideTriangle(v_01, a, b, c)) {
		v_corner.val = v_01;
		vsets.push_back(v_corner);
	}
	if (insideTriangle(v_10, a, b, c)) {
		v_corner.val = v_10;
		vsets.push_back(v_corner);
	}
	if (insideTriangle(v_11, a, b, c)) {
		v_corner.val = v_11;
		vsets.push_back(v_corner);
	}
	// Check if there is border_edge intersection
	edgeAxisIntersect(a, b, c, true,  x_min, y_min-Epsilon, y_max+Epsilon, vsets);
	edgeAxisIntersect(a, b, c, true,  x_max, y_min-Epsilon, y_max+Epsilon, vsets);	
	edgeAxisIntersect(a, b, c, false, y_min, x_min-Epsilon, x_max+Epsilon, vsets);	
	edgeAxisIntersect(a, b, c, false, y_max, x_min-Epsilon, x_max+Epsilon, vsets);

}

void reorderVerts(std::vector<Vector2AD>& verts) {
	if(verts.size() > 0) {
		std::vector<Vector2AD> verts_ordered;
		Vector2AD anchor = verts[0];
		std::vector<std::pair<Float,int> > angles(verts.size()-1);
		for (size_t i = 1; i < verts.size(); i++)
			angles[i-1] = std::make_pair((verts[i]-anchor).val.normalized().x(), i);
		std::sort(angles.begin(), angles.end());
		verts_ordered.push_back(verts[0]);
		for (size_t i = 1; i < verts.size(); i++) {
			const Vector2AD& v = verts[angles[i-1].second];
			if ((v.val - anchor.val).norm() > Epsilon)
				verts_ordered.push_back(v);
		}
		verts.swap(verts_ordered);
	}
}

// Pixel integral + "Solid Angle"
struct Prob0_7
{
	Prob0_7();
	Float eval(const Ray &ray);
	void main();


	Camera camera;
	Shape shape;
	Float fov;
	VectorAD a, b, c;
};

Prob0_7::Prob0_7() {
	// Initialize Shape...
	int nvertices = 3;
	int nfaces = 1;
	float vtxPositions[nvertices*3] = {
        -1.0f, -1.0f, 5.0f,
         1.0f, -1.0f, 5.0f,
         0.0f,  1.0f, 5.0f,
    };

    int vtxIndices[nfaces*3] = {
        0, 1, 2,
    };
	shape = Shape(vtxPositions, vtxIndices, nullptr, nullptr, nvertices, nfaces, -1, -1, -1, -1);
	Eigen::Matrix<Float, 3, 3> vtxVelocities;
	vtxVelocities << 1.0f, 0.0f, 0.0f,
					 0.0f, 1.0f, 0.0f,
					 0.0f, 0.0f, 1.0f;
	shape.initVelocities(vtxVelocities, 0);
	a = shape.getVertexAD(0);
	b = shape.getVertexAD(1);
	c = shape.getVertexAD(2);

	// Initialize Camera
	Vector ang_vel = Vector(0,0,0);
	Vector pos_vel = Vector(0,0,0);
	VectorAD x, w;
	x.val = Vector::Zero();
	x.grad() = pos_vel;
	w.val = (a.val + 0.5*(b.val-a.val) + 0.5*(c.val-a.val)).normalized();
	w.grad() = ang_vel.cross(w.val);

	VectorAD right, up;
	coordinateSystemAD(w, right, up);

	fov = 90.0f;
	camera = Camera(x.val, up.val, w.val, fov, 1, 1, 0.0, -1);
	camera.initVelocities(pos_vel, ang_vel,0);
	camera.initCameraEdges(shape);

}

Float Prob0_7::eval(const Ray &ray) {
	return std::isfinite(rayIntersectTri(a.val, b.val, c.val, ray)) ? 1.0: 0.0;
}

void Prob0_7::main() {
	// AD validation
	const VectorAD& x = camera.cpos;
	const VectorAD& w = camera.cframe.n;
	const VectorAD& s = camera.cframe.s;
	const VectorAD& t = camera.cframe.t;
	VectorAD a1 = x + (a - x)/(a - x).dot(w),
			 b1 = x + (b - x)/(b - x).dot(w),
			 c1 = x + (c - x)/(c - x).dot(w);

	if ( std::abs((a1.val - (x.val + w.val)).dot(w.val)) > Epsilon ||
		 std::abs((b1.val - (x.val + w.val)).dot(w.val)) > Epsilon ||
		 std::abs((c1.val - (x.val + w.val)).dot(w.val)) > Epsilon )
	{
		std::cerr << "Badness 1: Behind camera" << std::endl;
		return;
	}

	Float aspect_ratio = (Float)camera.height/camera.width;
	Vector2 viewFrustrum;
	viewFrustrum.x() = tan(0.5*fov);
	viewFrustrum.y() = viewFrustrum.x() * aspect_ratio;

	if (std::abs(a1.val.x()) > viewFrustrum.x() || std::abs(a1.val.y()) > viewFrustrum.y() ||
		std::abs(b1.val.x()) > viewFrustrum.x() || std::abs(b1.val.y()) > viewFrustrum.y() ||
		std::abs(c1.val.x()) > viewFrustrum.x() || std::abs(c1.val.y()) > viewFrustrum.y() )
	{
		std::cerr << "Badness 2: Clipped by screen" << std::endl;
		return;
	}

	VectorAD a2(a1.dot(s), a1.dot(t), 0.0),
			 b2(b1.dot(s), b1.dot(t), 0.0),
			 c2(c1.dot(s), c1.dot(t), 0.0);
	Float pixel_area = square(2*tan(0.5*fov * M_PI/180.0)/camera.width);
	FloatAD val_ad = 0.5*((b2 - a2).cross(c2 - a2)).norm()/pixel_area;
	printf("[Test 2] AD (grad)  : %.2le\n", val_ad.grad());
	// printf("[Test 2] AD (val)  : %.2le\n", val_ad.val);


    Vector2AD a3,b3,c3;
	projectPointToCamera(a, camera, a3);
	projectPointToCamera(b, camera, b3);
	projectPointToCamera(c, camera, c3);

	int num_pixel = camera.getNumPixels();
	std::vector<std::vector<Vector2AD>> verts(num_pixel);
	Float x_range = 1.0/camera.width,
		  y_range = 1.0/camera.height;
	for (int i = 0; i < num_pixel; i++) {
		Float x_min = (i % camera.width) * x_range,
			  x_max = x_min + x_range,
			  y_min = (i / camera.width) * y_range,
			  y_max = y_min + y_range;
		findAllValidVertices(a3, b3, c3, x_min, x_max, y_min, y_max, verts[i]);
		reorderVerts(verts[i]);
		verts[i].erase(std::unique(verts[i].begin(), verts[i].end(), [](const Vector2AD& a, const Vector2AD& b) { return (a.val-b.val).norm() < Epsilon;} ),
						verts[i].end());
		// for (size_t k= 0; k < verts[i].size(); k++)
		// 	printf("[pixel %d] (%.3f , %.3f)\n", i, verts[i][k].val.x(), verts[i][k].val.y());
		FloatAD area;
		if (verts[i].size() >= 3) {
			for (size_t k = 1; k < verts[i].size()-1; k++) {
				VectorAD e1(verts[i][k].x() - verts[i][0].x(), verts[i][k].y() - verts[i][0].y(), 0.0);
				VectorAD e2(verts[i][k+1].x() - verts[i][0].x(), verts[i][k+1].y() - verts[i][0].y(), 0.0);
				area += 0.5*(e2.cross(e1)).norm();
			}
			area *= num_pixel;
		}
		printf("[pixel %d] val = %.2le, grad = %.2le\n", i, area.val, area.grad());
	}



	long long N = 100000000LL;
	Float delta = 1e-4;
    int nworker = omp_get_num_procs();
    std::vector<Statistics> stats(nworker);
    std::vector<RndSampler> samplers;
    for (int i = 0; i < nworker; i++)
    	samplers.push_back(RndSampler(0, i));

#pragma omp parallel for num_threads(nworker)
	for (long long omp_i = 0; omp_i < N; omp_i++) {
		// const int tid = 0;
		const int tid = omp_get_thread_num();
		Vector2i ipixel;
		Vector2AD y;
		Vector2 norm;
		camera.sampleEdge(samplers[tid].next1D(), ipixel, y, norm);
		Ray ray_p = camera.samplePrimaryRay(ipixel.x(), ipixel.y(), y.val + y.grad()*delta);
		Ray ray_m = camera.samplePrimaryRay(ipixel.x(), ipixel.y(), y.val - y.grad()*delta);

		Float deltaFunc = eval(ray_m) - eval(ray_p);
		Float val = std::abs(norm.dot(y.grad()))*deltaFunc*camera.totLen;
		stats[tid].push(val);
	}

	for ( int i = 1; i < nworker; ++i ) stats[0].push(stats[i]);
	printf("[Test 2] Edge  : %.2le +- %.2le\n", stats[0].getMean(), stats[0].getCI());
}

struct Prob0_8
{
	Prob0_8();
	Float eval(const Ray &ray);
	void main();

	Camera camera;
	Shape shape;
	Float fov;
	VectorAD a, b, c;	
};


Prob0_8::Prob0_8() {
	// Initialize Shape...
	int nvertices = 3;
	int nfaces = 1;
	float vtxPositions[nvertices*3] = {
        -1.0f,  1.0f, 2.0f,
         1.0f,  0.0f, 2.0f,
        -1.0f, -1.0f, 2.0f,
    };

    int vtxIndices[nfaces*3] = {
        0, 1, 2,
    };
	shape = Shape(vtxPositions, vtxIndices, nullptr, nullptr, nvertices, nfaces, -1, -1, -1, -1);
	Eigen::Matrix<Float, 3, 3> vtxVelocities;
	vtxVelocities << 0.0f, 0.0f, 0.0f,
					 0.0f, 0.0f, 0.0f,
					 1.0f, 1.0f, 1.0f;
	shape.initVelocities(vtxVelocities, 0);
	a = shape.getVertexAD(0);
	b = shape.getVertexAD(1);
	c = shape.getVertexAD(2);

	// Initialize Camera
	Vector ang_vel = Vector(0,0,0);
	Vector pos_vel = Vector(0,0,0);
	VectorAD x, w;
	x.val = Vector::Zero();
	x.grad() = pos_vel;
	// w.val = (a.val + 0.5*(b.val-a.val) + 0.5*(c.val-a.val)).normalized();
	w.val = Vector(0,0,1);
	w.grad() = ang_vel.cross(w.val);

	VectorAD right, up;
	coordinateSystemAD(w, right, up);

	fov = 90.0f;
	camera = Camera(x.val, up.val, w.val, fov, 3, 3, 0.0, -1);

	camera.initVelocities(pos_vel, ang_vel, 0);
	camera.initCameraEdges(shape);
    int num_edges = camera.edges.size();
    if (num_edges > 0)
        camera.cdf_edges[0] /= camera.totLen;
    for (int idx = 1; idx < num_edges; idx++)
        camera.cdf_edges[idx] = camera.cdf_edges[idx]/camera.totLen + camera.cdf_edges[idx-1];	
}

Float Prob0_8::eval(const Ray &ray) {
	return std::isfinite(rayIntersectTri(a.val, b.val, c.val, ray)) ? 1.0: 0.0;
}


void Prob0_8::main() {
	long long N = 10000000000LL;
	Float delta = 1e-5;
    int nworker = omp_get_num_procs();
    int num_pixel = camera.getNumPixels();
    std::vector<Statistics> stats(num_pixel*nworker);
    std::vector<RndSampler> samplers;
    for (int i = 0; i < nworker; i++)
    	samplers.push_back(RndSampler(0, i));


    Vector2AD a2,b2,c2;
	projectPointToCamera(a, camera, a2);
	projectPointToCamera(b, camera, b2);
	projectPointToCamera(c, camera, c2);
	// printf("v0 = (%.5f, %.5f)\n", a2.val.x(), a2.val.y());
	// printf("v1 = (%.5f, %.5f)\n", b2.val.x(), b2.val.y());
	// printf("v2 = (%.5f, %.5f)\n", c2.val.x(), c2.val.y());

	std::vector<std::vector<Vector2AD>> verts(num_pixel);
	Float x_range = 1.0/camera.width,
		  y_range = 1.0/camera.height;
	Float avg = 0;
	for (int i = 0; i < num_pixel; i++) {
		Float x_min = (i % camera.width) * x_range,
			  x_max = x_min + x_range,
			  y_min = (i / camera.width) * y_range,
			  y_max = y_min + y_range;
		findAllValidVertices(a2, b2, c2, x_min, x_max, y_min, y_max, verts[i]);
		reorderVerts(verts[i]);
		verts[i].erase(std::unique(verts[i].begin(), verts[i].end(), [](const Vector2AD& a, const Vector2AD& b) { return (a.val-b.val).norm() < Epsilon;} ),
						verts[i].end());

		FloatAD area;
		if (verts[i].size() >= 3) {
			for (size_t k = 1; k < verts[i].size()-1; k++) {
				VectorAD e1(verts[i][k].x() - verts[i][0].x(), verts[i][k].y() - verts[i][0].y(), 0.0);
				VectorAD e2(verts[i][k+1].x() - verts[i][0].x(), verts[i][k+1].y() - verts[i][0].y(), 0.0);
				area += 0.5*(e2.cross(e1)).norm();
			}
			area *= num_pixel;
		}
		avg += area.grad();
		printf("[pixel %d] grad = %.2le\n", i, area.grad());
	}
	printf("[avg] grad = %.2le\n", avg/num_pixel);


	std::cout << "Edge Sampling result result..." << std::endl;
	N = 100000000LL;
	delta = 1e-4;
#pragma omp parallel for num_threads(nworker)
	for (long long omp_i = 0; omp_i < N; omp_i++) {
		// const int tid = 0;
		const int tid = omp_get_thread_num();
		Vector2i ipixel;
		Vector2AD y;
		Vector2 norm;
		camera.sampleEdge(samplers[tid].next1D(), ipixel, y, norm);
		Ray ray_p = camera.samplePrimaryRay(ipixel.x(), ipixel.y(), y.val + norm*delta);
		Ray ray_m = camera.samplePrimaryRay(ipixel.x(), ipixel.y(), y.val - norm*delta);

		// Ray ray = camera.samplePrimaryRay(ipixel.x(), ipixel.y(), y.val);
		// Float scalar = 5.0/ray.dir.z();

		int index = ipixel.x() + ipixel.y() * camera.width;

		Float deltaFunc = eval(ray_m) - eval(ray_p);
		Float val = norm.dot(y.grad()) * deltaFunc*camera.totLen;
		for (int i = 0; i < num_pixel; i++) {
			if ( i == index)
				stats[index*nworker + tid].push(val);
			else
				stats[i*nworker + tid].push(0.0);
		}
	}

	for ( int k = 0; k < num_pixel; k++) {
		for ( int i = 1; i < nworker; ++i ) {
			stats[k*nworker].push(stats[k*nworker+i]);
		}
		Float mean = (stats[k*nworker].getN() == 0) ? 0.0 : stats[k*nworker].getMean();
		Float CI = (stats[k*nworker].getN() == 0) ? 0.0 : stats[k*nworker].getCI();
		printf("pixel[%d]  : %.2le +- %.2le\n", k, mean, CI);
	}

	for (int i = 1; i < num_pixel; i++)
		stats[0].push(stats[i*nworker]);
	printf("overall  : %.2le +- %.2le\n", stats[0].getMean(), stats[0].getCI());

}



void edge_test() {
	Prob0_8 prob0_8;
	prob0_8.main();	
}