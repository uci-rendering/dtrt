#pragma once
#ifndef EDGE_TEST0_H__
#define EDGE_TEST0_H__


void edge_test0();

// Pixel integral + "Solid Angle"
struct Prob0_7
{
	Prob0_7();
	void run();
	Float eval(const Ray &ray);

	void computeReference();
	void computeEdgeIntegral();
	void computeFiniteDiff(Float delta, const std::string& fn);

	Camera camera;
	Shape shape;
	Float fov;
	VectorAD a, b, c;
};


#endif