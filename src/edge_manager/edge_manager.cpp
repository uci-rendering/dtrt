#include "edge_manager.h"
#include "scene.h"
#include "camera.h"
#include "line_clip.h"

EdgeManager::EdgeManager(const Scene& scene) {
	primary_edges_distrb.clear();
    const std::vector<const Shape*>& shape_list = scene.shape_list;
    for (size_t i = 0; i < shape_list.size(); i++) {
        int bsdf_id = shape_list[i]->bsdf_id;
        if (!scene.bsdf_list[bsdf_id]->isNull() || shape_list[i]->light_id >= 0)
        	initPrimaryEdgesFromShape(*shape_list[i], scene.camera);
    }
    if (primary_edges_distrb.size() > 0)
        primary_edges_distrb.normalize();
}

void EdgeManager::initPrimaryEdgesFromShape(const Shape& shape, const Camera& cam) {
    int num_edges = shape.edges.size();
    for (int idx = 0; idx < num_edges; idx++) {
        const Edge& edge = shape.getEdge(idx);
        int isSihoulette = shape.isSihoulette(edge, cam.cpos.val);
        if ( isSihoulette > 0) {
            VectorAD v0 = shape.getVertexAD(edge.v0);
            VectorAD v1 = shape.getVertexAD(edge.v1);
            projectEdgeToScreen(v0, v1, cam);
        }
    }
}

void EdgeManager::projectEdgeToScreen(VectorAD v0, VectorAD v1, const Camera& cam) {
	float clip_near = cam.clip_near;
	Float z0 = (v0 - cam.cpos).dot(cam.cframe.n).val,
		  z1 = (v1 - cam.cpos).dot(cam.cframe.n).val;
    if (z0 < clip_near && z1 < clip_near)
    	return;
    else if (z0 < clip_near)
    	v0 += (clip_near-z0)/(z1-z0) * (v1-v0);
    else if (z1 < clip_near)
    	v1 += (clip_near-z1)/(z0-z1) * (v0-v1);

    // Vertex position in camera local space
    Vector v0_local = xfm_point(cam.world_to_cam, v0.val);
    Vector v1_local = xfm_point(cam.world_to_cam, v1.val);

    // Vertex(AD) on image plane (Perspective Proj.)
    v0 = (v0 - cam.cpos)/(v0 - cam.cpos).dot(cam.cframe.n);
    v1 = (v1 - cam.cpos)/(v1 - cam.cpos).dot(cam.cframe.n);
    v0 = cam.cframe.toLocal(v0);
    v1 = cam.cframe.toLocal(v1);

    // Vertex(AD) in Screen space (0 ~ 1)
    auto fov_factor = cam.cam_to_ndc(0,0);
    auto aspect_ratio = Float(cam.width) / Float(cam.height);
    Vector2AD v0_screen_raw, v1_screen_raw;
    Vector2AD v0_screen, v1_screen;
    v0_screen_raw.x() = v0.x() * fov_factor * 0.5f + 0.5f;
    v1_screen_raw.x() = v1.x() * fov_factor * 0.5f + 0.5f;
    v0_screen_raw.y() = v0.y() * fov_factor * (-0.5f) * aspect_ratio + 0.5f;
    v1_screen_raw.y() = v1.y() * fov_factor * (-0.5f) * aspect_ratio + 0.5f;

    float xmin = 0.0f, xmax = 1.0f, ymin = 0.0f, ymax = 1.0f;
    const CropRectangle& rect = cam.rect;
    if (rect.isValid()) {
    	xmin = (float) rect.offset_x / cam.width;
    	xmax = (float)(rect.offset_x + rect.crop_width)/ cam.width;
    	ymin = (float) rect.offset_y / cam.height;
    	ymax = (float)(rect.offset_y + rect.crop_height) / cam.height;
    }


    if (clip_line(v0_screen_raw, v1_screen_raw, 
    			  v0_screen,     v1_screen,
    			  xmin, xmax, ymin, ymax))
    {

		Vector v0_plane = v0.val, v1_plane = v1.val;

	    bool v0_clipped = (v0_screen_raw.val - v0_screen.val).norm() > Epsilon;
	    Vector tmp0 = v0_local, tmp1 = v1_local;
	    if (v0_clipped) {
	        Vector v0_plane_clipped((v0_screen.val.x() - 0.5f) * 2.f / fov_factor,
	                       	 		(v0_screen.val.y() - 0.5f) * (-2.f) / (aspect_ratio*fov_factor), 
	                       	 		1.0);
	        Float t0 = (v0_plane_clipped - v0.val).norm()/(v1.val - v0.val).norm();
	        v0_plane = v0.val + t0 * (v1.val - v0.val);
	        Float t1 = computeIntersectionInTri(Vector::Zero(), v0.val, v1.val, tmp0, tmp1, t0);
	        v0_local = tmp0 + t1 * (tmp1-tmp0);
	    }
	    bool v1_clipped = (v1_screen_raw.val - v1_screen.val).norm() > Epsilon;
	    if (v1_clipped) {
	        Vector v1_plane_clipped((v1_screen.val.x()-0.5f) * 2.f / fov_factor,
	                       			(v1_screen.val.y()-0.5f) * (-2.f) / (aspect_ratio*fov_factor),
	                       			1.0);
	        Float t0 = (v1_plane_clipped - v0.val).norm()/(v1.val-v0.val).norm();
	        v1_plane = v0.val + t0 * (v1.val-v0.val);
	        Float t1 = computeIntersectionInTri(Vector::Zero(), v0.val, v1.val, tmp0, tmp1, t0);
	        v1_local = tmp0 + t1 * (tmp1-tmp0);
	    }

	    v0_screen.x() = v0_screen.x() * cam.width;
	    v0_screen.y() = v0_screen.y() * cam.height;
	    v1_screen.x() = v1_screen.x() * cam.width;
	    v1_screen.y() = v1_screen.y() * cam.height;
        primary_edges.push_back(PrimaryEdge{v0_screen, v1_screen,
        									v0_plane,  v1_plane,
        									v0_local,  v1_local});
        primary_edges_distrb.append((v0_screen.val - v1_screen.val).norm());
    }
}

Float EdgeManager::samplePrimaryEdge(const Camera& cam, Float rnd, Vector2i& xyPixel, Vector2AD& p, Vector2& norm) const {
    int idx_edge = primary_edges_distrb.sampleReuse(rnd);
    const Vector2AD& v0 = primary_edges[idx_edge].v0s,
    				 v1 = primary_edges[idx_edge].v1s;
    p = v0 + rnd * (v1-v0);
    auto fov_factor = cam.cam_to_ndc(0,0);
    auto aspect_ratio = Float(cam.width) / Float(cam.height);
    Vector tmp((p.val.x()/cam.width  - 0.5f) * 2.f / fov_factor,
               (p.val.y()/cam.height - 0.5f) * (-2.f) / (aspect_ratio*fov_factor), 
               1.0f);
    xyPixel.x() = std::floor(p.val.x());
    xyPixel.y() = std::floor(p.val.y());
    norm.x() = (v1.val.y() - v0.val.y());
    norm.y() = (v0.val.x() - v1.val.x());
    norm.normalize();
    p.x() = p.x() - xyPixel.x();
    p.y() = p.y() - xyPixel.y();

    Float t0 = (tmp - primary_edges[idx_edge].v0p).norm() / (primary_edges[idx_edge].v1p - primary_edges[idx_edge].v0p).norm();
    Float t1 = computeIntersectionInTri(Vector::Zero(), primary_edges[idx_edge].v0p, primary_edges[idx_edge].v1p,
                                                    	primary_edges[idx_edge].v0c, primary_edges[idx_edge].v1c, t0);
    return (primary_edges[idx_edge].v0c + t1 * (primary_edges[idx_edge].v1c - primary_edges[idx_edge].v0c)).norm();
}