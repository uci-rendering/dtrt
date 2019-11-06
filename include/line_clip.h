#pragma once
#ifndef CLIP_LINE_H__
#define CLIP_LINE_H__

#include "fwd.h"

// From https://www.wikiwand.com/en/Cohen%E2%80%93Sutherland_algorithm
// Thank you wikipedia!

// Cohen–Sutherland clipping algorithm clips a line from
// P0 = (x0, y0) to P1 = (x1, y1) against a rectangle with
// diagonal from (xmin, ymin) to (xmax, ymax).

using OutCode = int;
constexpr int OC_INSIDE = 0;
constexpr int OC_LEFT = 1;
constexpr int OC_RIGHT = 2;
constexpr int OC_BOTTOM = 4;
constexpr int OC_TOP = 8;

OutCode compute_out_code(const Vector2 &v, float xmin, float xmax, float ymin, float ymax) {
    OutCode code;
    code = OC_INSIDE;           // initialised as being inside of [[clip window]]
    if (v[0] < xmin - Epsilon) {        // to the left of clip window
        code |= OC_LEFT;
    } else if (v[0] > xmax + Epsilon) { // to the right of clip window
        code |= OC_RIGHT;
    }
    if (v[1] < ymin - Epsilon) {        // below the clip window
        code |= OC_BOTTOM;
    } else if (v[1] > ymax + Epsilon) { // above the clip window
        code |= OC_TOP;
    }
    return code;
};

bool clip_line(const Vector2AD &v0, const Vector2AD &v1, Vector2AD &v0c, Vector2AD &v1c,
               float xmin = 0.0, float xmax = 1.0, float ymin = 0.0, float ymax = 1.0) {
    // Compute the bit code for a point (x, y) using the clip rectangle
    // bounded diagonally by (xmin, ymin), and (xmax, ymax)

    // compute outcodes for P0, P1, and whatever point lies outside the clip rectangle
    OutCode outcode0 = compute_out_code(v0.val, xmin, xmax, ymin, ymax);
    OutCode outcode1 = compute_out_code(v1.val, xmin, xmax, ymin, ymax);
    bool accept = false;
    v0c = v0;
    v1c = v1;

    while (true) {
        if (!(outcode0 | outcode1)) {
            // bitwise OR is 0: both points inside window; trivially accept and exit loop
            accept = true;
            break;
        } else if (outcode0 & outcode1) {
            // bitwise AND is not 0: both points share an outside zone (LEFT, RIGHT, TOP,
            // or BOTTOM), so both must be outside window; exit loop (accept is false)
            break;
        } else {
            // failed both tests, so calculate the line segment to clip
            // from an outside point to an intersection with clip edge
            Vector2AD v;

            // At least one endpoint is outside the clip rectangle; pick it.
            OutCode outcodeOut = outcode0 ? outcode0 : outcode1;

            // Now find the intersection point;
            // use formulas:
            //   slope = (y1 - y0) / (x1 - x0)
            //   x = x0 + (1 / slope) * (ym - y0), where ym is ymin or ymax
            //   y = y0 + slope * (xm - x0), where xm is xmin or xmax
            // No need to worry about divide-by-zero because, in each case, the
            // outcode bit being tested guarantees the denominator is non-zero
            if (outcodeOut & OC_TOP) {           // point is above the clip window
                v = v0c + (v1c - v0c) * (ymax - v0c.y().val) / (v1c.y().val - v0c.y().val);
            } else if (outcodeOut & OC_BOTTOM) { // point is below the clip window
                v = v0c + (v1c - v0c) * (ymin - v0c.y().val) / (v1c.y().val - v0c.y().val);
            } else if (outcodeOut & OC_RIGHT) {  // point is to the right of clip window
                v = v0c + (v1c - v0c) * (xmax - v0c.x().val) / (v1c.x().val - v0c.x().val);
            } else if (outcodeOut & OC_LEFT) {   // point is to the left of clip window
                v = v0c + (v1c - v0c) * (xmin - v0c.x().val) / (v1c.x().val - v0c.x().val);
            }           

            // Now we move outside point to intersection point to clip
            // and get ready for next pass.
            if (outcodeOut == outcode0) {
                v0c = v;
                outcode0 = compute_out_code(v0c.val, xmin, xmax, ymin, ymax);
            } else {
                v1c = v;
                outcode1 = compute_out_code(v1c.val, xmin, xmax, ymin, ymax);
            }
        }
    }
    return accept;
}

#endif