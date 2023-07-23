#include <iostream>
#include <glm/glm.hpp>
#include <tinynurbs/tinynurbs.h>
#include <emmintrin.h>
#include <LinearSearch.h>
#include <MatrixBlend.h>


using namespace std;

int main(){

    tinynurbs::Curve<float> crv; // Planar curve using float32
    crv.control_points = {glm::vec3(-1, 0, 0), // std::vector of 3D points
                        glm::vec3(0, 1, 0),
                        glm::vec3(1, 0, 0)
                        };
    crv.knots = {0, 0, 0, 1, 1, 1}; // std::vector of floats
    crv.degree = 2;
    glm::vec3 pt = tinynurbs::curvePoint(crv, 0.f);
    // Outputs a point [-1, 0]
    glm::vec3 tgt = tinynurbs::curveTangent(crv, 0.5f);
    // Outputs a vector [1, 0]  
    tinynurbs::curveSaveOBJ("output_curve.obj", crv);
    return 0;

}