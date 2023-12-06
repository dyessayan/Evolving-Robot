#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <random>
#include <thread>

#define GLM_FORCE_RADIANS
//Make sure to include glew before gl.h and glfw3.h
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>


/*
TODO: Improve timesteps/second - Either:
    RK4 integration - probably not too hard ... or 
    parallelization - Probably a little bit harder, may or may not speed up more than RK4 integration would

    Omega L_0 <- resting length _0= a+bin(omega*t + c)
    Parameters: a, b, c for resting length, k spring constant
*/
//Global Constants
const double G  = -9.81; // -9.81m/s^2 * (1 unit / 0.1 m)
const double dt =  0.0005;
double t        =  0;
double ground   = -1;
const double DF =  0.999; //Damping applied to velocity (changed from 0.999)
const uint8_t PARAMS_PER_SPRING  = 4;
const uint8_t SPRINGS_PER_CUBE   = 28;
const uint16_t PARAMS_PER_CUBE   = (PARAMS_PER_SPRING * SPRINGS_PER_CUBE)*2 + 3; // set of parameters for each spring in the cube, + 3 for its position
const uint16_t EVOLVABLE_PARAMS_PER_CUBE = PARAMS_PER_CUBE - 3;
const uint8_t CUBE_LOCATION_PARAMS       = 3;
const double PROBABILITY_ADD            = 0.01;
const double PROBABILITY_RMV            = 0.01;
const int MAX_NUM_CUBES                 = 12;
const int MIN_NUM_CUBES                 = 2;
std::vector<glm::vec3> starting_vertices;

//Simulation Parameters
const double STARTING_TRANSLATION_X = 0.0;
const double STARTING_TRANSLATION_Y = 0.0;
const double STARTING_TRANSLATION_Z = 0.5; //height the cube is dropped from

const double STARTING_ROT_X         = 0.0; //Starting counter-clockwise rotation about the x-axis in degrees
const double STARTING_ROT_Y         = 0.0;
const double STARTING_ROT_Z         = 0.0;

const float CAMERA_START_X = -5.0;
const float CAMERA_START_Y = -5.0;
const float CAMERA_START_Z =  4.0;

const int GLFW_WINDOW_HEIGHT = 800;
const int GLFW_WINDOW_WIDTH  = 600;



//Debug flags
const bool DEBUG                    = false;
const bool DEBUG_ADD                = false;
const bool DEBUG_RMV                = false;
const bool DEBUG_RESTORATION        = false;
const bool DEBUG_TIME               = false;
const bool DEBUG_VERTICES           = false;
const bool DEBUG_SPRINGS            = false;
const bool DEBUG_INIT               = false;
const bool LOG_ENERGY               = false;
const bool LOG_SPRING_PERFORMANCE   = false;

//Global variables
glm::vec3 cameraPos = glm::vec3(CAMERA_START_X, CAMERA_START_Y, CAMERA_START_Z);
glm::vec3 centerOfView = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraFront = glm::normalize(centerOfView - cameraPos);
glm::vec3 worldUp      = glm::vec3(0.0f, 0.0f, 1.0f);

glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, worldUp));
glm::vec3 cameraUp    = glm::normalize(glm::cross(cameraRight, cameraFront));
glm::mat4 view = glm::lookAt(
    cameraPos,                      // Camera location in world space
    centerOfView,                   // Center of view
    worldUp                         // 0efines the direction of "up"
);

float cameraSpeed   = 5*dt;
float prevX         = GLFW_WINDOW_WIDTH/(float)2.0;
float prevY         = GLFW_WINDOW_HEIGHT/(float)2.0;
float yaw           = -90.0f;
float pitch         = 0.0f;

bool firstMouse = true;

std::default_random_engine standard_normal_engine;
double snrv() {
    std::normal_distribution<double> z{0, 1};
    return z(standard_normal_engine);
}

class RNG {
    public:
    std::vector<double> mu;
    std::vector<double> sigma;
    int num_params;
    std::default_random_engine generator;
    
    RNG() : generator{} {

    };
    RNG(double m, double s, int n) : mu{m}, sigma{s}, num_params{n}, generator{} {
            
    };

    double snrv() {
        std::normal_distribution<double> z{0, 1};
        return z(generator);
    }
    std::vector<double> sample(std::vector<double> sigma) {
        std::vector<double> p(sigma.size(), 0.0);
        for(int i = 0; i < sigma.size(); i++) {
            std::normal_distribution<double> d {0, sigma.at(i)};
            p.at(i) = (d(generator));
        }
        return p;
    }


    void set_params(std::vector<double> &m, std::vector<double> &s) {
        mu = m;
        sigma = s;
    }
};

RNG globalRNG = RNG();

std::string toString(std::vector<double> v, int per_line = -1, std::string sep = ", ", bool brackets = true) {
    std::ostringstream oss;
    if (brackets) oss << "{";
    for(int i = 0; i < v.size(); i++) {
        oss << std::to_string(v.at(i));
        if (v.size() > 0 && i < v.size() - 1) oss << sep;
        if (per_line > 0 && (i + 1) % per_line == 0) oss << "\n";
    }
    if (brackets) oss << "}";
    return oss.str();
}
std::string toString(std::vector<int> v, int per_line = -1, std::string sep = ", ", bool brackets = true) {
    std::ostringstream oss;
    if (brackets) oss << "{";
    for(int i = 0; i < v.size(); i++) {
        oss << std::to_string(v.at(i));
        if (v.size() > 0 && i < v.size() - 1) oss << sep;
        if (per_line > 0 && i % per_line == 0) oss << "\n";
    }
    if (brackets) oss << "}";
    return oss.str();
}
std::string toString(std::vector<bool> v, int per_line = -1, std::string sep = ", ", bool brackets = true) {
    std::ostringstream oss;
    if (brackets) oss << "{";
    for(int i = 0; i < v.size(); i++) {
        if (v.at(i)) oss << " True";
        else oss << "False";
        //oss << std::to_string(v.at(i));
        if (v.size() > 0 && i < v.size() - 1) oss << sep;
        if (per_line > 0 && i % per_line == 0) oss << "\n";
    }
    if (brackets) oss << "}";
    return oss.str();
}
std::string toString(std::vector<std::vector<double>> v, int per_line = -1, std::string sep = ", ", bool brackets = true) {
    std::ostringstream oss;
    for(int i = 0; i < v.size(); i++) {
        oss << toString(v.at(i), per_line, sep, brackets);
    }
    return oss.str();
}
std::string toString(std::vector<std::vector<int>> v, int per_line = -1, std::string sep = ", ", bool brackets = true) {
    std::ostringstream oss;
    for(int i = 0; i < v.size(); i++) {
        oss << toString(v.at(i), per_line, sep, brackets);
    }
    return oss.str();
}

bool nearlyeq(std::vector<double> a, std::vector<double> b, double epsilon, bool verbose = false) {
    bool eq = true;
    if(a.size() != b.size()) {
        if (verbose) std::cout << "Cannot compare vectors of unequal sizes ( "<< a.size() << ", " << b.size() << ")\n";
        return false;
    } 
    for(int i = 0; i < a.size(); i++) {
        if(fabs(a.at(i)-b.at(i)) > epsilon) {
            if (verbose) std::cout << "At index " << i << " a is " << a.at(i) << ", b is " << b.at(i) << " difference is " << fabs(a.at(i)-b.at(i)) << std::endl;
            eq = false;
        }
    }
    return eq;
}
void pprint(glm::dvec3 p) { 
    std::cout << "(" << p.x << " " << p.y << " " << p.z << ")" << std::endl;
}
void pprint(glm::vec4 p) { 
    std::cout << "(" << p.x << " " << p.y << " " << p.z << " " << p.w << ")" << std::endl;
}
// Probably write a function to turn logs (dubug couts) off/on with a debug switch later, just hack it for now

class Mass {
    
    public:
    double mass;
    int id[3];
    glm::dvec3 position;
    glm::dvec3 velocity;
    glm::dvec3 acceleration;
    glm::dvec3 external_forces;

    Mass() : mass(0), position({0,0,0}), velocity({0,0,0}), acceleration({0,0,0}), external_forces({0,0,0}) {};
    Mass(double m) : mass(m), position({0,0,0}), velocity({0,0,0}), acceleration({0,0,0}), external_forces({0,0,0}) {};

    Mass(double m, glm::dvec3 p, glm::dvec3 v, glm::dvec3 a, glm::dvec3 e) : mass(m), position(p), velocity(v), acceleration(a), external_forces(e) {};

    static std::shared_ptr<Mass> create() {
        return std::make_shared<Mass>();
    }
    static std::shared_ptr<Mass> create(double m) {
        return std::make_shared<Mass>(m);
    }

    double get_mass(std::string sys) {
        if (sys == "SI") return mass;
        else if (sys == "Screen") return mass*10;
        return -999999999999.99;
    }
    void add_external_force(glm::dvec3 f) {external_forces += f;}
    void add_gravitational_force() {
        if (DEBUG) std::cout << "Gravitational Force Magnitude: " << std::to_string(glm::length(get_mass("SI")*G*glm::dvec3{0, 0, 1})) << std::endl; 
        external_forces += get_mass("SI")*G*glm::dvec3{0, 0, 1};
    }
    void add_restoration_force() { 
        if(position.z < -1) {
            double frc = (0.1/0.0005)*G; //-10000.0f; (was 0.0005 in case of emergency)
            //glm::dvec3 direction = glm::normalize(centroid-position);
            
            if(DEBUG) {
                std::cout << "Position: (" << position.x << ", " << position.y << ", " << position.z << ")\n";
                std::cout << "Restoration Force Applied: " << std::to_string(-glm::length(frc*glm::dvec3(0,0,(position.z-ground)*0.1))) << std::endl;
            }
            external_forces += frc*glm::dvec3(0,0,(position.z-ground)*0.1); //Convert position to meters and also added 1-
        }
    }
    glm::dvec3 add_restoration_force(glm::dvec3 &centroid, bool tagged = false) { 
        double frc = (mass/0.000000375)*G; //-10000.0f; (was 0.0005 in case of emergency)
        glm::dvec3 direction = glm::dvec3(0,0,1);//glm::normalize(centroid-position);
        glm::dvec3 tangential_component = velocity - glm::dot(velocity, direction)*direction;
        if(position.z < ground) {
            if(DEBUG || DEBUG_RESTORATION) {
                std::cout << "Position: (" << position.x << ", " << position.y << ", " << position.z << ")\n"<< "ground: " << ground << std::endl;;
                std::cout << "Gravity restoration: ";
                pprint(frc*(position.z-ground)*0.1*direction);
                
                 if (glm::length2(velocity * glm::normalize(glm::dvec3(1.0, 1.0, 0.0))) > 1e-6){
                     //std::cout << "Friction (kinetic): ";
                    pprint(mass*-0.3*(tangential_component));
                 }
                 else {
                    //std::cout << "Friction (static): ";
                    pprint( (mass)*-0.4*tangential_component);
                 }

                //std::cout << "Restoration Force Applied: " << " ";
               // pprint(frc*(position.z-ground)*0.1*direction);
            }
            external_forces += frc*(position.z-ground)*mass*direction; //glm::dvec3(0,0,1); //Convert position to meters and also added 1-
            
            if (glm::length2(tangential_component) > 1e-6) external_forces += mass*-0.3*tangential_component;
            else  external_forces += (mass)*-0.4*tangential_component;

            //
            if (((frc*(position.z-ground)*0.1) > 0) && DEBUG_RESTORATION) return frc*(position.z-ground)*0.1*direction;
        }
        return glm::dvec3(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

    }
    double potential_energy() {
        double gravitational_potential = mass*-G*(position.z - ground);
        return gravitational_potential;
    }
    glm::dvec3 update(glm::dvec3 &centroid, bool tagged=false) {
        if (DEBUG) std::cout << "Adding Gravitational Force!\n";
        add_gravitational_force();
        if (DEBUG) pprint(external_forces);
        if (DEBUG) std::cout << "Adding Restoration Force!\n";
        glm::dvec3 rforce = add_restoration_force(centroid, tagged);
        if (DEBUG) pprint(external_forces);
        if (DEBUG) std::cout << "Mass: " << mass << std::endl;
        acceleration = external_forces/(mass*10); //10 conversion factor to screen units, acceleration now in screen units per second squared
        if (DEBUG) std::cout << "Acceleration!: " << std::endl;
        if (DEBUG) pprint(acceleration);
        velocity    += acceleration*dt;
        velocity    *= DF;
        if (DEBUG) std::cout << "Velocity: " << std::endl;
        if (DEBUG) pprint(velocity);
        position    += velocity*dt;
        if (DEBUG) std::cout << "Position: " << std::endl;
        if (DEBUG) pprint(position);
        external_forces = glm::dvec3(0,0,0);
        return rforce;
    }
};
class Spring {
    public:
    int id;
    std::shared_ptr<Mass> mass1;
    std::shared_ptr<Mass> mass2;
    glm::vec3 centroid;
    bool is_dynamic;            //whether to adjust the length based on timetep or not
    double a;                   //intercept
    double b;                   //amplitude
    double c;                   //period
    double d;                   //phase
    double a0;
    double b0;
    double c0;
    double d0;
    double rest_length;      //m   = 10
    double rest_length0;
    double spring_constant;  //N/m = N/(10 units)


    Spring() : mass1(), mass2(), rest_length(0), spring_constant(0) {};
    Spring(std::shared_ptr<Mass> m1, std::shared_ptr<Mass> m2, double rl, double sc) : mass1(m1), mass2(m2), rest_length(rl), spring_constant(sc) {};
    Spring(double rl, double sc) : rest_length(rl), spring_constant(sc) {};
        Spring(std::shared_ptr<Mass> m1, std::shared_ptr<Mass> m2, double rl, double sc, double a, double b, double c, double d) : mass1(m1),
         mass2(m2), rest_length(rl), spring_constant(sc), a(a), b(b), c(c), d(d), a0(a), b0(b), c0(c), d0(d), rest_length0(a+b*sin(2*3.14159*t/c + d)) {};

    static std::shared_ptr<Spring> create() {
        return std::make_shared<Spring>();
    }
    static std::shared_ptr<Spring> create(double rl, double sc) {
        return std::make_shared<Spring>(rl, sc);
    }
    static std::shared_ptr<Spring> create(std::shared_ptr<Mass> m1, std::shared_ptr<Mass> m2, double rl, double sc) {
        if (DEBUG) std::cout << "Creating Spring! " << std::endl;
        double rl_real = glm::length(m1->position - m2->position)/10;
        if (DEBUG) std::cout << "Created with Resting Length: " << rl_real << std::endl;
        return std::make_shared<Spring>(m1, m2, rl_real, 1*sc, rl_real, 0.25*rl_real, 1, 0);
    }

    void change_defaults(double a, double b, double c, double d) {
        a0 = a;
        b0 = b;
        c0 = c;
        d0 = d;
        rest_length0 = a + b*sin(2*3.14159*t/c + d);
    }

    void set_parameters_default() {
        a = a0;
        b = b0;
        c = c0;
        d = d0;
        rest_length = a + b*sin(2*3.14159*t/c + d);
    }

    void set_paramaters(double a, double b, double c, double d) {
        a = a;
        b = b;
        c = c;
        d = d;
    }
    void mutate_paramaters(double pa, double pb, double pc, double pd) {
        /* Pass the Perturbations which will be added to the spring's 
           default parameters.
        */
        a += pa;
        b += pb;
        c += pc;
        d += pd;

        if (a < 0) a = -a;
        if (b < 0) b = -b;
        if (c < 1e-6) c = 1e-6;
        d = std::min(std::max(d, -3.1416), 3.1416);
    }
    std::vector<double> get_parameters() {
        std::vector<double> r;
        r.push_back(a);
        r.push_back(b);
        r.push_back(c);
        r.push_back(d);
        return r;
    }
    double _length() { 
        return glm::length(mass1->position - mass2->position)/10;} //Spring Length in meters

    void set_resting_length_by_point_and_mass(int mass_index, glm::dvec3 &point) {
        glm::dvec3 pos;
        double orig_length = rest_length;
        if(mass_index == 0) pos = mass1->position;// rest_length =  glm::length(mass1->position - point)/10;
        if(mass_index == 1) pos = mass2->position; //rest_length =  glm::length(mass2->position - point)/10;
        double length_at_projection = glm::length(pos - point)/10;
        rest_length =  length_at_projection;// + (length_at_projection - orig_length)//glm::length(pos - point)/10;
    }

    void set_resting_length(double new_length) {
        //can use rest_length = original_length + 1*sin(2*pi*t + 0) then mutate a + b*sin(2*pi*t/c + d)
        rest_length = new_length;
    }

    void update_mass_forces (int type = 0, bool special_debug = false) {
        if(type == 0) {
            if (DEBUG || special_debug) std::cout << "Spring Length: " << _length() << " m" << std::endl;
            if (DEBUG || special_debug) std::cout << "Rest Length: " << rest_length << " m" << std::endl;
            double magnitude = -spring_constant*(_length() - rest_length);
            std::string marker;
            if (std::fabs(magnitude) > 1e-7) marker = " ************ ";
            else marker = "";
            if (DEBUG_SPRINGS || special_debug) {
                std::ios::fmtflags oldflags = std::cout.flags();
                int oldprecision = std::cout.precision();
                std::cout << std::fixed <<std::setprecision(36) <<std::right << "\nSpring Force Magnitude from\nMass 1 (" << 
                    mass1->id[0] << ", " << mass1->id[1] << ", " <<
                    mass1->id[2] << ") originally; \n(" << mass1->position.x << ", " << mass1->position.y << ", " << mass1->position.z << ") now to\nMass 2 (" <<
                    mass2->id[0] << ", " << mass2->id[1] << ", " <<
                    mass2->id[2] << ") originally; \n(" << mass2->position.x << ", " << mass2->position.y << ", " << mass2->position.z <<") now\n" << 
                    marker << magnitude << marker;
                std::cout.flags(oldflags);
                std::cout.precision(oldprecision);
            }
            //Unit vector pointing from mass 1 to mass 2
            glm::dvec3 direction = (mass1->position-mass2->position)/(_length() * 10);
            glm::dvec3 force = magnitude * direction;
            if (DEBUG) std::cout << "Spring Force! Between vertex at (m1, m2 positions below respectively) \n";
            if (DEBUG) pprint(mass1->position);
            if (DEBUG) pprint(mass2->position);
            if (DEBUG) std::cout << "Their difference (m1-m2) followed by direction of force, and force applied to m1: \n";
            if (DEBUG) pprint(mass1->position-mass2->position); // The direction of this is toward mass1
            if (DEBUG) pprint(direction);                       // Spring constant is also negative
            if (DEBUG) pprint(force);
            mass1->add_external_force(force);
            mass2->add_external_force(-force);
        }
        else if (type == 1) {
            //for connections in the grid
            glm::dvec3 pos1 = mass1->position;
            glm::dvec3 pos2 = mass2->position;

            glm::dvec3 correction = 0.5*(pos1 - pos2);
            mass1->position -= correction;
            mass2->position += correction;

            mass1->velocity -= correction/dt;
            mass2->velocity += correction/dt;
        }
        if(is_dynamic) rest_length = a + b*sin(2*3.14159*t/c + d);
    }
    
};
struct cube_init_params {
    double STARTING_TRANSLATION_X {::STARTING_TRANSLATION_X};
    double STARTING_TRANSLATION_Y {::STARTING_TRANSLATION_Y};
    double STARTING_TRANSLATION_Z {::STARTING_TRANSLATION_Z};
    
    double STARTING_ROT_X {::STARTING_ROT_X};
    double STARTING_ROT_Y {::STARTING_ROT_Y};
    double STARTING_ROT_Z {::STARTING_ROT_Z};

    double VERTEX_MASS {0.1};

    //cube_init_params() {};

};
struct BB {
    glm::dvec3 mins;
    glm::dvec3 maxes;
};
std::array<double, 24> unitcube_vertices {
            -1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
            -1.0,  1.0, -1.0,
            -1.0,  1.0,  1.0,
             1.0, -1.0, -1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0, -1.0,
             1.0,  1.0,  1.0,
        };
class Cube {
    public:
    glm::dvec3 center {0.0, 0.0, 0.0};
    BB bb;
    double side_length;
    double mass;
    bool tagged;

    std::vector<double> parameters {};
    std::vector<double> p0 {};
    std::vector<double> cube_perturbations {};
    std::unordered_map<int, std::array<std::array<int, 3>, 2>> face_triangles;
    std::unordered_map<int, int> face_indices;
    std::array<std::shared_ptr<Mass>, 8> masses;
    std::array<std::shared_ptr<Spring>, 28> springs;
    glm::vec3 _centroid;
    glm::vec3 initial_centroid;
    std::array<glm::dvec3, 12> normals;
    glm::mat4  model;
    cube_init_params initial_locations;
    std::array<double, 24> v0 {
        -0.5, -0.5, -0.5,
            -0.5, -0.5, +0.5,
            -0.5, +0.5, -0.5,
            -0.5, +0.5, +0.5,
            +0.5, -0.5, -0.5,
            +0.5, -0.5, +0.5,
            +0.5, +0.5, -0.5,
            +0.5, +0.5, +0.5
    };
    std::array<double, 24> vertices {
            center.x-0.5, center.y-0.5, center.z-0.5,
            center.x-0.5, center.y-0.5, center.z+0.5,
            center.x-0.5, center.y+0.5, center.z-0.5,
            center.x-0.5, center.y+0.5, center.z+0.5,
            center.x+0.5, center.y-0.5, center.z-0.5,
            center.x+0.5, center.y-0.5, center.z+0.5,
            center.x+0.5, center.y+0.5, center.z-0.5,
            center.x+0.5, center.y+0.5, center.z+0.5
        };
        

    Cube(std::vector<double> params) : center(glm::dvec3{0.0, 0.0, 0.0}), masses(std::array<std::shared_ptr<Mass>, 8>  {Mass::create(),Mass::create(),Mass::create(),Mass::create(),Mass::create(),
    Mass::create(),Mass::create(),Mass::create()}), model(glm::mat4(1.0f)), normals{}, parameters{params} {
        std::array<double, 24> v { get_vertices() };
        if (DEBUG) std::cout << "created massses!" << std::endl;

        //translate(2, DROP_HEIGHT, vertices);
        //pprint_vertices_matrix();
        for(int i = 0; i < masses.size(); i++) {
            if (DEBUG) std::cout << "\nConstructor set Masses\n";
            masses.at(i)->position = glm::dvec3(v[3*i], v[3*i+1], v[3*i+2]);
            if (DEBUG ) std::cout<< std::endl << i << ": (" << masses[i]->position.x<< ", " << masses[i]->position.y <<", "<< masses[i]->position.z << ")" << std::endl;
        }

        for(int i = 0; i < masses.size()-1; i++) {
            for(int j = i+1; j < masses.size(); j++) {
                int i_offset = (15*i-i*i)/2;
                
                if (DEBUG ) std::cout << "creating spring " << i_offset + j - i - 1 << std::endl;
                if (DEBUG ) std::cout << "i: " <<i << ", j: " << j << std::endl;
                springs[i_offset + j-i - 1] = Spring::create(masses[i], masses[j],  0.1, 0.1/0.000099*G*-1 );
            }
        }
    }; 
    Cube() : center(glm::dvec3{0.0, 0.0, 0.0}), masses(std::array<std::shared_ptr<Mass>, 8>  {Mass::create(),Mass::create(),Mass::create(),Mass::create(),Mass::create(),
    Mass::create(),Mass::create(),Mass::create()}), model(glm::mat4(1.0f)), normals{}, tagged{false} {
        std::array<double, 24> v { get_vertices() };
        v0 = v;
        if (DEBUG) std::cout << "created massses!" << std::endl;

        //translate(2, DROP_HEIGHT, vertices);
        //pprint_vertices_matrix();
        for(int i = 0; i < masses.size(); i++) {
            if (DEBUG) std::cout << "\nConstructor set Masses\n";
            masses.at(i)->position = glm::dvec3(v[3*i], v[3*i+1], v[3*i+2]);
            if (DEBUG ) std::cout<< std::endl << i << ": (" << masses[i]->position.x<< ", " << masses[i]->position.y <<", "<< masses[i]->position.z << ")" << std::endl;
        }

        for(int i = 0; i < masses.size()-1; i++) {
            for(int j = i+1; j < masses.size(); j++) {
                int i_offset = (15*i-i*i)/2;
                
                if (DEBUG ) std::cout << "creating spring " << i_offset + j - i - 1 << std::endl;
                if (DEBUG ) std::cout << "i: " <<i << ", j: " << j << std::endl;
                springs[i_offset + j-i - 1] = Spring::create(masses[i], masses[j],  0.1, 0.1/0.000099*G*-1 );
            }
        }
        parameters.resize(SPRINGS_PER_CUBE * PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS);
        cube_perturbations.resize(SPRINGS_PER_CUBE * PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS);
        set_parameters_default();
        p0 = parameters;
    }; 

    Cube(double umass) : center(glm::dvec3{0.0, 0.0, 0.0}), masses(std::array<std::shared_ptr<Mass>, 8>  {Mass::create(umass),Mass::create(umass),Mass::create(umass),Mass::create(umass),Mass::create(umass),
        Mass::create(umass),Mass::create(umass),Mass::create(umass)}), model(glm::dmat4(1.0f)) {
        std::array<double, 24> v { get_vertices() };
        for(int i = 0; i < masses.size(); i++) {
            masses.at(i)->position = glm::dvec3(v[3*i], v[3*i+1], v[3*i+2]);
        }
    };
    void restore_default_parameters(bool reinit = true) {
        /*  Reset Everything to Factory Defaults
            Set acceleratio, velocity and force of all masses to zero
            set position to the position of a unit cube centered at the origin
            set the springss parameters to their defaults
            reinitialize cube's location with the default configuration
        */
        vertices = v0;
        for(int i = 0; i < masses.size(); i++) {
            masses.at(i)->acceleration = glm::dvec3{0, 0, 0};
            masses.at(i)->velocity = glm::dvec3{0, 0, 0};
            masses.at(i)->external_forces = glm::dvec3{0, 0, 0};
            masses.at(i)->position = glm::dvec3(v0.at(3*i), v0.at(3*i+1), v0.at(3*i+2));
        }
        for(int i = 0; i < springs.size(); i++) {
            springs.at(i)->set_parameters_default();
        }
        _init_cube_location(initial_locations);
        set_parameters_default(reinit);
    }
    void set_grid_location(double r, double c, double e) {
        /* 
            pass row, column, elevation in that order as doubles
        */
        parameters.at(0)         = r;
        cube_perturbations.at(0) = r;
        parameters.at(1)         = c;
        cube_perturbations.at(1) = c;
        parameters.at(2)         = e;
        cube_perturbations.at(2) = e;
    }
    void print_config() {
        std::cout << "Centroid: ";
        pprint(dcentroid());
        std::cout << "\nMasses: \n";
        for(int i = 0; i < masses.size(); i++) {
            std::cout << "Mass " << i << std::endl;
            std::cout << "\tmass " << masses.at(i)->mass << " kg";
            std::cout << "\n\tExternal Forces: ";
            pprint(masses.at(i)->external_forces);
            std::cout << "\n\tAcceleration: ";
            pprint(masses.at(i)->acceleration);
            std::cout << "\n\tVelocity: ";
            pprint(masses.at(i)->velocity);
            std::cout << "\n\tPosition: ";
            pprint(masses.at(i)->position);
        }

        std::cout << "\n\nSprings: \n";
        for(int i = 0; i < springs.size(); i++) {
            std::cout << "Spring " << i << std::endl;
            std::cout << "\tmasses (" << springs.at(i)->mass1->id[0] << ", " << springs.at(i)->mass1->id[1]
                        << ", " << springs.at(i)->mass1->id[2] << ") and " << springs.at(i)->mass2->id[0] << ", " << springs.at(i)->mass2->id[1]
                        << ", " << springs.at(i)->mass2->id[2] << ")";
            std::cout << "\n\tResting Length: " << springs.at(i)->rest_length;
            std::cout << "\n\tParameters\n\t\ta: " << springs.at(i)->a <<
                    "\n\tParameters\n\t\tb: " << springs.at(i)->b << "\n\tParameters\n\t\tc: " << springs.at(i)->c <<
                    "\n\tParameters\n\t\td: " << springs.at(i)->d;
        }
        pprint_vertices_matrix();
    }
    void set_parameters_default(bool reinit = true) {
        /* 
            Set the Cube instance's parameters to be equal to its springs parameters, which initialize in a cube shape
        */
       if(reinit) {
            parameters.at(0) = -1;
            parameters.at(1) = -1;
            parameters.at(2) = -1;
       }
        for(int i = 0; i < springs.size(); i++) {
            //std::cout << i*PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS << std::endl;
            std::vector<double> spring_params = springs.at(i)->get_parameters();
            parameters.at(i*PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS)     = spring_params.at(0);
            //std::cout << i*PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS + 1 << std::endl;
            parameters.at(i*PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS + 1) = spring_params.at(1);
            //std::cout << i*PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS + 2 << std::endl;
            parameters.at(i*PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS + 2) = spring_params.at(2);
            //std::cout << i*PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS + 3 << std::endl;
            parameters.at(i*PARAMS_PER_SPRING + CUBE_LOCATION_PARAMS + 3) = spring_params.at(3);
            //std::cout << "Spring params: " << spring_params.at(0) << " " << spring_params.at(1)<<  " " << spring_params.at(2) << "\n";
            //std::cout << "Cube params: " << parameters.at(4*i + 0) << " " << parameters.at(4*i + 1)<<  " " <<parameters.at(4*i +2) << "\n";
        }
        //std::cout << "Done (set params default)\n";
    // for(double i : parameters) {
    //     std::cout << i << " ";
    // }
    // std::cout << "\n";
    }

    void set_parameters(std::pair<std::vector<double>::iterator, std::vector<double>::iterator> &params) {
        /*  Pass two iterators to a vector<double> and directly use these to set 
            **parameters** in this Cube instance
        */
        int i = 0;
        //std::cout << "parameters size: " << params.second - params.first << std::endl;
        for(std::vector<double>::iterator it = params.first; it != params.second; it++) {
            parameters.at(i++) = *(it);
        }
    }
    std::vector<double> get_default_parameters() {
        return p0;
    }
    std::vector<double> get_parameters() {
        std::vector<double> accumulated_perturbations;
        accumulated_perturbations.push_back(parameters.at(0));
        accumulated_perturbations.push_back(parameters.at(1));
        accumulated_perturbations.push_back(parameters.at(2));
        //std::cout << "Sizes: (parameters, p0, cube_perturbations): " << parameters.size() << ", " << p0.size() << ", " << cube_perturbations.size() << std::endl;
        for(int i = 0; i < cube_perturbations.size() - CUBE_LOCATION_PARAMS; i++) {
            accumulated_perturbations.push_back(parameters.at(CUBE_LOCATION_PARAMS + i) - p0.at(CUBE_LOCATION_PARAMS + i));
        }
        // std::cout << "getting parameters \n";
        // for(int i = 0; i < parameters.size(); i++) {
        //     std::cout << parameters.at(i) << " ";
        // }
        return accumulated_perturbations;
    }
    void make_dynamic() {
        for(auto &spring : springs) {
            spring->is_dynamic = true;
        }
    }
    void mutate(std::array<double, 28> &perturbations) {
        int spring_parameters = 4;
        for(int i = 0; i < perturbations.size(); i++) {
            double a = perturbations.at(i/spring_parameters);
            double b = perturbations.at(i/spring_parameters + 1);
            double c = perturbations.at(i/spring_parameters + 2);
            double d = perturbations.at(i/spring_parameters + 3);
            springs.at(i/spring_parameters)->mutate_paramaters(a, b, c, d);
        }
    }
    void mutate(std::pair<std::vector<double>::iterator, std::vector<double>::iterator> &perturbations) {
        /*
            Pass a pair of iterators to a vector of doubles, the first and last perturbation
        */
        int spring_parameters = 4;
        int i = 0;
        //std::cout << perturbations.second - perturbations.first << std::endl;
        for(std::vector<double>::iterator it = perturbations.first; it != perturbations.second; it++) {
            double a = *(it++);//perturbations.at(i/spring_parameters);
            double b = *(it++);//perturbations.at(i/spring_parameters + 1);
            double c = *(it++);//perturbations.at(i/spring_parameters + 2);
            double d = *(it);//perturbations.at(i/spring_parameters + 3);
            if (parameters.at(4*i + CUBE_LOCATION_PARAMS + 3) + d > 3.1416 || parameters.at(4*i + CUBE_LOCATION_PARAMS + 3) + d < -3.1416) d = 0;
            //std::cout << "i: " << 4*i << ", " << 4*i+1 << ", " << 4*i+2 << ", " << 4*i+3 << std::endl;

            //std::cout << "a, b, c, d: " << a << ", " << b << ", " << c << ", " << d << std::endl;
            springs.at(i)->mutate_paramaters(a, b, c, d);
    
            parameters.at(4*i + CUBE_LOCATION_PARAMS)     += a;
            parameters.at(4*i + CUBE_LOCATION_PARAMS + 1) += b;
            parameters.at(4*i + CUBE_LOCATION_PARAMS + 2) += c;
            parameters.at(4*i + CUBE_LOCATION_PARAMS + 3) += d;
            cube_perturbations.at(4*i + CUBE_LOCATION_PARAMS)     = a;
            cube_perturbations.at(4*i + CUBE_LOCATION_PARAMS + 1) = b;
            cube_perturbations.at(4*i + CUBE_LOCATION_PARAMS + 2) = c;
            cube_perturbations.at(4*i + CUBE_LOCATION_PARAMS + 3) = d;
            if (parameters.at(4*i + CUBE_LOCATION_PARAMS) < 0) parameters.at(4*i + CUBE_LOCATION_PARAMS) *= -1;
            if (parameters.at(4*i + CUBE_LOCATION_PARAMS + 1) < 0) parameters.at(4*i + CUBE_LOCATION_PARAMS + 1) *= -1;
            if (parameters.at(4*i + CUBE_LOCATION_PARAMS + 2) < 1e-6) parameters.at(4*i + CUBE_LOCATION_PARAMS + 2) = 1e-6;
            double tol = 1e-6;
            if (tagged) {
                pprint_vertices_matrix();
                double err = 0.0;
                double rate = 0.2;
                int count = 0;
                do {
                    err = 0.0;
                    for(auto & spring : springs) {
                        std::cout << "length: " << spring->_length() << ", rest_length: " << spring->rest_length << std::endl;
                        err += std::fabs(spring->_length() - spring->rest_length);
                    }
                    std::cout << "Error: " << err << std::endl;
                    if (err < tol) std::cout << "Finished in " << count << " iterations\n";
                    for(auto &spring : springs) {
                        glm::dvec3 dir = spring->mass1->position - spring->mass2->position;
                        double spring_error = glm::length(dir)-spring->rest_length;
                        std::cout << "mass 1 position: ";
                        pprint(spring->mass1->position);
                        std::cout << "\nmass 2 position: ";
                        pprint(spring->mass2->position);
                        std::cout << "\nrate, spring_error, dir: " << rate << ", " << spring_error << ", ";
                        pprint(dir);
                        std::cout << "\n";
                        spring->mass1->position -= rate*spring_error*dir;
                        spring->mass2->position += rate*spring_error*dir;
                        std::cout << "mass 1 position: ";
                        pprint(spring->mass1->position);
                        std::cout << "\nmass 2 position: ";
                        pprint(spring->mass2->position);
                    }
                    std::cout << "err end << " << count << ": " << err << std::endl;
                } while (err > tol);
            i++;
            }
        }
    }
    std::array<GLuint, 36> get_faces()
        { 
            return std::array<GLuint, 36>{
            // return indices of the triangles making up the cube
            // oriented in a counter-clockwise direction when looking at the cube face
            // from outside the cube
            2, 0, 1,                            //These are all negative in x coordinate
            3, 2, 1,                            // 0, 1, 2 and 3

            4, 6, 7,                            // Positive x coordinate
            5, 4, 7,                            // 

            0, 4, 5,                            //
            1, 0, 5,

            6, 2, 3,                            //
            7, 6, 3,

            2, 6, 4,                            //
            0, 2, 4,

            1, 5, 7,                            //
            3, 1, 7
        };
    }
    void computeBB() {
        bb.mins  = masses.at(0)->position;
        bb.maxes = masses.at(0)->position;
        for(int i = 1; i < masses.size(); i++) {
            glm::dvec3 p = masses.at(i)->position;
            if (p.x < bb.mins.x)  bb.mins.x  = p.x;
            if (p.x > bb.maxes.x) bb.maxes.x = p.x;
            if (p.y < bb.mins.y)  bb.mins.y  = p.y;
            if (p.y > bb.maxes.y) bb.maxes.y = p.y;
            if (p.z < bb.mins.z)  bb.mins.z  = p.z;
            if (p.z > bb.maxes.z) bb.maxes.z = p.z;
        }
    }
    void create_face_triangles() {
        face_triangles.emplace(2*3*5*7, std::array<std::array<int, 3>, 2> {std::array<int, 3> {0, 1, 2}, std::array<int, 3> {2, 3, 1}});
        face_triangles.emplace(11*13*17*19, std::array<std::array<int, 3>, 2> {std::array<int, 3> {4, 5, 6}, std::array<int, 3> {6, 7, 5}});
        face_triangles.emplace(2*3*11*13, std::array<std::array<int, 3>, 2> {std::array<int, 3> {0, 1, 4}, std::array<int, 3> {4, 5, 1}});
        face_triangles.emplace(5*7*17*19, std::array<std::array<int, 3>, 2> {std::array<int, 3> {2, 3, 6}, std::array<int, 3> {6, 7, 3}});
        face_triangles.emplace(2*5*11*17, std::array<std::array<int, 3>, 2> {std::array<int, 3> {0, 2, 4}, std::array<int, 3> {4, 6, 2}});
        face_triangles.emplace(3*7*13*19, std::array<std::array<int, 3>, 2> {std::array<int, 3> {1, 3, 5}, std::array<int, 3> {5, 7, 3}});

        face_indices.emplace(2*3*5*7, 0);
        face_indices.emplace(11*13*17*19, 2);
        face_indices.emplace(2*3*11*13, 4);
        face_indices.emplace(5*7*17*19, 6);
        face_indices.emplace(2*5*11*17, 8);
        face_indices.emplace(2*3*5*7, 10);
    }
    std::array<std::array<int, 3>, 2> get_face_triangles(std::array<int, 4> &vs) {
        int key = 1;
        int ps[] {2, 3, 5, 7, 11, 13, 17, 19};
        for(int i = 0; i < 6; i++) {
            key *= ps[i];
        }
        return face_triangles[key];
    }
    std::array<std::array<int, 3>, 2> get_face_triangles(int vs) {
        int key = vs;
        return face_triangles[key];
    }
    int get_face_index(int vs) {
        return face_indices[vs];
    }
    int get_hash(std::array<int, 4> &vs) {
        int key = 1;
        int ps[] {2, 3, 5, 7, 11, 13, 17, 19};
        for(int i = 0; i < 4; i++) {
            key *= ps[vs[i]];
        }
        std::cout << key << std::endl;
        return key;
    }
    glm::dvec3 &get_normal_i(int i) {
        return normals[i];
    }
    void compute_normals() {
        std::array<GLuint, 36> faces = get_faces();
        for(int i = 0; i < faces.size(); i+=3) {
            glm::dvec3 vertex1 = masses.at(faces[i])->position;
            glm::dvec3 vertex2 = masses.at(faces[i+1])->position;
            glm::dvec3 vertex3 = masses.at(faces[i+2])->position;
            glm::dvec3 edge1 = vertex1-vertex2;
            glm::dvec3 edge2 = vertex3-vertex2;
            glm::dvec3 n     = glm::normalize(glm::cross(edge1, edge2));
            normals[i/3]       = n;
        }
    }

    glm::vec3 centroid() {
        glm::vec3 centroid {0,0,0};
        for(int i = 0; i < masses.size(); i++) {
            centroid += masses.at(i)->position;
        }
        return (centroid / (float)masses.size());
    }
    glm::dvec3 dcentroid() {
        glm::dvec3 centroid {0,0,0};
        for(int i = 0; i < masses.size(); i++) {
            centroid += masses.at(i)->position;
        }
        return (centroid / (double)masses.size());
    }
    double kinetic_energy() {
        glm::vec3 deltap = centroid() - _centroid;
        _centroid = centroid();
        double v = glm::length(deltap);
        return 0.5*mass*v*v;
    }
    double potential_energy() {
        return mass*-G*(centroid().z - ground);
    }

    void log_energy(std::string logfile) {
        std::ofstream f(logfile, std::ios_base::app);
        if(!f.is_open()) {
            std::cout << "Error: Could not open logfile: " << logfile << std::endl;
        }
        f << t << ", " << potential_energy() << ", " << kinetic_energy() << std::endl;
        f.close();
    }

    void set_masses(double m) {
        for(auto &_mass : masses) {
            _mass->mass = m;
        }
    }
    void set_masses(std::array<double, 8> &m) {
        for(int i = 0; i < masses.size(); i++) {
            masses[i]->mass = m[i];
        }
    }

    void translate(int axis, double amt) {
        std::string axes[] {"+x", "+y", "+z"};
        if(DEBUG) std::cout << "translating " << std::to_string(amt) << " units in the " << axes[axis] << " direction\n";
        for(int i = 0; i < vertices.size(); i++) {
            if ((i % 3) == axis) {
                vertices[i] += amt;
                if (axis == 0) {
                masses[(int)i/3]->position.x = vertices[i];
                }
                if (axis == 1) {
                    masses[(int)i/3]->position.y = vertices[i];
                }
                if (axis == 2) {
                    masses[(int)i/3]->position.z = vertices[i];
                }
                if (DEBUG) std::cout<< i << ": (" << masses[i/3]->position.x << 
                ", " << masses[i/3]->position.y << 
                ", " << masses[i/3]->position.z << ")" << std::endl;

            }            
        }
    }

    void rotate(int axis, double degrees) {
        double theta_radians = glm::radians(degrees);
        double sin_theta = std::sin(glm::radians(degrees));
        double cos_theta = std::cos(glm::radians(degrees));
        if(DEBUG || DEBUG_INIT) std::cout << "rotating " << degrees << "degrees (" << theta_radians << " radians) about axis " << axis << std::endl;
        for(int i = 0; i < vertices.size()/3; i++) {
            if(DEBUG) std::cout << "i: " << i << std::endl;
            std::vector<double> v;
            for(int j = 0; j < 3; j++) {
                if(DEBUG) std::cout << "j: " << j << std::endl;
                if (j != axis) {
                    //axis of rotation is fixed, push pack the other two coordinates
                    if(DEBUG) std::cout << "pushing back (3i+j): " << 3*i + j << std::endl;
                    v.push_back(vertices[3*i+j]);
                }
            }
            int index = 0; //we need to apply a different function to the first and second element in this vector
            // vertices[3*i+j] = v.at(0)*cos_theta - v.at(1)*cos_theta (first time)
            // vertices[3*i+j] = v.at(0)*sin_theta + v.at(1)*sin_theta (second time)
            for(int j = 0; j < 3; j++) {
                if ((j != axis)) {
                    double coordinate1 = v.at(0);
                    double coordinate2 = v.at(1);
                    if(DEBUG || DEBUG_INIT) {
                        std::cout << "Old Coordinates: " << coordinate1 << ", " << coordinate2 << std::endl;
                        std::cout << "New Coordinates: " << coordinate1*cos_theta - coordinate2*sin_theta << ", " << 
                            coordinate1*sin_theta + coordinate2*cos_theta << std::endl;
                    }
                    if (index == 0) vertices[3*i + j] = coordinate1*cos_theta - coordinate2*sin_theta;
                    if (index == 1) vertices[3*i + j] = coordinate1*sin_theta + coordinate2*cos_theta;
                    index++;
                }
            }
            masses[i]->position.x = vertices[3*i];
            masses[i]->position.y = vertices[3*i + 1];
            masses[i]->position.z = vertices[3*i + 2];
            if (DEBUG || DEBUG_INIT) std::cout<< i << ": (" << masses[i]->position.x << 
                ", " << masses[i]->position.y << 
                ", " << masses[i]->position.z << ")" << std::endl;
        }
    }
    void _init_cube_location(cube_init_params &params) {

       // std::cout << "Initial Vertex Locations" << std::endl;
        initial_locations = params;
        rotate(0, params.STARTING_ROT_X);
        rotate(1, params.STARTING_ROT_Y);
        rotate(2, params.STARTING_ROT_Z);


        translate(0, params.STARTING_TRANSLATION_X);
        translate(1, params.STARTING_TRANSLATION_Y);
        translate(2, params.STARTING_TRANSLATION_Z);

        set_masses(params.VERTEX_MASS);
        _centroid = centroid();
        for(int i = 0; i < masses.size(); i++) {
            mass += masses.at(i)->mass;
            //For dubugging to track the denote the index of the vertex with the same orientation in the unit cube
            masses[i]->id[0] = unitcube_vertices[3*i];
            masses[i]->id[1] = unitcube_vertices[3*i+1];
            masses[i]->id[2] = unitcube_vertices[3*i+2];
        }
        computeBB();
        compute_normals();
        create_face_triangles();
        initial_centroid = dcentroid();
    }
    
    Cube(glm::dvec3 c) : center(c), model(glm::translate(glm::dmat4(1.0f), c)) {};
    //I think we can find the centroid by taking an average of locations of vertices, or something similar with some thought...
    //Cube(std::array<Mass *, 8> m) : center()

    std::array<double, 24> &get_vertices() {
        return vertices;
    };

    void _set_vertex(int i, const glm::dvec3 &val) {
        vertices[i]   = val.x;
        vertices[i+1] = val.y;
        vertices[i+2] = val.z;
    }
    void update_vertices() {
        for(int i = 0; i < masses.size(); i++) {
            _set_vertex(3*i, masses[i]->position);
        }
    }
    void pprint_mass_positions() {
        std::cout << "\n\n********************Mass Positions**************************\n";
        std::ios::fmtflags old_settings = std::cout.flags();
        int oldprecision = std::cout.precision();
        for(int i = 0; i < masses.size(); i++) {
            std::cout << std::fixed << std::setprecision(10)
            << std::right << std::setw(10) <<masses.at(i)->position.x  << std::right << std::setw(10) << masses.at(i)->position.y <<
                 std::right << std::setw(10) << masses.at(i)->position.z << std::endl;
        }
        std::cout << "*******************************************************************\n";
        std::cout.flags(old_settings);
        std::cout.precision(oldprecision);
    }
    void pprint_vertices(std::string fmt = "def", int width = 0, int precision = 6) {
        std::cout << "\n\n********************VERTICES**************************\n";
        std::ios::fmtflags old_settings = std::cout.flags();
        int oldprecision = std::cout.precision();
        for(int i = 0; i < vertices.size()-2; i+=3) {
            std::cout << "Vertex " << (int)i/3 << " originally " << "(" << (int)unitcube_vertices[i] <<", " <<
                 (int)unitcube_vertices[i+1] << ", " << (int)unitcube_vertices[i+2] << ")" << std::endl;
            std::cout << std::fixed << std::setprecision(precision)  <<
                 "\nx: " << std::setw(width) <<vertices[i] << "\ny: " << std::setw(width) << vertices[i+1] <<
                 "\nz: " << std::setw(width) << vertices[i + 2] << std::endl << std::endl;
        }
        std::cout << "*******************************************************************\n";
        std::cout.flags(old_settings);
        std::cout.precision(oldprecision);
    }
    void pprint_vertices_matrix(int width = 0, int precision = 6) {
         std::cout << "\n\n********************VERTICES**************************\n";
        std::ios::fmtflags old_settings = std::cout.flags();
        int oldprecision = std::cout.precision();
        for(int i = 0; i < vertices.size()-2; i+=3) {
            std::cout << std::fixed << std::setprecision(precision)
            << std::right << std::setw(10) <<vertices[i]  << std::right << std::setw(10) << vertices[i+1] <<
                 std::right << std::setw(10) << vertices[i + 2] << std::endl;
        }
        std::cout << "*******************************************************************\n";
        std::cout.flags(old_settings);
        std::cout.precision(oldprecision);
    }

    void update() {
        glm::dvec3 cent = dcentroid();
        if (DEBUG_SPRINGS) std::cout << std::endl << "======================================= Springs: Timestep " << (int)(t/dt) <<
         "==================================================" << std::endl << std::endl;
        for(int i = 0; i < springs.size(); i++) {
            //std::cout << "Updating Spring " << i << std::endl;
            springs[i]->update_mass_forces();
        }
        std::vector<glm::dvec3> rforces;
        for(int i = 0; i < 8; i++) {
            // if(tagged) {
            //     std::cout << "Mass " << i << " position: ";
            //     pprint(masses.at(i)->position);
            //     std::cout << "\nExternal Forces: ";
            //     pprint(masses.at(i)->external_forces);
            //     std::cout << "\nVelocity: ";
            //     pprint(masses.at(i)->velocity);
            // }
            glm::dvec3 rforce = masses[i]->update(cent);
            if (!((std::isnan(rforce.x)) || (std::isnan(rforce.y)) || (std::isnan(rforce.z)))) {
                rforces.push_back(rforce);
                std::ios::fmtflags old_settings = std::cout.flags();
                int oldprecision = std::cout.precision();
                std::cout << std::fixed << std::setprecision(36) << "Vector " << i << " Originally (" << (int)unitcube_vertices[3*i] << 
                     ", " << (int)unitcube_vertices[3*i+1] << ", " << (int)unitcube_vertices[3*i+2] << "): \nx: " << std::setw(48) << std::right <<
                     rforce.x <<  "\ny: "<< std::setw(48) << rforce.y  << "\nz: " <<  std::setw(48)<<  rforce.z << std::endl;
                std::cout.flags(old_settings);
                std::cout.precision(oldprecision);
            }
        }
        if(rforces.size() > 0 && (DEBUG_RESTORATION || DEBUG)) {
            int timestep = t/dt;
            glm::dvec3 net_rforce;
            net_rforce = std::accumulate(rforces.begin(), rforces.end(), glm::dvec3 {0.0, 0.0, 0.0} );
            std::ios::fmtflags old_settings = std::cout.flags();
            int oldprecision = std::cout.precision();
            std::cout << "Net Restoration Force Acting on Cube: " << std::endl;
            //pprint(net_rforce);
            std::cout << std::fixed << std::setprecision(36) << "x: " << std::setw(48) << std::right <<
                     net_rforce.x <<  "\ny: "<< std::setw(48) << net_rforce.y  << "\nz: " <<  std::setw(48)<<  net_rforce.z << std::endl;
            std::cout.flags(old_settings);
            std::cout.precision(oldprecision);
            std::cout << std::endl << "*******Timestep " << timestep << "Above! *********" << std::endl;
        }
        update_vertices();
        if (DEBUG) pprint_vertices();
        computeBB();
        compute_normals();
        //model = glm::translate(model, dt*masses[0]->velocity);
    }
    void update(double &cumulative_evaluation_times, std::string spring_eval_filename) {
        double eval_this_timestep = 0;
        for(int i = 0; i < 27; i++) {
            auto start = std::chrono::steady_clock::now();
            springs[i]->update_mass_forces();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            double time_in_ms = elapsed.count();
            eval_this_timestep += time_in_ms;
        }
        cumulative_evaluation_times += eval_this_timestep;
        log_evaluation_times(cumulative_evaluation_times,eval_this_timestep, spring_eval_filename);
        glm::dvec3 cent = dcentroid();
        for(int i = 0; i < 8; i++) {
            masses[i]->update(cent);
        }
        update_vertices();
        computeBB();
        compute_normals();
        if (DEBUG) pprint_vertices();
        //model = glm::translate(model, dt*masses[0]->velocity);
    }
    void log_evaluation_times(double &cumulative_evaluation_time, double instantaneous_time, std::string spring_eval_filename) {
        std::ofstream spring_log(spring_eval_filename, std::ios_base::app);
        if (!spring_log.is_open()) {
            std::cout << "Could Not Open Logfile for Spring Evaluation at " << spring_eval_filename << std::endl;
        }
        double timestep;
        double num_springs = 24;
        timestep = ((t/dt) + 1.0);
        double avg_evals_per_second = (num_springs * timestep)/cumulative_evaluation_time;
        double instantaneous_evals_per_second = num_springs/instantaneous_time;
        spring_log << std::to_string(t) << ", " << std::to_string(instantaneous_evals_per_second) << ", " << avg_evals_per_second << std::endl;
        spring_log.close();
    }
    

    double get_mass() {
        double m = 0;
        for(int i = 0; i < 8; i++) {
            m +=masses.at(i)->mass;
        }
        return m;
    }

    std::array<glm::dvec3, 12> &get_normals() {
        return normals;
    }
    double get_volume() {
        //std::array<double, 24> v {get_vertices()};
        return 0.1*0.1*0.1;
    }

    void set_mass(uint16_t i, double mass_val) {
        masses.at(i)->mass = mass_val;
    }
};



class Grid {

    public:
    int height;
    int width;
    int length;
    int ground_level;
    int center_row;
    int center_column;
    glm::dvec3 centroid;
    std::vector<std::vector<std::vector<std::shared_ptr<Cube>>>> elements;
    std::vector<std::shared_ptr<Cube>> elements_list;
    std::vector<std::shared_ptr<Spring>> connections;
    std::vector<std::vector<int>> open_spots;
    std::vector<double> parameters;
    
    Grid(int h, int l, int w) : Grid(h, l, w, std::move(std::vector<double> {})) {
        // std::vector<double> v;
        // Grid(h, l, w, v);
        //open_spots.push_back(std::vector<int> {height/2, length/2, width/2});
    }

    Grid(int h, int l, int w, std::vector<double> &&p) : height{std::max(h, 1)}, length{std::max(l, 1)}, width{std::max(w, 1)},
                                                     parameters{p}{
        //std::cout << "start of grid constructor\n";
        elements.resize(height);
        for(auto &slice : elements) {
            slice.resize(length);
            for(auto &row : slice) {
                row.resize(width);
            }
        }
        std::vector<int> initial_opening {length/2, width/2, height/2};

        make_open(initial_opening);
        //std::cout << "Constructor Parameter Size: " << parameters.size() << std::endl;
        //std::cout << "Size Open Spots: " << open_spots.size() << std::endl;
        //std::cout << "end of grid constructor\n";
    }

    void make_open(std::vector<int> loc) {
        /*std::cout << "Opening: Row " << std::to_string(loc.at(0)) << ", Column " << std::to_string(loc.at(1)) << ", Elevation "
                         << std::to_string(loc.at(2)) << std::endl;*/
        open_spots.push_back(loc);
        //std::cout << "Open Spots Now: " << toString(open_spots) << std::endl;
    }

    void make_dynamic() {
        for(auto &cube : elements_list) {
            cube-> make_dynamic();
        }
    }
    void tag_cube(int i = -1) {
        if (i == -1) i = rand() % elements_list.size();
        elements_list.at(i)->tagged = true;
    }

    void restore_defaults(bool reinit = true) {
        for(int i = 0; i < elements_list.size(); i++) {
            //std::cout << "Restore " << i << std::endl;
            elements_list.at(i)->restore_default_parameters(reinit);
        }
        //std::cout << "Done Restoring defaults\n";
        //set_parameters_default(reinit);
    }

    void set_parameters_default(bool reinit = true) {
        /* Sets cube's parameters equal to its springs. Use restore_defaults to restore springs to default parameters*/
        for(int i = 0; i < elements_list.size(); i++) {
            elements_list.at(i)->set_parameters_default(reinit);
        }
    }

    void set_parameters(std::vector<double> p) {
        elements_list.clear();
        elements.clear();
        open_spots.clear();
        connections.clear();
        parameters.clear();
       
        if (DEBUG) std::cout << "Setting parameters Manually  for " << p.size() / PARAMS_PER_CUBE << " cubes\n";
        int num_cubes = p.size() / PARAMS_PER_CUBE;
        int new_h = 0; int new_l = 0; int new_w = 0;
        int new_ground = 1e9; int new_center_row = 1e9; int new_center_column = 1e9;
        for(int i = 0; i < PARAMS_PER_CUBE*num_cubes; i += PARAMS_PER_CUBE) {
            // cube parameters store location is row, column, elevation - corresponds to length, width, height
            if ((int)p.at(i+0) + 2 > new_l) new_l = (int)p.at(i+0) + 2;
            if ((int)p.at(i+0) < new_center_row) new_center_row = (int)p.at(i+0);
            if ((int)p.at(i+1) + 2 > new_w)      new_w = (int)p.at(i+1) + 2;
            if ((int)p.at(i+1) < new_center_column) new_center_column = (int)p.at(i+1);
            if ((int)p.at(i+2) + 2 > new_h) new_h = (int)p.at(i+2) + 2;
            if ((int)p.at(i+2) < new_ground) new_ground = (int)p.at(i+2);
        }
        height = std::max(new_h, height);
        length = std::max(new_l, length);
        width  = std::max(new_w, width);
        center_row = new_center_row;
        center_column=new_center_column;
        ground_level = new_ground;
        elements.resize(height);
        for(int i = 0; i < height; i++) {
            elements.at(i).resize(length);
        }
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < length; j++) {
                elements.at(i).at(j).resize(width);
            }
        }
        //std::cout << "num_cubes: " << num_cubes << std::endl;
        for(int i = 0; i < num_cubes; i++) {
            std::vector<double>::const_iterator first = p.begin() + i*PARAMS_PER_CUBE;
            std::vector<double>::const_iterator last  = p.begin() + (i+1)*PARAMS_PER_CUBE;
            std::vector<double> pi {first, last};
            std::shared_ptr<Cube> to_add {new Cube()};
            //elements_list.push_back(to_add);
            int elevation = (int)pi.at(2);
            int row       = (int)pi.at(0);
            int column    = (int)pi.at(1);
            cube_init_params cip;
            
            // std::cout << "Elevation: " << elevation << ", Ground Level: " << ground_level << 
            //         ", Translation: " << 1.0*(elevation-ground_level)+0.5 << std::endl;
            // std::cout << "Row: " << row << ", Center_row: " << center_row << 
            //         ", Translation: " << 1.0*(row-center_row)+0.5 << std::endl;
            // std::cout << "Column: " << column << ", Center_column: " << center_column << 
            //         ", Translation: " << 1.0*(column-center_column)+0.5 << std::endl;
            cip.STARTING_TRANSLATION_X = 1.0*(column-center_column)+0.5; 
            cip.STARTING_TRANSLATION_Y = 1.0*(row-center_row)+0.5;
            cip.STARTING_TRANSLATION_Z = 1.0*(elevation-ground_level+ground)+0.5; // + 0.5 because default cube is initialized as unit cube centered
                                                                        // at the origin i.e, 0.5 units below ground for the bottom masses
            to_add->_init_cube_location(cip);
            add(to_add, row, column, elevation, true); // set initial to true here so we do not change parameters
        }
        //adjust_grid_locations(0, 0, 0);
        tag_cube(-1);
        parameters = p;
        std::cout << "Size of Parameters after Setting: " << parameters.size() << std::endl;
        restore_defaults(false);
        generate_perturbations(parameters);
    }
    std::vector<double> get_stds(int index) {
        std::vector<double> stds;
        int start  = index * PARAMS_PER_CUBE + CUBE_LOCATION_PARAMS + PARAMS_PER_SPRING * SPRINGS_PER_CUBE;
        int finish = start + PARAMS_PER_SPRING * SPRINGS_PER_CUBE;
        stds.insert(stds.begin(), parameters.begin() + start, parameters.begin() + finish);
        return stds;
    }
    std::vector<double> get_cube_params(int index) {
        std::vector<double> cube_params;
        int start  = index * PARAMS_PER_CUBE;
        int finish = start + PARAMS_PER_SPRING * SPRINGS_PER_CUBE + CUBE_LOCATION_PARAMS;
        cube_params.insert(cube_params.begin(), parameters.begin() + start, parameters.begin() + finish);
        return cube_params;
    }
    std::vector<double> get_grid_locations(int index) {
        std::vector<double> result;
        int start  = index * PARAMS_PER_CUBE;
        int finish = start + CUBE_LOCATION_PARAMS;
        result.insert(result.begin(), parameters.begin() + start, parameters.begin() + finish);
        return result;
    }
    
    void set_grid_locations(int index, double r, double c, double e) {
        int start  = index * PARAMS_PER_CUBE;
        int finish = start + CUBE_LOCATION_PARAMS;
        parameters.at(start    ) = r;
        parameters.at(start + 1) = c;
        parameters.at(start + 2) = e;
    }
    void set_grid_locations(std::shared_ptr<Cube> cube, double r, double c, double e) {
        auto it    = std::find(elements_list.begin(), elements_list.end(), cube);
        int index  = -1;
        if (it != elements_list.end()) index = it - elements_list.begin();
        int start  = index * PARAMS_PER_CUBE;
        int finish = start + CUBE_LOCATION_PARAMS;
        if (index > 0) {
            parameters.at(start    ) = r;
            parameters.at(start + 1) = c;
            parameters.at(start + 2) = e;
        }
        else {
            std::cout << "Element Not Found (set_grid_location)!\n";
        }
    }
    std::vector<double> get_grid_locations(std::shared_ptr<Cube> cube) {
        /* Returns <row, column, elevation> as doubles*/
        auto it    = std::find(elements_list.begin(), elements_list.end(), cube);
        int index  = -1;
        if (it != elements_list.end()) index = it - elements_list.begin();
        std::vector<double> loc = get_grid_locations(index);
        return loc;
    }
    std::vector<int> get_grid_locations(int index, bool as_int) {
        /* Returns <row, column, elevation> as ints*/
        std::vector<double> result {get_grid_locations(index)};
        std::vector<int> return_vals(3, 0);
        return_vals.at(0) = (int)result.at(0);
        return_vals.at(1) = (int)result.at(1);
        return_vals.at(2) = (int)result.at(2);
        return return_vals;
    }
    void set_stds(int index, std::vector<double> &new_stds) {
        std::vector<double>::iterator start  = parameters.begin() + index * PARAMS_PER_CUBE + CUBE_LOCATION_PARAMS + 
            PARAMS_PER_SPRING * SPRINGS_PER_CUBE;
        std::vector<double>::iterator finish = start + PARAMS_PER_SPRING * SPRINGS_PER_CUBE;
        std::copy(new_stds.begin(), new_stds.end(), start);
    }
    void get_all_new_stds() {
        int num_params = SPRINGS_PER_CUBE * PARAMS_PER_SPRING;
        double tau  = (1/(double)std::sqrt(2*num_params));
        double _snrv = snrv();
        double tauZero = (1/(double)std::sqrt(2*std::sqrt(2*num_params)));
        auto transform_stds = [tau, tauZero, _snrv](double old_std_i) {return std::exp(tau*_snrv) * old_std_i * std::exp(tauZero * snrv());};// * std::exp(tauZero*rng.snrv()); };
        std::vector<std::vector<double>> stdevs(elements_list.size(), std::vector<double>(num_params, 0.1));
        for(int i = 0; i < elements_list.size(); i++) {
            std::vector<double> stds_i = get_stds(i);
            std::transform(stds_i.begin(), stds_i.end(), stdevs.at(i).begin(), transform_stds);
            set_stds(i, stdevs.at(i));
        }
    }
    std::vector<double> mutate_stds(std::vector<double> stds) {
        int num_params = SPRINGS_PER_CUBE * PARAMS_PER_SPRING;
        double tau  = (1/(double)std::sqrt(2*num_params));
        double _snrv = snrv();
        double tauZero = (1/(double)std::sqrt(2*std::sqrt(2*num_params)));
        auto transform_stds = [tau, tauZero, _snrv](double old_std_i) {return std::exp(tau*_snrv) * old_std_i * std::exp(tauZero * snrv());};// * std::exp(tauZero*rng.snrv()); };
        std::transform(stds.begin(), stds.end(), stds.begin(), transform_stds);
        return stds;
    }
    std::vector<double> generate_perturbations() {
        /* Generates and returns a vector of perturbations to this robots parameters */
        std::vector<double> perturbations;
        if((rand() / (double)RAND_MAX) < PROBABILITY_ADD && elements_list.size() < MAX_NUM_CUBES) {
            std::cout << "Adding Randomly\n";
            add(false, true);
        }
        if((rand() / (double)RAND_MAX) < PROBABILITY_RMV && elements_list.size() > MIN_NUM_CUBES) {
            std::cout << "Removing Randomly\n";
            remove();
        }
        for(int i = 0; i < elements_list.size(); i++) {
            std::vector<double> cube_params_i = get_cube_params(i);
            std::vector<double> stds_i = get_stds(i);
            stds_i = mutate_stds(stds_i);
            std::vector<double> perturbation_factors = globalRNG.sample(stds_i);
            std::transform(cube_params_i.begin() + CUBE_LOCATION_PARAMS, cube_params_i.end(), perturbation_factors.begin(), 
                cube_params_i.begin() + CUBE_LOCATION_PARAMS, [](double param, double perturb) {return param + perturb;});
            perturbations.insert(perturbations.end(), cube_params_i.begin(), cube_params_i.end());
            perturbations.insert(perturbations.end(), stds_i.begin(), stds_i.end());
        }
        return perturbations;
    }
    void generate_perturbations(std::vector<double> &p) {
        /* Changes the cubes parameters according to tbe perturbation vector passed in, this should probably have a different name */
        int i = 0; 
        int num_params = SPRINGS_PER_CUBE * PARAMS_PER_SPRING;//EVOLVABLE_PARAMS_PER_CUBE;
        //for(int i = 0; i < p.size(); i++) {std::cout << p.at(i) << " ";}
        //std::cout << "\nSize: " << p.size() << std::endl;
        parameters = p;
        for( auto &cube : elements_list) {
            // Skip the first 3 parameters for each cube, since these will now hold location, which is not evolvable, but allows
            // building the grid of cubes from passing a vector of parent parameters.
            std::pair<std::vector<double>::iterator, std::vector<double>::iterator> pi {p.begin() + (i+1)*CUBE_LOCATION_PARAMS + 
                2*i*num_params, p.begin() + (i+1)*CUBE_LOCATION_PARAMS + (2*i+1) * num_params};
            
            // auto get_std[](param) {return (param/(doube)8 + 0.02);};
            // std::transform(p.begin() + (i+1)*CUBE_LOCATION_PARAMS + (2*i+1) * num_params, p.begin() + (i+1)*CUBE_LOCATION_PARAMS + 
            //     2*(i+1)*num_params, );
            //cube->set_parameters(pi);
            //std::cout << "Mutate\n";
            //std::cout << "Perturbations for Params: " << std::to_string((i+1) * CUBE_LOCATION_PARAMS + 2*i*num_params) << " to " 
            //    << std::to_string( (i+1)*CUBE_LOCATION_PARAMS + (2*i+1) * num_params) << std::endl;
            cube->set_parameters_default(false);
            cube->mutate(pi);
            //std::cout << "Done mutating\n";
            i++;
        }
    }
    // void adjust_grid_locations() {
    //     /* Sometimes we need to grow the grid, then we need to adjust the indices so each cube appears in the proper location*/
    //     std::cout << "adjust grid locations without parameters\n";
    //     int ground_zero  = ground_level;
    //     ground_level = height;
    //     int dr = 0, dc = 0, de = 0;
    //     for(int i = 0; i < elements.size(); i++) {
    //         for(int j = 0; j < elements.at(i).size(); j++) {
    //             for(int k = 0; k < elements.at(i).at(j).size(); k++) {
    //                 if (elements.at(i).at(j).at(k) != nullptr)  {
    //                     std::vector<double> old_loc = get_grid_locations(elements.at(i).at(j).at(k));
    //                     int old_row, old_col, old_ele;
    //                     dr = j - old_row;
    //                     dc = k - old_col;
    //                     de = i - old_ele;
    //                     set_grid_locations(elements.at(i).at(j).at(k), j, k, i);
    //                     elements.at(i).at(j).at(k)->set_grid_location(j, k, i);
    //                     if (i < ground_level) ground_level = i;
    //                 }
    //             }
    //         }
    //     }
        
    //     if (ground_zero != ground_level) {
    //         for(int i = 0; i < elements_list.size(); i++) {
    //             cube_init_params cip = elements_list.at(i)->initial_locations;
    //             std::vector<double> translations = get_grid_locations(i);
    //             int row       = (int)translations.at(0);
    //             int column    = (int)translations.at(1);
    //             int elevation = (int)translations.at(2);
    //             cip.STARTING_TRANSLATION_X = 1.0*column;
    //             cip.STARTING_TRANSLATION_Y = 1.0*row;
    //             cip.STARTING_TRANSLATION_Z = 1.0*(elevation-ground_level)+0.5;
    //             elements_list.at(i)->_init_cube_location(cip);
    //         }
    //     }
    // }
    void log_open_spots() {
        for(int i = 0; i < open_spots.size(); i++) {
                std::cout << toString(open_spots.at(i));
        }
        std::cout << std::endl;
    }
    void log_grid_locs() {
        for(int i = 0; i < elements_list.size(); i++) {
            std::cout << toString(get_grid_locations(i));
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<int>> get_all_grid_locs(bool as_int) {
        std::vector<std::vector<int>> grid_locs;
        for(int i = 0; i < elements_list.size(); i++) {
            grid_locs.push_back(std::vector<int> {get_grid_locations(i, as_int)});
        }
        return grid_locs;
    }
    void adjust_grid_locations(int dr, int dc, int de) {
        /* Sometimes we need to grow the grid, then we need to adjust the indices so each cube appears in the proper location*/
        if (DEBUG) {
            std::cout << "Adjusting Grid Sizes and Indices\nelevation: " << de << "\nrow: " << dr << "\ncolumn: " << dc << 
                    "\ninitial open spots: ";
            log_open_spots();
            std::cout << "\ninitial grid locations: ";
            log_grid_locs();
        }
        ground_level = height;
        center_row   = length;
        center_column= width;

        for(int i = 0; i < elements_list.size(); i++) {
            std::vector<int> current_loc {get_grid_locations(i, true)}; // these hold r, c, e
            //std::cout << toString(current_loc);
            current_loc.at(0) += dr;
            current_loc.at(1) += dc;
            current_loc.at(2) += de;
            if (current_loc.at(2) < ground_level) ground_level = current_loc.at(2);
            set_grid_locations(i, current_loc.at(0), current_loc.at(1), current_loc.at(2));
        }
        reindex_open_spots(dr, dc, de);
        for(int i = 0; i < elements_list.size(); i++) {
            cube_init_params cip = elements_list.at(i)->initial_locations;
            std::vector<double> translations = get_grid_locations(i);
            int row       = (int)translations.at(0);
            int column    = (int)translations.at(1);
            int elevation = (int)translations.at(2);
            cip.STARTING_TRANSLATION_X = 1.0*column;
            cip.STARTING_TRANSLATION_Y = 1.0*row;
            cip.STARTING_TRANSLATION_Z = 1.0*(elevation-ground_level)+0.5;
            elements_list.at(i)->_init_cube_location(cip);
        }
        if (DEBUG || DEBUG_ADD) {
            std::cout << "After Adjusting Grid Sizes and Indices\nelevation: " << de << "\nrow: " << dr << "\ncolumn: " << dc << 
                    "\nfinal open spots: ";
            log_open_spots();
            std::cout << "\nfinal grid locations: ";
            log_grid_locs();
        }
    }
    void adjust_grid_locations(int dr, int dc, int de, std::vector<std::vector<int>> subset) {
        /* Sometimes we need to grow the grid, then we need to adjust the indices so each cube appears in the proper location*/
        
        if (DEBUG || DEBUG_ADD || DEBUG_RMV) {
            std::cout << "Subset: " << toString(subset) << std::endl;
            std::cout << "Adjusting Grid Sizes and Indices\nelevation: " << de << "\nrow: " << dr << "\ncolumn: " << dc << 
                    "\ninitial open spots: ";
            log_open_spots();
            std::cout << "\ninitial grid locations: ";
            log_grid_locs();
        }
        ground_level = height;
        std::vector<int> indices;
        for(int i = 0; i < subset.size(); i++) {
            indices.push_back(grid_to_list_index(subset.at(i)));
        }
        for(int i = 0; i < subset.size(); i++) {
            if (DEBUG || DEBUG_RMV) {
            std::cout << "Grid Locations before adjustment " << i << ": " << toString(get_all_grid_locs(true)) << std::endl; 
            }
            std::vector<int> current_loc {subset.at(i)};
            int idx = indices.at(i);
            if (DEBUG_RMV || DEBUG) std::cout << "Current Adjustment: " << toString(current_loc) << "\n";
            //std::cout << toString(current_loc);
            current_loc.at(0) += dr;
            current_loc.at(1) += dc;
            current_loc.at(2) += de;
            if (current_loc.at(2) < ground_level) ground_level = current_loc.at(2);
            set_grid_locations(indices.at(i), current_loc.at(0), current_loc.at(1), current_loc.at(2));
            if(DEBUG || DEBUG_RMV) {
                std::cout << "current_loc: " << toString(current_loc) << std::endl;
                std::cout << "grid locations index " << ": " << toString(get_grid_locations(i)) << std::endl;
                std::cout << "grid locations index (asint)" << ": " << toString(get_grid_locations(i, true)) << std::endl;
            }
        }
        reindex_open_spots(dr, dc, de);
        for(int i = 0; i < subset.size(); i++) {
            int index = grid_to_list_index(subset.at(i));
            std::shared_ptr<Cube> cube_to_move = elements_list.at(index);
            cube_init_params cip = cube_to_move->initial_locations;
            std::vector<int> translations = subset.at(i);
            int row       = translations.at(0);
            int column    = translations.at(1);
            int elevation = translations.at(2);
            cip.STARTING_TRANSLATION_X = 1.0*(column - center_column) + 0.5;
            cip.STARTING_TRANSLATION_Y = 1.0*(row - center_row) + 0.5;
            cip.STARTING_TRANSLATION_Z = 1.0*(elevation-ground_level + ground)+0.5;
            cube_to_move->_init_cube_location(cip);
        }
        if (DEBUG || DEBUG_ADD || DEBUG_RMV) {
            std::cout << "Subset:" << toString(subset) << std::endl;
            std::cout << "After Adjusting Grid Sizes and Indices\nelevation: " << de << "\nrow: " << dr << "\ncolumn: " << dc << 
                    "\nfinal open spots: ";
            log_open_spots();
            std::cout << "\nfinal grid locations: ";
            log_grid_locs();
        }
    }
    void reindex_open_spots(int dr, int dc, int de) {
        for(int i = 0; i < open_spots.size(); i++) {
            open_spots.at(i).at(0) += dr;
            open_spots.at(i).at(1) += dc;
            open_spots.at(i).at(2) += de;
        }
    }
    void recompute_open_spots() {
        open_spots.clear();
        for(int i = 0; i < elements_list.size(); i++) {
            std::vector<int> grid_location = get_grid_locations(i, true);
            int row = grid_location.at(0);
            int col = grid_location.at(1);
            int ele = grid_location.at(2);
            if ((elements.at(ele+1).at(row).at(col) == nullptr) && std::find(open_spots.begin(), open_spots.end(),
                std::vector<int> {row, col, ele+1}) == open_spots.end()) open_spots.push_back(std::vector<int> {row, col, ele+1});
            if ((elements.at(ele-1).at(row).at(col) == nullptr) && std::find(open_spots.begin(), open_spots.end(),
                std::vector<int> {row, col, ele-1}) == open_spots.end()) open_spots.push_back(std::vector<int> {row, col, ele-1});
            if ((elements.at(ele).at(row+1).at(col) == nullptr) && std::find(open_spots.begin(), open_spots.end(),
                std::vector<int> {row+1, col, ele}) == open_spots.end()) open_spots.push_back(std::vector<int> {row+1, col, ele});
            if ((elements.at(ele).at(row-1).at(col) == nullptr) && std::find(open_spots.begin(), open_spots.end(),
                std::vector<int> {row-1, col, ele}) == open_spots.end()) open_spots.push_back(std::vector<int> {row-1, col, ele});
            if ((elements.at(ele).at(row).at(col+1) == nullptr) && std::find(open_spots.begin(), open_spots.end(),
                std::vector<int> {row, col+1, ele}) == open_spots.end()) open_spots.push_back(std::vector<int> {row, col+1, ele});
            if ((elements.at(ele).at(row).at(col-1) == nullptr) && std::find(open_spots.begin(), open_spots.end(),
                std::vector<int> {row, col-1, ele}) == open_spots.end()) open_spots.push_back(std::vector<int> {row, col-1, ele});
        }
    } 

    std::vector<double> get_parameters() {
        //std::vector<double> return_vec;
        // std::vector<double> my_params;
        // int num_cubes = elements_list.size();
        // int params_in_cube = SPRINGS_PER_CUBE * PARAMS_PER_SPRING;
        // return_vec.resize(PARAMS_PER_CUBE * num_cubes);
        // for(int i = 0; i < num_cubes; i++) {
        //     std::vector<double> cube_params = elements_list.at(i)->get_parameters();
        //     for(int j = 0; j < cube_params.size(); j++) {
        //         //std::cout << "cube params: " << cube_params.at(j) << std::endl;
        //         if(cube_params.at(j) != parameters.at(PARAMS_PER_CUBE*i + j)) {
        //             std::cout << "cube " << i << " parameter " << j << " (" << cube_params.at(j) << ") != parameters " << 
        //             PARAMS_PER_CUBE*i + j << "(" << parameters.at(PARAMS_PER_CUBE*i + j) << ")" << std::endl;
        //         }
        //         return_vec.at(PARAMS_PER_CUBE*i + j) = cube_params.at(j);
        //     }
        // }
        // if (DEBUG) std::cout << "parameters.size(): " << parameters.size() << "\nnum_cubes * PARAMS_PER_CUBE: " << PARAMS_PER_CUBE * num_cubes <<
        //              "\nnum_cubes: " << elements_list.size() << ", " << num_cubes << std::endl;
        // //std::cout << "cube params = grid param: " << (return_vec == parameters) << std::endl;
        // if(!((nearlyeq(return_vec, parameters, 1e-7)))) std::cout << "\nWarning: Cube Params != Grid Params (returning cube params)\n\n";
        return parameters;
    }
    std::vector<int> random_location() {
        int num_spots  = open_spots.size();
        if (DEBUG) std::cout << "Size Open_Spots in random_location: " << open_spots.size() << std::endl;
        int rand_index;
        if (num_spots == 0) throw std::runtime_error("No open spots on the cube grid?");
        else rand_index = rand() % num_spots;
        std::vector<int> spot {open_spots.at(rand_index)};
        open_spots.erase(open_spots.begin() + rand_index);
        /* Open Spots so this is (r, c, e)*/
        return spot;
    }
    int grid_to_list_index(int r, int c, int e) {
        auto search_it = std::find(elements_list.begin(), elements_list.end(), elements.at(e).at(r).at(c));
        int search_index = -1;
        if (search_it != elements_list.end()) search_index = search_it - elements_list.begin();
        return search_index;
    }
    int grid_to_list_index(std::vector<int> l) {
        int r = l.at(0);
        int c = l.at(1);
        int e = l.at(2);
        return grid_to_list_index(r, c, e);
    }
    void remove() {
        int indx = rand() % elements_list.size();
        std::vector<int> gl{get_grid_locations(indx, true)};
        int r = gl.at(0), c = gl.at(1), e = gl.at(2);
        remove(r, c, e);
    }
    void remove(int row, int column, int elevation) {
        if (elements_list.size() <= 1) return;
        if (DEBUG || DEBUG_RMV) std::cout << "\nRemove (r, c, e): " << row << ", " << column << ", " << elevation << "\n";
        std::shared_ptr<Cube> rmv = elements.at(elevation).at(row).at(column);
        auto it = std::find(elements_list.begin(), elements_list.end(), rmv);
        int index = it - elements_list.begin();
        if (DEBUG || DEBUG_RMV) std::cout << "index: " << index << std::endl;
        elements.at(elevation).at(row).at(column).reset();    // zero out the cube from the grid
        std::vector<bool> visited(elements_list.size(), false); // visited <- boolearn vector same size as # of cubes
        bool done = (elements_list.size() <= 2);
        int search_row = row, search_column = column, search_elevation = elevation;
        std::vector<std::vector<std::vector<int>>> components;
        std::vector<std::vector<int>> starts;
        std::vector<std::vector<int>> queue;
        // Put each of rmv's neighbors into a list as possible start points for search for connected components
        if(elements.at(elevation).at(row-1).at(column) != nullptr) starts.push_back(std::vector<int>{row-1, column, elevation});
        if(elements.at(elevation).at(row+1).at(column) != nullptr) starts.push_back(std::vector<int>{row+1, column, elevation});
        if(elements.at(elevation).at(row).at(column-1) != nullptr) starts.push_back(std::vector<int>{row, column-1, elevation});
        if(elements.at(elevation).at(row).at(column+1) != nullptr) starts.push_back(std::vector<int>{row, column+1, elevation});
        if(elements.at(elevation-1).at(row).at(column) != nullptr) starts.push_back(std::vector<int>{row, column, elevation - 1});
        if(elements.at(elevation+1).at(row).at(column) != nullptr) starts.push_back(std::vector<int>{row, column, elevation + 1});
        if (DEBUG || DEBUG_RMV) std::cout << "Starts\n" << toString(starts) << std::endl;
        // mark the removed cube as visited
        visited.at(index) = true;
        // if there is any cube we haven't visited we'll search for connected components
        for(int i = 0; i < visited.size(); i++) {
            if (!visited.at(i)) done = false;
        }
        if (DEBUG || DEBUG_RMV) {
            std::cout << "Grid Locations: " << toString(get_all_grid_locs(true)) << std::endl;
            std::cout << "Visited\n" << toString(visited) << std::endl;
        }
        while(!done) {
            //!done means some cube has not yet been visited
            //starts is the possible starts of different components
            bool unvisited_start = false;   // if this is not set to true there is no search start left which is unvisited
            std::vector<std::vector<int>> component;
            std::vector<int> current;
            int search_index;
            while(!unvisited_start) {
                // Make sure we only start a search from a neighbor of cube if it was not a part of a component we already found
                current = starts.back(); //get the back of the removed cube's neighbors and its index
                auto search_it = std::find(elements_list.begin(), elements_list.end(), elements.at(current.at(2)).at(current.at(0)).at(current.at(1)));
                search_index = grid_to_list_index(current);
                //remove the neighbor that is now current from starts
                starts.pop_back();
                if(!visited.at(search_index)) {
                    queue.push_back(current);
                    unvisited_start = true;
                }
            }
            if (DEBUG || DEBUG_RMV) std::cout << "start search: " << toString(current) << std::endl;
            if (DEBUG || DEBUG_RMV) std::cout << "Visited\n" << toString(visited) << std::endl;
            if (DEBUG || DEBUG_RMV) std::cout << "Grid Locations: " << toString(get_all_grid_locs(true)) << std::endl;
            //std::cout << "Starts\n" << toString(starts) << std::endl;
            while(queue.size() > 0) {
                if (DEBUG || DEBUG_RMV) std::cout << "Queue: " << toString(queue) << std::endl;
                current = queue.back();
                queue.pop_back();
                if (DEBUG || DEBUG_RMV) std::cout << "Current: " << toString(current) << std::endl;
                search_index = grid_to_list_index(current);
                if (DEBUG || DEBUG_RMV) std::cout << "Search_index: " << search_index << std::endl;
                visited.at(search_index) = true;
                component.push_back(current);
                search_elevation    = current.at(2);
                search_row          = current.at(0);
                search_column       = current.at(1);
                int left, right, up, down, front, back;
                std::vector<int> left_grid  {search_row, search_column-1, search_elevation};
                std::vector<int> right_grid {search_row, search_column+1, search_elevation};
                std::vector<int> up_grid    {search_row, search_column, search_elevation+1};
                std::vector<int> down_grid  {search_row, search_column, search_elevation-1};
                std::vector<int> front_grid {search_row+1, search_column, search_elevation};
                std::vector<int> back_grid  {search_row-1, search_column, search_elevation};
                left  = grid_to_list_index(left_grid);
                right = grid_to_list_index(right_grid);
                up    = grid_to_list_index(up_grid);
                down  = grid_to_list_index(down_grid);
                front = grid_to_list_index(front_grid);
                back  = grid_to_list_index(back_grid);
                if((elements.at(search_elevation).at(search_row).at(search_column-1) != nullptr) && left != -1 &&(!visited.at(left))) {
                    queue.push_back(left_grid);
                }
                if((elements.at(search_elevation).at(search_row).at(search_column+1) != nullptr) && right != -1 && (!visited.at(right))) {
                    queue.push_back(right_grid);
                }
                if((elements.at(search_elevation+1).at(search_row).at(search_column) != nullptr) && up != -1 && (!visited.at(up))) {
                    queue.push_back(up_grid);
                }
                if((elements.at(search_elevation-1).at(search_row).at(search_column) != nullptr) && down != -1 && (!visited.at(down))) {
                    queue.push_back(down_grid);
                }
                if((elements.at(search_elevation).at(search_row+1).at(search_column) != nullptr) && front != -1 && (!visited.at(front))) {
                    queue.push_back(front_grid);
                }
                if((elements.at(search_elevation).at(search_row-1).at(search_column) != nullptr) && back != -1 && (!visited.at(back))) {
                    queue.push_back(back_grid);
                }
            }
            if (DEBUG || DEBUG_RMV) std::cout << "Component: \n";
            if (DEBUG || DEBUG_RMV) std::cout << toString(component) << std::endl;
            components.push_back(component);
            if (DEBUG || DEBUG_RMV) std::cout << "Components (" << "deleted " << row << ", " << column << ", " << elevation << ")" << std::endl;
            if (DEBUG || DEBUG_RMV) for(auto com : components) std::cout << toString(com) << std::endl;
            if (DEBUG || DEBUG_RMV) std::cout << "Visited: " << toString(visited) << std::endl;
            if (DEBUG || DEBUG_RMV) std::cout << "Grid Locations: " << toString(get_all_grid_locs(true)) << std::endl;
            done = true;
            for(int i = 0; i < visited.size(); i++) {
                if (!visited.at(i)) {
                    done = false;
                    break;
                } 
            }
        }
        elements_list.erase(it);
        if (DEBUG || DEBUG_RMV) std::cout << "Erase Cube's Parameters\n";
        int parameters_erase_start = index * PARAMS_PER_CUBE;
        int parameters_erase_end   = parameters_erase_start + PARAMS_PER_CUBE;
        parameters.erase(parameters.begin() + parameters_erase_start, parameters.begin() + parameters_erase_end);
        if(components.size() > 1) {
            if (DEBUG || DEBUG_RMV) std::cout << "Components: " << components.size() << " Grid is not Connected, Adjusting\n";
            std::vector<std::vector<int>> component_to_move{components.at(0)};
            if (component_to_move.at(0) == std::vector<int> {row, column-1, elevation}) {
                adjust_grid_locations(0, 1, 0, component_to_move);
            }
            else if (component_to_move.at(0) == std::vector<int> {row, column+1, elevation}) {
                adjust_grid_locations(0,-1, 0, component_to_move);
            }
            else if (component_to_move.at(0) == std::vector<int> {row, column, elevation+1}) {
                adjust_grid_locations(0, 0, -1, component_to_move);
            } 
            else if (component_to_move.at(0) == std::vector<int> {row, column, elevation-1, }) {
                adjust_grid_locations(0, 0, 1, component_to_move);
            } 
            else if (component_to_move.at(0) == std::vector<int> {row+1, column, elevation}) {
                adjust_grid_locations(-1, 0, 0, component_to_move);
            } 
            else if (component_to_move.at(0) == std::vector<int> {row-1, column, elevation}) {
                adjust_grid_locations(1, 0, 0, component_to_move);
            } 
            //components.erase(components.begin(), components.begin() + 1);
        }
        
        if(components.size() > 1) {
            if (DEBUG || DEBUG_RMV) std::cout << "Rebuild Graph to get connections... make elements null\n";
            for(int i = 0; i < elements.size(); i++) {
                for(int j = 0; j < elements.at(i).size(); j++) {
                    for(int k = 0; k < elements.at(i).at(j).size(); k++) {
                        elements.at(i).at(j).at(k).reset();
                    }
                }
            }
        }
        
        
        if(components.size() > 1) {
            if (DEBUG || DEBUG_RMV) std::cout << "Rebuild Elements\n";
            for(int i = 0; i < elements_list.size(); i++) {
                int e, r, c;
                std::vector<int> gl{get_grid_locations(i, true)};
                e = gl.at(2);
                r = gl.at(0);
                c = gl.at(1);
                add(elements_list.at(i), r, c, e, true, true);
            }       
        }
    }
    void add(bool initial = false, bool special = false) {
        std::shared_ptr<Cube> c {new Cube()};
        std::vector<int> loc = random_location();
        if (special) {
            std::cout << toString(loc) << std::endl;
            std::cout << "Current Spots: " << toString(get_all_grid_locs(true)) << std::endl;
        }
        int elevation = loc.at(2);
        int row = loc.at(0);
        int column = loc.at(1);
        if (DEBUG || DEBUG_ADD) std::cout << "Random add: row " << row << ", column " << column << ", elevation " << elevation << std::endl;
        add(c, row, column, elevation, initial);
        if (DEBUG || DEBUG_ADD) std::cout << "Added successfully\n";
    }
    void add(int row, int column, int elevation, bool initial = false) {
        std::shared_ptr<Cube> c {new Cube()};
        add(c, row, column, elevation, initial);
    }
    void add(std::shared_ptr<Cube> c, bool initial = false) {
        std::vector<int> loc = random_location();
        int elevation = loc.at(2);
        int row = loc.at(0);
        int column = loc.at(1);
        add(c, row, column, elevation, initial);
    }
    void add(std::shared_ptr<Cube> c, int row, int column, int elevation, bool initial = false, bool rebuild = false) {
        if (DEBUG || DEBUG_ADD || rebuild) std::cout << "Add: row " << row << ", column " << column << ", elevation " << elevation << std::endl;
        double connection_strength = 3e7;
        int adjusted_elevation = 0, adjusted_row = 0, adjusted_column = 0; 

        if (elevation <= 0) {
            if (DEBUG || DEBUG_ADD) std::cout << "elevation <= 0 (" << elevation << ")\n";
            for(int i = 0; i < height; i++) {
            elements.insert(elements.begin(), std::vector<std::vector<std::shared_ptr<Cube>>>(length, std::vector<std::shared_ptr<Cube>>(width, nullptr)));
            //elements.insert(elements.begin(), std::vector<std::vector<std::shared_ptr<Cube>>>(length, std::vector<std::shared_ptr<Cube>>(width, nullptr)));
            }
            elevation+=height;
            adjusted_elevation = height;
            height*=2;
        }
        if (elevation >= height - 1) {
            if (DEBUG || DEBUG_ADD) std::cout << "elevation >= height-1 (" << elevation << " >= " << std::to_string(height - 1) << ")\n";
            for(int i = 0; i < height; i++) {
            elements.push_back(std::vector<std::vector<std::shared_ptr<Cube>>>(length, std::vector<std::shared_ptr<Cube>>(width, nullptr)));
            //elements.push_back(std::vector<std::vector<std::shared_ptr<Cube>>>(length, std::vector<std::shared_ptr<Cube>>(width, nullptr)));
            }
            height*=2;
        }
        if (DEBUG || DEBUG_ADD) std::cout << "Height: " << height << ", Width: " << width << ", Length: " << length << std::endl;
        if (row <= 0) {
            if (DEBUG || DEBUG_ADD) std::cout << "row <= 0 (" << row << ")\n";
            for(int i = 0; i < elements.size(); i++) {
                for(int j = 0; j < length; j++) {
                    elements.at(i).insert(elements.at(i).begin(), std::vector<std::shared_ptr<Cube>>(width, nullptr));
                    //elements.at(i).insert(elements.at(i).begin(), std::vector<std::shared_ptr<Cube>>(width, nullptr));
                }
            }
            row+=length;
            adjusted_row = length;
            length*=2;
        }
        if (row >= length-1) {
            if (DEBUG || DEBUG_ADD) std::cout << "row >= length-1 (" << row << " >= " << std::to_string(length - 1) << ")\n";
            for(int i = 0; i < elements.size(); i++) {
                for(int j = 0; j < length; j++) {
                    elements.at(i).push_back(std::vector<std::shared_ptr<Cube>>(width, nullptr));
                    //elements.at(i).push_back(std::vector<std::shared_ptr<Cube>>(width, nullptr));
                }
            }
            length*=2;
        }
        //std::cout << "Height: " << height << ", Width: " << width << ", Length: " << length << std::endl;
        if (column <= 0) {
            if (DEBUG || DEBUG_ADD) std::cout << "col <= 0 (" << std::to_string(column) << ")\n";
            for(int i = 0; i < elements.size(); i++) {
                for(int j = 0; j < elements.at(i).size(); j++) {
                    for(int k = 0; k < width; k++) {
                    elements.at(i).at(j).insert(elements.at(i).at(j).begin(), std::shared_ptr<Cube>(nullptr));
                    //elements.at(i).at(j).insert(elements.at(i).at(j).begin(), std::shared_ptr<Cube>(nullptr));
                    }
                }
            }
            column+=width;
            adjusted_column = width;
            width*=2;
        }
        if (column >= width-1) {
            if (DEBUG || DEBUG_ADD) std::cout << "col >= width-1 (" << column << " >= " << std::to_string(width - 1) << ")\n";
            for(int i = 0; i < elements.size(); i++) {
                for(int j = 0; j < elements.at(i).size(); j++) {
                    for(int k = 0; k < width; k++) {
                    elements.at(i).at(j).push_back(std::shared_ptr<Cube>(nullptr));
                    //elements.at(i).at(j).push_back(std::shared_ptr<Cube>(nullptr));
                    }
                }
            }
            width*=2;
        }
        //std::cout << "Height: " << height << ", Width: " << width << ", Length: " << length << std::endl;
        //if (row < 0 || row > length) throw std::runtime_error("Vector bounds exceeded for 'row' in Grid.add method");
        //if (column < 0 || column > width) throw std::runtime_error("Vector bounds exceeded for 'column' in Grid.add method");
        
        // std::cout << "Height: " << height << ", Width: " << width << ", Length: " << length << std::endl;
        // std::cout << "Elevation: " << elevation << ", Column: " << column << ", Row: " << row << std::endl;
        // std::cout << "Elements Dim: " << elements.size() << " x " << elements.at(0).size() << " x " << elements.at(0).at(0).size() << std::endl;
        if (elevation < 0 || elevation > height) throw std::runtime_error("Vector bounds exceeded for 'height' in Grid.add method");
        if (row < 0 || row > length) throw std::runtime_error("Vector bounds exceeded for 'row' in Grid.add method");
        if (column < 0 || column > width) throw std::runtime_error("Vector bounds exceeded for 'column' in Grid.add method");
        
        // for(int i = 0; i < elements.size(); i++) {
        //     std::cout << "height " << i << " size (rows): " << elements.at(i).size() << std::endl;
        //     for(int j = 0; j < elements.at(i).size(); j++) {
        //         std::cout << "\theight " << i << " row " << j << " size (cols): " << elements.at(i).at(j).size() << std::endl;
        //     }
        // }
        //std::cout << "Attempting to add cube at " << elevation << " " << row << " " << column << std::endl;
        if (adjusted_column + adjusted_elevation + adjusted_row > 0) adjust_grid_locations(adjusted_row, adjusted_column, adjusted_elevation);
        if (!rebuild) elements_list.push_back(c);
        auto index = std::find(open_spots.begin(), open_spots.end(), std::vector<int> {row, column, elevation});
        if (index != open_spots.end()) open_spots.erase(index);

        //Grid indexed by elevation, row, column

        elements.at(elevation).at(row).at(column) = c;
        cube_init_params c_init;
        c_init.STARTING_TRANSLATION_X = 1.0*(column-center_column)+0.5;
        c_init.STARTING_TRANSLATION_Y = 1.0*(row-center_row)+0.5;
        c_init.STARTING_TRANSLATION_X = 1.0*(elevation-ground_level + ground)+0.5;
        c->_init_cube_location(c_init);
        if (DEBUG || DEBUG_ADD) std::cout << "Successfully Added to Grid\n";
        c->set_parameters_default();
        c->set_grid_location((double)row, (double)column, (double)elevation);
        if (DEBUG || DEBUG_ADD) std::cout << "Set cube's params\n";
        // If the cube is not being added as part of the constructor, then we should add the cube's parameters to our parameter list
        if (!initial) {
            // Added a new cube, need to modify the grid's parameters
            if(DEBUG) std::cout << "!initial\n";
            std::vector<double> c_params = c->get_parameters();
            if (DEBUG || DEBUG_ADD) {
                std::cout << "cube being added in position " << row << ", " << column << ", " << elevation << " parameters\n";
                std::cout << toString(c_params, 15) << std::endl;
            }
            std::vector<double> default_cube_params = c->get_default_parameters();
            if(DEBUG) std::cout << toString(default_cube_params) << std::endl;
            if(DEBUG) std::cout << "Got cube's params (" << toString(c_params) << ")\n";
            auto get_std = [](double param) { return (param/(double)8.0 + 0.02);};
            parameters.insert(parameters.end(), c_params.begin(), c_params.end());
            std::transform(default_cube_params.begin()+3, default_cube_params.end(), std::back_inserter(parameters), get_std);
            if(DEBUG || DEBUG_ADD) std::cout << "Adjusted Grid parameters\n";
        }
        //std::cout << "Added Cube to position: " << elevation << " " << row << " " << column << std::endl;
        //std::cout << "Checking Upstairs Neighbor\n";
        std::array<bool, 6> open_faces {true, true, true, true, true, true};
            /*0: above
              1: below
              2: left
              3: right
              4: back
              5: front
            */
        if (elevation < height-1 && elements.at(elevation+1).at(row).at(column) != nullptr) {
            if (DEBUG || DEBUG_ADD) std::cout << "check above\n";
            std::shared_ptr<Cube> above = elements.at(elevation+1).at(row).at(column);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(2*i + 1), above->masses.at(2*i), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
            open_faces.at(0) = false;
        }
        //std::cout << "Checking Downstairs Neighbor\n";
        
        if (elevation > 0 && elements.at(elevation-1).at(row).at(column) != nullptr) {
            if (DEBUG || DEBUG_ADD) std::cout << "check below\n";
            std::shared_ptr<Cube> below = elements.at(elevation-1).at(row).at(column);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(2*i), below->masses.at(2*i + 1), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
            open_faces.at(1) = false;
        }
        //std::cout << "Checking left Neighbor\n";
        if (column > 0 && elements.at(elevation).at(row).at(column-1) != nullptr) {
            if (DEBUG || DEBUG_ADD) std::cout << "check left\n";
            std::shared_ptr<Cube> left = elements.at(elevation).at(row).at(column-1);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(i), left->masses.at(i+4), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
            open_faces.at(2) = false;

        }
        //std::cout << "Checking right Neighbor\n";

        if (column < length - 1 && elements.at(elevation).at(row).at(column+1) != nullptr) {
            if (DEBUG || DEBUG_ADD) std::cout << "check right\n";
            std::shared_ptr<Cube> right = elements.at(elevation).at(row).at(column+1);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(i+4), right->masses.at(i), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
            open_faces.at(3) = false;
        }
        //std::cout << "Checking rear Neighbor (" << elevation << " " << row-1 << " " << column << ")\n";
        if (row > 0 && elements.at(elevation).at(row-1).at(column) != nullptr) {
            if (DEBUG || DEBUG_ADD) std::cout << "check behind\n";
            std::shared_ptr<Cube> behind = elements.at(elevation).at(row-1).at(column);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(2*i + 2 - (i%2)), behind->masses.at(2*i-(i%2)), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
            open_faces.at(4) = false;
        }
        //std::cout << "Checking front Neighbor\n";
        if (row < width - 1 && elements.at(elevation).at(row+1).at(column) != nullptr) {
            if (DEBUG || DEBUG_ADD) std::cout << "check front\n";
            std::shared_ptr<Cube> front = elements.at(elevation).at(row+1).at(column);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(2*i-(i%2)), front->masses.at(2*i + 2 - (i%2)), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
            open_faces.at(5) = false;
        }
        if (DEBUG || DEBUG_ADD) std::cout << "Connections Added\n";
        if (open_faces.at(0) && std::find(open_spots.begin(), open_spots.end(), std::vector<int> {row, column, elevation+1}) == open_spots.end()) {
            open_spots.push_back(std::vector<int> {row, column, elevation+1});
        }
        if (open_faces.at(1) && std::find(open_spots.begin(), open_spots.end(), std::vector<int> {row, column, elevation-1}) == open_spots.end()) {
            open_spots.push_back(std::vector<int> {row, column, elevation-1});
        }
        if (open_faces.at(2) && std::find(open_spots.begin(), open_spots.end(), std::vector<int> {row, column -1, elevation}) == open_spots.end()) {
            open_spots.push_back(std::vector<int> {row, column-1, elevation});
        }
        if (open_faces.at(3) && std::find(open_spots.begin(), open_spots.end(), std::vector<int> {row, column+1, elevation}) == open_spots.end()) {
            open_spots.push_back(std::vector<int> {row, column +1, elevation});
        }
        if (open_faces.at(4) && std::find(open_spots.begin(), open_spots.end(), std::vector<int> {row-1, column, elevation}) == open_spots.end()) {
            open_spots.push_back(std::vector<int> {row-1, column, elevation});
        }
        if (open_faces.at(5) && std::find(open_spots.begin(), open_spots.end(), std::vector<int> {row+1, column, elevation}) == open_spots.end()) {
            open_spots.push_back(std::vector<int> {row+1, column, elevation});
        }
        if (DEBUG || DEBUG_ADD) {
            std::cout << "Open Spots\n";
            for(int i = 0; i < open_spots.size(); i++) {
                std::cout << toString(open_spots.at(i), -1, ", ", true) << " ";
            }
            std::cout << std::endl;
        }
    }

    glm::dvec3 compute_centroid() {
        // Both sets and returns the centroid
        glm::dvec3 centroids_accumulator {0.0, 0.0, 0.0};
        int num_cubes = elements_list.size();
        int num_masses_per_cube = elements_list.at(0)->masses.size();
        for(int i = 0; i < elements_list.size(); i++) {
            for(int j = 0; j < elements_list.at(i)->masses.size(); j++) {
                glm::dvec3 vertex_pos = elements_list.at(i)->masses.at(j)->position;
                centroids_accumulator += (vertex_pos)/((double)num_masses_per_cube*(double)num_cubes);
            }
        }
        centroid = centroids_accumulator;
        return centroid;
    }

    void update() {
        //std::cout << "Start of grid update\n";
        for(int i = 0; i < elements_list.size(); i++) {
            elements_list.at(i)->update();
        }
        for(int i = 0; i < connections.size(); i++) {       //Here we are trying to enforce the connections between cubes
            connections.at(i)->update_mass_forces(1);        //by placing short and stiff springs between them. But this update
                                                            //depends on the cube's position update, which in turn depends on this                                                //this update. So this process may need adjustment
        }
        // for(int i = 0; i < elements_list.size(); i++) {
        //     elements_list.at(i)->update();
        // }
        //std::cout << "End of grid update\n";

    }
};


bool _need_collision_check(std::shared_ptr<Cube> cube1, std::shared_ptr<Cube> cube2) {
    BB bb1 = cube1->bb;
    BB bb2 = cube2->bb;

    if (bb1.maxes.x < bb2.mins.x) return false;
    if (bb1.mins.x > bb2.maxes.x) return false;
    if (bb1.maxes.y < bb2.mins.y) return false;
    if (bb1.mins.y > bb2.maxes.y) return false;
    if (bb1.maxes.z < bb2.mins.z) return false;
    if (bb1.mins.z > bb2.maxes.z) return false;
    return true;
}

class CollisionHandler {

    public:

    std::vector<std::shared_ptr<Cube>>   cubes;
    std::unordered_map<std::shared_ptr<Cube>, std::vector<std::shared_ptr<Spring>>> spring_map;

    CollisionHandler() : spring_map(std::unordered_map<std::shared_ptr<Cube>, std::vector<std::shared_ptr<Spring>>> {}) {};

    void add_cube_pair(std::shared_ptr<Cube> cube1, std::shared_ptr<Cube> cube2) {
        if(!cubes.empty()) {
            for(int i = 0; i < cubes.size()-1; i+=2) {
                std::cout << cubes.size();
                if((cubes.at(i) == cube1 && cubes.at(i+1) == cube2) || (cubes.at(i+1) == cube1 && cubes.at(i) == cube2)) return;
            }
        }
        cubes.push_back(cube1);
        cubes.push_back(cube2);
        spring_map.emplace(cube1, std::vector<std::shared_ptr<Spring>> {});
        spring_map.emplace(cube2, std::vector<std::shared_ptr<Spring>> {});
    }

    void remove_cube_pair(int cube0_index, int cube2_index) {
        std::shared_ptr<Cube> cube0 = cubes.at(cube0_index);
        std::shared_ptr<Cube> cube2  = cubes.at(cube2_index);
        auto it0 = spring_map.find(cube0);
        auto it1 = spring_map.find(cube2);
        if (it0 != spring_map.end()) spring_map.erase(it0);
        if (it1 != spring_map.end()) spring_map.erase(it1);
        cubes.erase(cubes.begin()+cube0_index, cubes.begin() + cube2_index+1);
    }



    void update() {
        if (DEBUG_SPRINGS) std::cout << std::endl << "======================================= Springs: Timestep " << (int)(t/dt) <<
         "==================================================" << std::endl << std::endl;
            if(!cubes.empty()) {
                checkForCollisions();
                // for(int i = 0; i < cubes.size(); i++) {
                //     std::vector<std::shared_ptr<Spring>> springs = spring_map[cubes.at(i)];
                //     std::cout << "Cube " << i << " Springs " << springs.size() << std::endl;
                //     // for(int j = 0; j < springs.size(); j++) {
                //     //     springs[j]->update_mass_forces(true);
                //     // }
                //     spring_map[cubes.at(i)].clear(); 

                // }
            }

        }

    void update(double &cumulative_evaluation_times, std::string spring_eval_filename) {
        double eval_this_timestep = 0;
        int numsprings;
        for(int i = 0; i < cubes.size(); i++) {
            std::vector<std::shared_ptr<Spring>> springs = spring_map[cubes.at(i)];
            for(int j = 0; j < springs.size(); j++) {
                auto start = std::chrono::steady_clock::now();
                springs[j]->update_mass_forces(true);
                 auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                double time_in_ms = elapsed.count();
                eval_this_timestep += time_in_ms;
            }
            numsprings += springs.size();
        }
        cumulative_evaluation_times += eval_this_timestep;
        log_evaluation_times(cumulative_evaluation_times,eval_this_timestep, spring_eval_filename, numsprings);
    }
    void log_evaluation_times(double &cumulative_evaluation_time, double instantaneous_time, std::string spring_eval_filename, int numsprings) {
        std::ofstream spring_log(spring_eval_filename, std::ios_base::app);
        if (!spring_log.is_open()) {
            std::cout << "Could Not Open Logfile for Spring Evaluation at " << spring_eval_filename << std::endl;
        }
        double timestep;
        double num_springs = numsprings;
        timestep = ((t/dt) + 1.0);
        double avg_evals_per_second = (num_springs * timestep)/cumulative_evaluation_time;
        double instantaneous_evals_per_second = num_springs/instantaneous_time;
        spring_log << std::to_string(t) << ", " << std::to_string(instantaneous_evals_per_second) << ", " << avg_evals_per_second << std::endl;
        spring_log.close();
    }

    
    glm::dvec3 barycentric(glm::dvec3 a, glm::dvec3 b, glm::dvec3 c, glm::dvec3 p) {
        //algorithm from "Realtime Collision Detection"
        glm::dvec3 v0 = b-a;
        glm::dvec3 v1 = c-a;
        glm::dvec3 v2 = p-a;

        double d00 = glm::dot(v0, v0);
        double d01 = glm::dot(v0, v1);
        double d11 = glm::dot(v1, v1);
        double d20 = glm::dot(v2, v0);
        double d21 = glm::dot(v2, v1);
        double denom = d00*d11 - d01*d01;
        double v = (d11*d20 - d01*d21) / denom;
        double w = (d00*d21 - d01*d20) / denom;
        double u = 1.0 - v - w;
        glm::dvec3 coords {u, v, w};
        return coords;
    }
    bool pointInTriangle(glm::dvec3 &a, glm::dvec3 &b, glm::dvec3 &c, glm::dvec3 &p) {
        glm::dvec3 bc = barycentric(a,b,c,p);
        return bc.y > 0.0 && bc.z > 0.0 && bc.y + bc.z < 1.0;
    }
    void HandleCollisions(int cube0_index, int cube2_index) {
        std::shared_ptr<Cube> cube0 = cubes[cube0_index];
        std::shared_ptr<Cube> cube2 = cubes[cube2_index];
        
        std::shared_ptr<Cube> cube_pair[] {cube0, cube2}; 
        for( std::shared_ptr<Cube> cube1 : cube_pair){
            std::shared_ptr<Cube> other_cube;
            if(cube1 == cube0) {
                other_cube = cube2;
            }
            else other_cube= cube0;
            BB bb2 = other_cube->bb;
            for(int i = 0; i < cube1->masses.size(); i++) {
                glm::dvec3 p = cube1->masses.at(i)->position;
                if ((p.x > bb2.mins.x && p.x < bb2.maxes.x) && (p.y > bb2.mins.y && p.y < bb2.maxes.y) && (p.z > bb2.mins.z && p.z < bb2.maxes.z)) {
                    if(pointInsidePolygon(p, other_cube)) {
                        //std::cout << "p: ";
                        //pprint(p);
                        // for(int mi = 0; mi < other_cube->masses.size(); mi++) {
                        //     std::cout << mi << ": ";
                        //     pprint(other_cube->masses.at(mi)->position);
                        // }

                        std::array<GLuint, 36> faces = other_cube->get_faces();
                        double shortest_distance = 1e9;
                        glm::dvec3 projection;
                        for(int f = 0; f < faces.size()/3; f++) {
                            int s_index = f*3;
                            std::array<double, 3> v0 {other_cube->masses.at(faces[s_index])->position.x, other_cube->masses.at(faces[s_index])->position.y, other_cube->masses.at(faces[s_index])->position.z};
                            std::array<double, 3> v1 {other_cube->masses.at(faces[s_index+1])->position.x, other_cube->masses.at(faces[s_index+1])->position.y, other_cube->masses.at(faces[s_index+1])->position.z};
                            std::array<double, 3> v2 {other_cube->masses.at(faces[s_index+2])->position.x, other_cube->masses.at(faces[s_index+2])->position.y, other_cube->masses.at(faces[s_index+2])->position.z};

                            std::array<std::array<double, 3>, 3> triangle = {v0, v1, v2};
                            glm::dvec3 p_bar = closest_point_on_triangle(p, triangle);
                            if (glm::length2(p-p_bar) < shortest_distance) {
                                shortest_distance = glm::length2(p - p_bar);
                                projection = p_bar;
                            }
                        }
                        //handle the collision as the point p penetrating cube 2's face
                        for(int mass_num = 0; mass_num < 8; mass_num++) {
                            std::shared_ptr<Spring> new_spring = Spring::create(other_cube->masses.at(mass_num), cube1->masses.at(i), 0.1, 
                            ((cube1->masses.at(i)->mass)/dt)*0.1/0.00998*G*-1);
                            glm::dvec3 resting_point = projection + 1.5*(projection-p);
                            new_spring->set_resting_length_by_point_and_mass(0, resting_point);
                            //spring_map[cube1].push_back(new_spring);
                            
                            new_spring->update_mass_forces(true);
                        }
                    }
                }
            }
        }
    }

    glm::dvec3 closest_point_on_triangle(glm::dvec3 &p, std::array<std::array<double, 3>, 3> &triangle) {
        glm::dvec3 v0 = glm::dvec3(triangle[0][0], triangle[0][1], triangle[0][2]);
        glm::dvec3 v1 = glm::dvec3(triangle[1][0], triangle[1][1], triangle[1][2]);
        glm::dvec3 v2 = glm::dvec3(triangle[2][0], triangle[2][1], triangle[2][2]);

        glm::dvec3 e10 = v1 - v0;
        glm::dvec3 e20 = v2 - v0;
        glm::dvec3 e21 = v2 - v1;

        double s_num = glm::dot(p-v0, e10);
        double s_denom = glm::dot(p-v1, -e10);
        double t_num = glm::dot(p-v0, e20);
        double t_denom = glm::dot(p-v2, -e20);

        if( s_num < 0.0 && t_num < 0.0) return v0;
        double u_num = glm::dot(p-v1, e21);
        double u_denom = glm::dot(p-v2, -e21);

        if (s_denom < 0.0 && u_num < 0.0) return v1;
        if (t_denom < 0.0 && u_denom < 0.0) return v2;

        glm::dvec3 normal = glm::cross(e10, e20);
        double sp_10 = glm::dot(normal, glm::cross(v0 - p,  v1 - p));

        if(sp_10 < 0 && s_num >= 0 && s_denom >= 0) return v0 + (s_num / (s_num+s_denom)) * e10;

        double sp_21 =  glm::dot(normal, glm::cross(v1 - p, v2 - p));
         if(sp_21 < 0 && u_num >= 0 && u_denom >= 0) return v1 + (u_num / (u_num+u_denom)) * e21;

         double sp_20 =  glm::dot(normal, glm::cross(v2 - p, v0 - p));
         if(sp_20 < 0 && t_num >= 0 && t_denom >= 0) return v0 + (t_num / (t_num+t_denom)) * e20;
       
        double u = sp_21 / (sp_10 + sp_20 + sp_21);
        double v = sp_20 / (sp_10 + sp_20 + sp_21);
        double w = 1.0 - u - v;

        return u*e10 + v*e20 + w*e21;
    }
    
    bool pointInsidePolygon(glm::dvec3 &p,std::shared_ptr<Cube> c) {
        glm::dvec3 d = glm::normalize(glm::dvec3(rand()/(double)RAND_MAX, rand()/(double)RAND_MAX, rand()/(double)RAND_MAX)); // Ray cast in a random direction
        std::array<GLuint, 36> f = c->get_faces();
        int intersection_points = 0;
        bool first_true = false;
        for(int i = 0; i < f.size(); i+=3) {
            if (first_true) {
                first_true = false;
                if((i % 2) == 1) continue; //skip the rest, no need to check the second triangle
            }
            glm::dvec3 v0 = c->masses.at(f[i])->position;
            glm::dvec3 v1 = c->masses.at(f[i+1])->position;
            glm::dvec3 v2 = c->masses.at(f[i+2])->position;
            glm::dvec3 diff1 = v1 - v0;
            glm::dvec3 diff2 = v2 - v0;
            glm::dvec3 normal = glm::cross(diff1, diff2);
            double d_dot_normal = glm::dot(d, normal);

            if (fabs(d_dot_normal > 1e-6)) {
                glm::dvec3 d_source_triangle = v0 - p; // distance between vertex 0 of the triangle and light source
                //this is the distance between the ray source and the intersection of the ray source and the plane generated by the triangle
                double t = glm::dot(normal, d_source_triangle);
                if (t >= 0) {
                    glm::dvec3 intersection = p + d*t;
                    if(pointInTriangle(v0, v1, v2, intersection)){
                        intersection_points += 1;
                        first_true = true;
                    }
                }
            }
        }
        if((intersection_points % 2) == 1) return true;
        return false;
    }
    void checkForCollisions() {
        for(int i = 0; i < cubes.size(); i+=2) {
            //std::cout << "Checking Cubes " << i << " and " << i+1 << "\n";
            //cubes.at(i)->pprint_vertices_matrix();
            //cubes.at(i+1)->pprint_vertices_matrix();
            bool found = false;
            for(int j = 0; j < 8; j++) {
                if(pointInsidePolygon(cubes[i]->masses.at(j)->position, cubes[i+1])) {
                    //std::cout << "Collision Detected Between Cubes " << i << " and " << i+1 << std::endl;
                    //pprint(cubes[i]->masses.at(j)->position);
                    HandleCollisions(i, i+1);
                    found = true;
                    break;
                }
                else if(pointInsidePolygon(cubes[i+1]->masses.at(j)->position, cubes[i])) {
                    //std::cout << "Collision Detected Between Cubes " << i+1 << " and " << i << std::endl;
                    //pprint(cubes[i+1]->masses.at(j)->position);
                    HandleCollisions(i+1, i);
                    found = true;
                    break;
                }
            }
            if(!found) {
                //std::cout << "-- No Collision Detected Between Cubes " << i << " and " << i+1 << std::endl;
                if(!_need_collision_check(cubes.at(i), cubes.at(i+1))) {
                    //std::cout << "Removing the pair" << std::endl;
                    remove_cube_pair(i, i+1);
                }

            }
        }
    }
};



std::string LoadShaderFile(const std::string& filePath) {
    std::string fullpath = "../src/" + filePath;
    std::ifstream shaderFile(fullpath);
    if(!shaderFile.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + fullpath);
    }
    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    return shaderStream.str();
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch(key) {
            case GLFW_KEY_UP:
                std::cout << "Up" << std::endl;
                cameraPos += cameraSpeed * cameraFront;
                break;
            case GLFW_KEY_DOWN:
            std::cout << "Down" << std::endl;
                cameraPos -= cameraSpeed * cameraFront;
                break;
            case GLFW_KEY_LEFT:
                std::cout << "Left" << std::endl;
                cameraPos -= cameraSpeed * cameraRight;
                break;
            case GLFW_KEY_RIGHT:
                std::cout << "Right" << std::endl;
                cameraPos += cameraSpeed * cameraRight;
                break;
            case GLFW_KEY_W:
                std::cout << "W" << std::endl;
                cameraPos += cameraSpeed * worldUp;
                break;
            case GLFW_KEY_S:
                std::cout << "S" << std::endl;
                cameraPos -= cameraSpeed * worldUp;
                break;
        }
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        if(glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    }
    view = glm::lookAt(
        cameraPos,                       // Camera location in world space
        cameraPos + cameraFront,         // Center of view
        cameraUp                         // 0efines the direction of "up"
    );

}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if(firstMouse) {
        prevX = xpos;
        prevY = ypos;
        firstMouse = false;
    }
    float xoffset = xpos - prevX;
    float yoffset = prevY - ypos;
    prevX = xpos;
    prevY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch = std::min(std::max(pitch+yoffset, -89.0f), 89.0f);

    glm::vec3 front;
    front.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
    front.y = std::sin(glm::radians(pitch));
    front.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);

    cameraRight = glm::normalize(glm::cross(cameraFront, worldUp));
    cameraUp    = glm::normalize(glm::cross(cameraRight, cameraFront));

    view = glm::lookAt(
        cameraPos,
        cameraPos + cameraFront,
        cameraUp
    );
}

void pprint(glm::dmat4 A) {
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
}



std::vector<double> test_params {2.000000, 2.000000, 2.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.032500, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.041651, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.041651, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.041651, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.041651, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 2.000000, 3.000000, 2.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.032500, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.041651, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.041651, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.041651, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.041651, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.037678, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700, 0.032500, 0.145000, 0.145000, 0.412700};

int main() {
    

    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

    GLFWwindow* window = glfwCreateWindow(800, 600, "GraphicsWindow", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    bool doublePrecisionSupported = glewIsSupported("GL_ARB_gpu_shader_fp64");
    if (doublePrecisionSupported) std::cout << "Double Precision Supported on this Hardware\n";
    else std::cout << "Double Precision Not Supported\n";
    const GLubyte* glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

    std::cout << "OPENGL VERSION: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Supported GLSL Version: " << glslVersion << std::endl;

    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const std::string vertexShaderSource = LoadShaderFile("vertex_shader.glsl");
    const char* vertexShaderSourceCStr = vertexShaderSource.c_str();
    glShaderSource(vertexShader, 1, &vertexShaderSourceCStr, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    
    const std::string fragmentShaderSource = LoadShaderFile("fragment_shader.glsl");
    const char* fragmentShaderSourceCStr = fragmentShaderSource.c_str();
    glShaderSource(fragmentShader, 1, &fragmentShaderSourceCStr, NULL);
    glCompileShader(fragmentShader);

    unsigned int geometryShader;
    geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
    const std::string geometryShaderSource = LoadShaderFile("geometry_shader.glsl");
    const char* geometryShaderSourceCStr = geometryShaderSource.c_str();
    glShaderSource(geometryShader, 1, &geometryShaderSourceCStr, NULL);
    glCompileShader(geometryShader);

    unsigned int gridVertexShader;
    gridVertexShader = glCreateShader(GL_VERTEX_SHADER);
    const std::string gridVertexShaderSource = LoadShaderFile("grid_vertex_shader.glsl");
    const char* gridVertexShaderSourceCStr = gridVertexShaderSource.c_str();
    glShaderSource(gridVertexShader, 1, &gridVertexShaderSourceCStr, NULL);
    glCompileShader(gridVertexShader);

    unsigned int gridFragmentShader;
    gridFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    
    const std::string gridFragmentShaderSource = LoadShaderFile("grid_fragment_shader.glsl");
    const char* gridFragmentShaderSourceCStr = gridFragmentShaderSource.c_str();
    glShaderSource(gridFragmentShader, 1, &gridFragmentShaderSourceCStr, NULL);
    glCompileShader(gridFragmentShader);

    glfwSetKeyCallback(window, key_callback);
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    //glfwSetCursorPosCallback(window, mouse_callback);
  
    glm::mat4 projection = glm::perspective(
        glm::radians(60.0f),            // Field of view
        4.0f/3.0f,                      // Aspect ratio for 600,800 window
        0.1f,                           // Near Clipping Plane
        100.0f                          // Far clipping plane
    );

    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 mvp = projection * view * model;


    std::vector<float> grid;
    for (int i = -1000; i < 1000; i++) {
        grid.push_back(float(i/5.0f));
        grid.push_back(1.5f);
        grid.push_back(-15.0f);
        grid.push_back(float(i/5.0f));
        grid.push_back(1.5f);
        grid.push_back(15.0f);
    }

    for (int i = -1000; i < 1000; i++) {
        grid.push_back(-15.0f);
        grid.push_back(1.5f);
        grid.push_back(float(i/5.0f));
        grid.push_back(15.0f);
        grid.push_back(1.5f);
        grid.push_back(float(i/5.0));
    }
    for (int i = -1000; i < 1000; i++) {
        grid.push_back(float(i/5.0f));
        grid.push_back(-15.0f);
        grid.push_back(-1.3f);
        grid.push_back(float(i/5.0f));
        grid.push_back(15.0f);
        grid.push_back(-1.3f);
    }

    for (int i = -1000; i < 1000; i++) {
        grid.push_back(-15.0f);
        grid.push_back(float(i/5.0f));
        grid.push_back(-1.3f);
        grid.push_back(15.0f);
        grid.push_back(float(i/5.0f));
        grid.push_back(-1.3f);
    }
    /*if(DEBUG) {
        for(int i = 0; i < grid.size(); i++) {
            if((i+1) % 6 != 0){
                std::cout << grid.at(i) << ", ";
            }
            else {
                std::cout << grid.at(i) << std::endl;
            }
        }
    }*/
    GLuint gridVBO, gridVAO;
    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);

    glBindVertexArray(gridVAO);

    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, grid.size() * sizeof(float),  &grid[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);

    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    // glBindVertexArray(0);
    // unsigned int VAO;
    // glGenVertexArrays(1, &VAO);
    // glBindVertexArray(VAO);

    // unsigned int VBO;
    // glGenBuffers(1, &VBO);
    // glBindBuffer(GL_ARRAY_BUFFER, VBO);

    std::cout << "Creating Cube 1 params" << std::endl;
    cube_init_params cube1_params;
        cube1_params.STARTING_TRANSLATION_X = 0.0;
        cube1_params.STARTING_TRANSLATION_Y = 0.0;
        cube1_params.STARTING_TRANSLATION_Z = -0.5;

        cube1_params.STARTING_ROT_X = 0.0;
        cube1_params.STARTING_ROT_Y = 0.0;
    std::cout << "Creating Cube 2 params" << std::endl;
    cube_init_params cube2_params;
        cube2_params.STARTING_TRANSLATION_X = 0.5;
        cube2_params.STARTING_TRANSLATION_Y = 0.0;
        cube2_params.STARTING_TRANSLATION_Z = -0.5;

        cube2_params.STARTING_ROT_X = 0.0;
        cube2_params.STARTING_ROT_Y = 0.0;
    
    RNG rng = RNG(0.0, 0.01, 4*28*2);
    std::vector<double> initial_means;
    std::vector<double> initial_std;
    std::vector<double> initial_parameters;
    for(int i = 0; i < 28 * 2; i++) {
        initial_parameters.push_back(0.1);
        initial_parameters.push_back(0.01);
        initial_parameters.push_back(1.0);
        initial_parameters.push_back(0.0);

        initial_means.push_back(0.0);
        initial_means.push_back(0.0);
        initial_means.push_back(0.0);
        initial_means.push_back(0.0);
    }
    Grid g = Grid(0, 0, 0);
    std::cout << "Adding randomly: \n";
    g.add();
    g.add();
    std::vector<double> g_init_params {g.get_parameters()};
    // std::cout << "G parameters: \n";
    // std::cout << toString(g_init_params) << "\n";


    Grid g2 = Grid(0, 0, 0);
    g2.set_parameters(test_params);
    // std::cout << "G2 Params\n\n";
    // std::cout << toString(g2.get_parameters());
    // std::cout << "\n\nSize g2 params: " << g2.get_parameters().size() << std::endl;
    // std::cout << "Size Params: " << test_params.size() << std::endl;
    // std::cout << PARAMS_PER_CUBE << std::endl;
    // std::cout << test_params.size() / (double)PARAMS_PER_CUBE << std::endl;
    // std::cout << "\n****G2 Initial Elements" << " *******" << std::endl;
    for(int i = 0; i < g2.elements_list.size(); i++) {
        //std::cout << "\n****G2 Initial Element " << i << " *******" << std::endl;
        // std::cout << toString(g2.get_cube_params(i), 12) << std::endl << std::endl;
        // std::cout << "***** Standard Deviations: " << i << " *****" << std::endl;
        // std::cout << toString(g2.get_stds(i), 12) << std::endl << std::endl;
        //std::cout << "***** Grid Location: " << i << " *****" << std::endl;
        std::cout << toString(g2.get_grid_locations(i), 12) << " "; // << std::endl << std::endl;
    }
    g2.add();
    std::cout << "\n****G2 Elements " << "After Adding  Cube Randomly *******" << std::endl;
    for(int i = 0; i < g2.elements_list.size(); i++) {
        //std::cout << "\n****G2 Element " << i << " After Adding 1 Cube Randomly *******" << std::endl;
        // std::cout << toString(g2.get_cube_params(i), 12) << std::endl;
        // std::cout << "***** Standard Deviations: " << i << " *****" << std::endl;
        // std::cout << toString(g2.get_stds(i), 12) << std::endl << std::endl;
        //std::cout << "***** Grid Location: " << i << " *****" << std::endl;
        std::cout << toString(g2.get_grid_locations(i), 12) << " "; // << std::endl << std::endl;
    }
    g2.add();
    g2.add();
    g2.add();
    g2.add();
    g2.add();
    g2.add();
    
    std::cout << "\n****G2 Elements " << "After Adding 7 Cubes Randomly *******" << std::endl;
    for(int i = 0; i < g2.elements_list.size(); i++) {
        //std::cout << "\n****G2 Element " << i << " After Adding 3 Cubes Randomly *******" << std::endl;
        // std::cout << toString(g2.get_cube_params(i), 12) << std::endl;
        // std::cout << "***** Standard Deviations: " << i << " *****" << std::endl;
        // std::cout << toString(g2.get_stds(i), 12) << std::endl << std::endl;
        //std::cout << "***** Grid Location: " << i << " *****" << std::endl;
        std::cout << toString(g2.get_grid_locations(i), 12) << " ";// << std::endl << std::endl;
    }
    g2.remove();
    std::cout << "\n****G2 Elements " << "After Removing 1 Cubes Randomly *******" << std::endl;
    for(int i = 0; i < g2.elements_list.size(); i++) {
        //std::cout << "\n****G2 Element " << i << " After Adding 3 Cubes Randomly *******" << std::endl;
        // std::cout << toString(g2.get_cube_params(i), 12) << std::endl;
        // std::cout << "***** Standard Deviations: " << i << " *****" << std::endl;
        // std::cout << toString(g2.get_stds(i), 12) << std::endl << std::endl;
        //std::cout << "***** Grid Location: " << i << " *****" << std::endl;
        std::cout << toString(g2.get_grid_locations(i), 12) << " ";// << std::endl << std::endl;
    }
    g2.remove();
    std::cout << "\n****G2 Elements " << "After Removing 2 Cubes Randomly *******" << std::endl;
    for(int i = 0; i < g2.elements_list.size(); i++) {
        //std::cout << "\n****G2 Element " << i << " After Adding 3 Cubes Randomly *******" << std::endl;
        // std::cout << toString(g2.get_cube_params(i), 12) << std::endl;
        // std::cout << "***** Standard Deviations: " << i << " *****" << std::endl;
        // std::cout << toString(g2.get_stds(i), 12) << std::endl << std::endl;
        //std::cout << "***** Grid Location: " << i << " *****" << std::endl;
        std::cout << toString(g2.get_grid_locations(i), 12) << " ";// << std::endl << std::endl;
    }
    g2.remove();
    std::cout << "\n****G2 Elements " << "AfterRemoving 3 Cubes Randomly *******" << std::endl;
    for(int i = 0; i < g2.elements_list.size(); i++) {
        //std::cout << "\n****G2 Element " << i << " After Adding 3 Cubes Randomly *******" << std::endl;
        // std::cout << toString(g2.get_cube_params(i), 12) << std::endl;
        // std::cout << "***** Standard Deviations: " << i << " *****" << std::endl;
        // std::cout << toString(g2.get_stds(i), 12) << std::endl << std::endl;
        //std::cout << "***** Grid Location: " << i << " *****" << std::endl;
        std::cout << toString(g2.get_grid_locations(i), 12) << " ";// << std::endl << std::endl;
    }
    std::cout << "***** g2 Final Grid Locations: \n";
    for(int i = 0; i < g2.elements_list.size(); i++) {
        //std::cout << "***** g2 Final Grid Location: " << i << " *****" << std::endl;
        std::cout << toString(g2.get_grid_locations(i), 12) << " ";// << std::endl << std::endl;
    }
    std::vector<double> g2params = g2.get_parameters();
    g.set_parameters(g2params);
    std::cout << (g.get_parameters() == g2.get_parameters()) << "\n"; 

    // std::cout << "Perturbation Vector\n";
    // std::cout << toString(g.generate_perturbations());
    if (DEBUG) std::cout << "Before creating the cube!" << std::endl;
    std::shared_ptr<Cube> firstcube  {new Cube};
    std::shared_ptr<Cube> secondcube {new Cube};
    if (DEBUG) std::cout << "AFTER creating the cubes!" << std::endl;
    //std::cout << "Initializing cube 1's parameters" << std::endl;
    //firstcube->_init_cube_location(cube1_params);
    //std::cout << "Initializing cube 2's parameters" << std::endl;
    //secondcube->_init_cube_location(cube2_params);
    std::cout << "Adding First Cube\n";
    g.add(firstcube);//  , 0, 0, 0);
    std::cout << "Added First Cube\n";
    std::cout << "Adding Second Cube\n";
    g.add(secondcube);//, 1, 0, 0);
    std::cout << "Added Second Cube\n";
    g.make_dynamic();
    g.set_parameters_default();
    initial_parameters = g.get_parameters();

    for(int i = 0; i < initial_parameters.size(); i++) {
        //std::cout << "Initial Parameter: " << initial_parameters.at(i);
        initial_std.push_back((initial_parameters.at(i) + 0.1)/10);
        //std::cout << "Std: " << initial_std.at(i) << std::endl;
        initial_means.push_back(0.0);
    }
    rng.set_params(initial_means, initial_std);
 
    std::vector<std::shared_ptr<Cube>> cubes;
    std::vector<std::shared_ptr<Cube>> discrete_elements;

    std::vector<Grid> cube_grids;
    cubes.push_back(firstcube);
    cubes.push_back(secondcube);

    cube_grids.push_back(g);
    //std::cout << "Added cube_grid to list\n";
   // return 1;

    CollisionHandler *handler = new CollisionHandler();

    std::vector<GLuint> VAOs(cubes.size()), VBOs(2*cubes.size()), EBOs(cubes.size());

    std::vector<std::array<float, 24>>      vertices_of_cube;
    std::vector<std::array<GLuint, 36>>     faces_of_cube;
    std::vector<std::array<float, 36>>      colors_of_cube;
    for (int i = 0; i < cubes.size(); i++){
        glGenVertexArrays(1, &VAOs[i]);
        glGenBuffers(1, &VBOs[2*i]);

        glBindVertexArray(VAOs[i]);

        glBindBuffer(GL_ARRAY_BUFFER, VBOs[2*i]);
        auto & vertices = cubes[i]->get_vertices();
    /*
    Convert vertices to float for openGL while retaining the double value for our computations.*/
        std::array<float, 24> fvertices;
        std::transform(vertices.begin(), vertices.end(), fvertices.begin(), [](const double & value)-> float {return static_cast<float>(value); });
    
        //vertices_of_cube.push_back(fvertices);
        std::array<GLuint, 36> faces {cubes[i]->get_faces()};

        glBufferData(GL_ARRAY_BUFFER, fvertices.size() * sizeof(float), fvertices.data(), GL_STATIC_DRAW);
    //glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        //unsigned int VBC;
        glGenBuffers(1, &VBOs[2*i+1]);
        glBindBuffer(GL_ARRAY_BUFFER, VBOs[2*i+1]);
        double colors[] = {
          1.0,   0.0,  0.0,
          1.0,   0.5,  0.5,
          0.0,   0.5,  0.0, //
          0.0,   0.75,  0.5,
          0.5,   1.0,  0.2, 
          0.5,   0.5,  0.0, //

           0.0,  1.0,  0.0,
           0.5,  0.0,  0.5,
           0.5,  0.5,  0.5,
           0.5,  0.5,  0.5,
           0.0,  0.5,  1.0,
           0.0,  0.5,  1.0,

           0.0,  0.0,  1.0,
           0.0,  1.0,  0.0,
           0.3,  0.3,  0.3,
           0.3,  0.3,  0.3,
           1.0,  1.0,  0.0,
           0.5,  0.0,  0.5,

           0.5,   0.25, 0.15,
           0.25,  0.5,  0.25,
           0.5,   0.5,  0.5,
           0.5,   0.5,  0.5,
           0.9,   0.1,  0.5,
           0.5,   0.8,  0.2,

           0.5,   0.75, 0.25,
           1.0,   1.0,  1.0,
           0.5,   0.5,  0.5,
           0.5,   0.5,  0.5,
           0.3,   0.3,  0.3,
           0.5,   0.2,  0.5,

           0.0,   0.0,  0.0,
           0.1,   0.9,  0.1,
           0.2,   0.8,  0.2,
           0.2,   0.8,  0.2,
           0.3,   0.7,  0.3,
           0.4,   0.6,  0.4,
    };
    std::array<float, 36> fcolors;
    std::transform(colors, colors + 36, fcolors.begin(), [](const double & value)-> float {return static_cast<float>(value); });
    glBufferData(GL_ARRAY_BUFFER, sizeof(fcolors), fcolors.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    GLuint elementbuffer;
    glGenBuffers(1, &EBOs[i]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOs[i]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(GLuint), &faces[0], GL_DYNAMIC_DRAW);
    }
    unsigned int shaderProgram;
    unsigned int gridShaderProgram;
    shaderProgram = glCreateProgram();
    gridShaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader); 
    glAttachShader(gridShaderProgram, gridVertexShader);
    glAttachShader(shaderProgram, geometryShader);
    glAttachShader(shaderProgram, fragmentShader);
    glAttachShader(gridShaderProgram, gridFragmentShader);
    glBindAttribLocation(shaderProgram, 0, "aPos");
    glBindAttribLocation(shaderProgram, 1, "aColor");
    glBindAttribLocation(gridShaderProgram, 0, "inPosition");
    // glBindAttribLocation(gridShaderProgram, 1, "aColor");


    glLinkProgram(shaderProgram);
    glLinkProgram(gridShaderProgram);
    glUseProgram(shaderProgram);

    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader,GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout <<"ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    glGetShaderiv(fragmentShader,GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout <<"ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    glGetShaderiv(geometryShader,GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(geometryShader, 512, NULL, infoLog);
        std::cout <<"ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog );
        std::cout << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(geometryShader);

    GLuint MatrixID = glGetUniformLocation(shaderProgram, "MVP");
    GLuint gridMatrixID = glGetUniformLocation(gridShaderProgram, "MVP"); 

    glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 lightPos   = glm::vec3(-1.0f, -1.0f, 10.0f);
    //cameraPos defined above

    glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));
    glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);
    glUniformMatrix4fv(gridMatrixID, 1, GL_FALSE, &mvp[0][0]);

    // glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    /*Here we are logging the vertices is faces
    
    */
    // if (DEBUG) {

    //     for(int i = 0; i < faces.size(); i++) {
    //     int ind = faces[i];
    //     GLdouble vertex[3] = {
    //         fvertices[ind*3],
    //         fvertices[ind * +1],
    //         fvertices[ind * 3 + 2]
    //     };
    //     std::cout << i << ": (" << vertices[ind * 3] << ", " << vertices[ind * 3 + 1] << ", " << vertices[ind * 3 + 2] << ")\n";
    //     }
    // }
    

    //glUseProgram(shaderProgram);
    // glBindVertexArray(VAO);
    // glEnableVertexAttribArray(0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    int renderloop_count = 1;
    int print_vertices_every_n_loops = 5;
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_time);
    char buffer[256];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", now_tm);
    
    
    std::shared_ptr<Cube> cube1 {new Cube};
    std::shared_ptr<Cube> cube2 {new Cube};
    cube1->_init_cube_location(cube1_params);
    cube2->_init_cube_location(cube2_params);
    Grid ng = Grid(3,3,3,std::move(initial_parameters));
    ng.add(cube1, 0, 0, 0);
    ng.add(cube2, 1, 0, 0);
    ng.make_dynamic();
    ng.set_parameters_default();
    //cube_grids.push_back(cg);

    std::vector<std::vector<double>> population;
    double cumulative_spring_evaluation_time = 0;
    uint8_t draw_every = 25;
    int generation = 0; 
    int population_size = 35; //35
    std::cout << "Initializing Population\n";
    for(int i = 0; i < population_size; i++) {
        Grid grid_population = Grid(0, 0, 0);
        int rand_int = (rand() % 3) + 2;
        for(int j = 0; j < rand_int; j++) {
            grid_population.add();
        }
        population.push_back(grid_population.get_parameters());
    }
    bool render = false;
    std::vector<double> best_params;
    std::vector<std::vector<double>> all_fitnesses;
    double best_fitness = -1e9;
    int max_generations = 2;
    int render_every_n_generations = 1;
    int log_every_n_generations = 100;
    std::vector<double> fitnesses;
    std::vector<double> best_fitnesses;
    std::vector<double> best_average_velocities;
    std::vector<double> old_std = initial_std;
    while(generation < max_generations) {
        render = false;
        renderloop_count = 0;
        
        if (generation % render_every_n_generations == 0) render = true;
        std::vector<std::vector<double>> population_parameters;
        std::vector<double> average_velocities;

        std::vector<std::vector<double>> offspring;
        std::cout << "Generation " << generation << std::endl;
        //std::cout << "Population Size: " << population_size << std::endl;
        for(int i = 0; i < population_size; i++) {
            if (generation == 0) fitnesses.push_back(-1e9);
            std::vector<double> parent_params = population.at(i);
            offspring.push_back(parent_params);
            //std::cout << "parent params (truncated): " << std::endl;
            // for(int i2 = 0; i2 < 10; i2++) {
            //         std::cout << parent_params.at(i2) << " ";
            //     }
            std::cout << "Setting params " << i << " Parent " << std::endl;
            ng.set_parameters(parent_params);
            for(int j = 0; j < 5 + (5 - (i/7)); j++) { //BRING THIS BACK AFTER DEBUGGING
                if(generation == 0) fitnesses.push_back(-1e9);
                std::cout << "Creating Params for " << i << " Parent --  Child " << j << std::endl;
                std::vector<double> perturbations = ng.generate_perturbations();    //globalRNG.sample(initial_std);
                //std::cout << "got rng sample...\n";
                //.generate_perturbations(population_parameters.at(i+j));
                // std::vector<double> new_params;
                // for(int k = 0; k < parent_params.size(); k++) {
                //     new_params.push_back(parent_params.at(k) + perturbations.at(k));
                // }
                offspring.push_back(perturbations);
            }
        }
        //return 1;
        //std::cout << "\nGenerated Offspring\n";
        /*
        Parameters are stored the difference from the default value, these should at least not be nan, or inf
        They are stored, d_a1, d_b1, d_c1, d_d1, d_a2, d_b2, d_c2, d_d2, ... every c should not be too small so
                            d_c shouldn't be too close to negative one and it should definitely not be less than negative one
        */
        std::ofstream db("../src/out/parameters/allparams_generation_" + std::to_string(generation) + "_.csv");
        for(int i = 0; i < offspring.size(); i++) {
            std::vector<double> os {offspring.at(i)};
            for(int j = 0; j < os.size(); j++) {
                db << os.at(j) << "\t";
            }
            db << "\n";
        }
        db.close();
        std::vector<double> best;
        double best_fitness_this_gen = -1e6; 
        for(int i = 0; i < offspring.size(); i++) {
            ng.restore_defaults();
            ng.set_parameters(offspring.at(i));
            ng.make_dynamic();
            std::cout << "tag cube (offspring " << i << ")\n";

            glm::dvec3 start = ng.compute_centroid();
            double t = 0;
            while(t < 30) {
                ng.update();
                t += dt;
            }
            glm::dvec3 finish = ng.compute_centroid();
            double fitness = std::max(std::abs(finish.x - start.x) - std::abs(finish.z - start.z),  std::abs(finish.y - start.y) - std::abs(finish.z-start.z));
            double avg_velocity = fitness + std::abs(finish.z-start.z);
            average_velocities.push_back(avg_velocity);
            //std::cout << "Fitness: " << fitness <<std::endl;
            fitnesses.at(i) = fitness;
        }
        //std::cout << "Finished Fitness Evaluation: " << fitnesses.size() << std::endl;
        std::vector<int> indices (fitnesses.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&fitnesses](int i1, int i2) {return (fitnesses[i1] > fitnesses[i2] || std::isnan(fitnesses[i2]));});
        best_fitness_this_gen = fitnesses.at(indices.at(0));
        best_average_velocities.push_back(average_velocities.at(indices.at(0)));
        //std::cout << "*** Best Fitness: " << best_fitness_this_gen << std::endl;
        if (best_fitness_this_gen > best_fitness) {
            best_fitness = best_fitness_this_gen;
            best = offspring.at(indices.at(0));
            best_params = best;
        }
        best_fitnesses.push_back(best_fitness_this_gen);

        std::cout << "i\tindices[i]\tfitness" << std::endl;
        for(int i = 0; i < 15; i++) {
            std::cout << i << "\t" << indices.at(i) << "\t" << fitnesses.at(indices.at(i)) << std::endl;
        }

        population.clear();
        std::vector<double> saved_fitnesses;
        for(int i = 0; i < population_size; i++) {
            population.push_back(offspring.at(indices.at(i)));
            saved_fitnesses.push_back(fitnesses.at(indices.at(i)));
            //std::vector<double> offspring_params = offspring.at(indices.at(i));
            // for(int j = 0; j < mean.size(); j++) {
            //     mean.at(j) += (offspring_params.at(j))/(double)population_size;
            // }
        }
        all_fitnesses.push_back(saved_fitnesses);
        //old_mean = mean;
        /*
        Standard deviationshould all be positive and not too close to zero or extremely large
        */
            if(generation % log_every_n_generations == 0) {
            std::string energy_logfile = "../src/out/energy_curves/log_cube_energy_gen_" + std::to_string(generation) +"_"+ std::string(buffer) +  ".csv";
            std::string spring_logfile = "../src/out/spring_performance/spring_evals_gen_"+ std::to_string(generation)+"_" + std::string(buffer) + ".csv";
            std::string parameter_logfile = "../src/out/parameters/parameters_gen_" + std::to_string(generation)+"_"+ std::string(buffer) + ".csv";
            std::string fitnesses_logfile = "../src/out/parameters/fitnesses_gen_"+ std::to_string(generation)+"_" + std::string(buffer) + ".csv";
            std::string learning_curve_logfile = "../src/out/parameters/learning_curve_gen_" + std::to_string(generation)+"_"+ std::string(buffer) + ".csv";
            std::string velocity_logfile = "../src/out/parameters/velocity_gen_"+ std::to_string(generation)+ "_" + std::string(buffer) + ".csv";
            std::ofstream f(parameter_logfile);
            for(int i = 0; i < best_params.size() - 1; i++) {
                f << best_params.at(i) << ", ";
            }
            f << best_params.at(best_params.size() - 1) << std::endl;
            f.close();
            std::ofstream f1(fitnesses_logfile);
            for(int generation = 0; generation < all_fitnesses.size(); generation++) {
                for(int individual = 0; individual < all_fitnesses.at(0).size(); individual++) {
                f1 << all_fitnesses.at(generation).at(individual) << ", ";
                }
                f1 << std::endl;
            }

            f1.close();
            std::ofstream f2(learning_curve_logfile);
            for(int generation = 0; generation < best_fitnesses.size(); generation++) {
                f2 << best_fitnesses.at(generation) << ", ";
            }
            f2.close();
            std::ofstream f3(velocity_logfile);
            for(int generation = 0; generation < best_average_velocities.size(); generation++) {
                f3 << best_average_velocities.at(generation) << std::endl;
            }
            f3.close();
        }
        if(render) {
            std::vector<Grid> render_cube_grids;
            Grid render_grid = Grid(0,0,0);
            render_grid.restore_defaults();
            render_grid.set_parameters(population.at(0));
            render_grid.make_dynamic();
            std::cout << "Rendered Grid Locations: ";
            std::cout << toString(render_grid.get_all_grid_locs(true)) << std::endl;
            for(int cl = 0; cl < render_grid.elements_list.size(); cl++) {
                std::cout << "Cube " << cl << " at location " << toString(render_grid.get_grid_locations(cl, true)) << std::endl;
                render_grid.elements_list.at(cl)->pprint_vertices_matrix();
            }
            render_cube_grids.push_back(g);
            discrete_elements.clear();
            //discrete_elements.push_back(cube_grids.at(0));
            double t = 0;
            int last_int = 0;
            while (t < 30 && !glfwWindowShouldClose(window)) {
                if (glfwWindowShouldClose(window)) {
                    glfwTerminate();
                    return 0;
                }
                if ((int)t > last_int) {
                    std::cout << "Render t: " << t << std::endl;
                    last_int = (int)t;
                }
                    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                    if(!discrete_elements.empty()) {
                        for(int i = 0; i < discrete_elements.size()-1; i++) {
                            for(int j = i+1; j < discrete_elements.size(); j++) {
                                if (_need_collision_check(cubes[i], cubes[j])) {
                                    std::cout << "check needed: " << i << ", " << j << std::endl;

                                    handler->add_cube_pair(cubes[i], cubes[j]);
                                }
                            }
                        }
                    }
                    //std::cout << "Updating handler\n";
                    //handler->update();
                    for (int i = 0; i < cubes.size(); i++) {
                        //processInput(window);
                        std::string a;
                        // std::cout << "Render Loop Count: " << renderloop_count << std::endl;
                        // std::cin >> a;
                        if((DEBUG_VERTICES) && (renderloop_count % print_vertices_every_n_loops == 0)) {
                            std::cout << "Timestep " << renderloop_count << std::endl;
                            cubes[i]->pprint_vertices("def", 48, 40);
                        }
                        //if(LOG_ENERGY) cubes[i]->log_energy(energy_logfile);
                    }
                    glUseProgram(shaderProgram);
                    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);
                    for(int i = 0; i < render_cube_grids.size(); i++) {
                    
                        if(renderloop_count % draw_every == 0) {
                            for(int j = 0; j < cube_grids.at(i).elements_list.size(); j++) {
                                std::shared_ptr<Cube> cube = cube_grids.at(i).elements_list.at(j);
                                
                                std::array<float, 24> fvertices;
                                std::array<double, 24> vertices = cube->get_vertices();
                                std::transform(vertices.begin(), vertices.end(), fvertices.begin(), [](const double & value)-> float {return static_cast<float>(value); });
                                glBindVertexArray(VAOs[i]);
                                glBindBuffer(GL_ARRAY_BUFFER, VBOs[2*i]);
                                glBufferData(GL_ARRAY_BUFFER, fvertices.size() * sizeof(float), fvertices.data(), GL_DYNAMIC_DRAW);;
                                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (void*)0);
                            }
                        }
                        //glDrawArrays(GL_TRIANGLES, 0 , 12*3);
                        //std::cout << "Updating cube grid "<<i << " \n";
                        if(LOG_SPRING_PERFORMANCE) render_cube_grids[i].update();//(cumulative_spring_evaluation_time, spring_logfile);
                        else render_cube_grids[i].update();
                        mvp = projection * view; //* cube->model;
                        GLenum err;
                        while ((err = glGetError()) != GL_NO_ERROR) {
                            std::cout << "OpenGL ERROR: " << err << std::endl;
                        }
                        //auto & vertices = cubes[i]->get_vertices();
                    // glBindBuffer(GL_ARRAY_BUFFER, VBOs[i]);
                    // glBufferData(GL_ARRAY_BUFFER, fvertices.size() * sizeof(double), fvertices.data(), GL_DYNAMIC_DRAW);
                    }
                    if(renderloop_count % draw_every == 0) {
                        glUseProgram(gridShaderProgram);
                        glUniformMatrix4fv(gridMatrixID, 1, GL_FALSE, &mvp[0][0]);
                        glBindVertexArray(gridVAO);
                        glDrawArrays(GL_LINES, 0, grid.size() /3);
                        GLenum err;
                        while ((err = glGetError()) != GL_NO_ERROR) {
                            std::cout << "OpenGL ERROR: " << err << std::endl;
                            throw std::runtime_error("Error in drawing lines");
                        }
                        view = glm::lookAt(
                            cameraPos,                      // Camera location in world space
                            cameraPos + cameraFront,                   // Center of view
                            cameraUp                         // 0efines the direction of "up"
                        );
                        glfwPollEvents();
                        glfwSwapBuffers(window);
                    }
                    
                    renderloop_count += 1;
                    t = t + dt;

                }
        }
        generation++;
    }
    

    
    
    glfwTerminate();
    return 0;
}

