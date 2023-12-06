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
std::vector<glm::vec3> starting_vertices;

//Simulation Parameters
const double STARTING_TRANSLATION_X = 0.0;
const double STARTING_TRANSLATION_Y = 0.0;
const double STARTING_TRANSLATION_Z = 2.0; //height the cube is dropped from

const double STARTING_ROT_X         = 30.0; //Starting counter-clockwise rotation about the x-axis in degrees
const double STARTING_ROT_Y         = 0.0;
const double STARTING_ROT_Z         = 0.0;

const float CAMERA_START_X = -5.0;
const float CAMERA_START_Y = -5.0;
const float CAMERA_START_Z =  4.0;

const int GLFW_WINDOW_HEIGHT = 800;
const int GLFW_WINDOW_WIDTH  = 600;



//Debug flags
const bool DEBUG                    = false;
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
    glm::dvec3 add_restoration_force(glm::dvec3 &centroid) { 
        double frc = (mass/0.000000375)*G; //-10000.0f; (was 0.0005 in case of emergency)
        glm::dvec3 direction = glm::dvec3(0,0,1);//glm::normalize(centroid-position);
        glm::dvec3 tangential_component = velocity - glm::dot(velocity, direction)*direction;
        if(position.z < ground) {
            if(DEBUG || DEBUG_RESTORATION) {
                std::cout << "Position: (" << position.x << ", " << position.y << ", " << position.z << ")\n";
                std::cout << "Gravity restitution: ";
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
    glm::dvec3 update(glm::dvec3 &centroid) {
        if (DEBUG) std::cout << "Adding Gravitational Force!\n";
        add_gravitational_force();
        if (DEBUG) pprint(external_forces);
        if (DEBUG) std::cout << "Adding Restoration Force!\n";
        glm::dvec3 rforce = add_restoration_force(centroid);
        if (DEBUG) pprint(external_forces);
        if (DEBUG) std::cout << "Mass: " << mass << std::endl;
        acceleration = external_forces/(mass*10); //10 conversion factor to screen units, acceleration now in screen units per second squared
        if (DEBUG) std::cout << "Acceleration!: " << std::endl;
        if (DEBUG) pprint(acceleration);
        velocity    += acceleration*dt;
        velocity    *= DF;
        if (DEBUG) std::cout << "Velocity!: " << std::endl;
        if (DEBUG) pprint(velocity);
        position    += velocity*dt;
        if (DEBUG) std::cout << "Position!: " << std::endl;
        if (DEBUG) pprint(position);
        external_forces = glm::vec3(0,0,0);
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
         mass2(m2), rest_length(rl), spring_constant(sc), a(a), b(b), c(c), d(d), a0(a), b0(b), c0(c), d0(d), rest_length0(rl) {};

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
        return std::make_shared<Spring>(m1, m2, rl_real, 1*sc, rl_real, 1, 1, 3.1416);
    }

    void set_parameters_default() {
        a = a0;
        b = b0;
        c = c0;
        d = d0;
        rest_length = rest_length0;
    }

    void set_paramaters(double a, double b, double c, double d) {
        a = a;
        b = b;
        c = c;
        d = d;
    }
    void mutate_paramaters(double pa, double pb, double pc, double pd) {
        a += pa;
        b += pb;
        c += pc;
        d += pd;

        if (a < 0) a = -a;
        if (b < 0) b = -b;
        if (c < 1e-6) c = 1e-6;
        d = std::min(std::max(d, 0.0), 2*3.1416);
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
    Mass::create(),Mass::create(),Mass::create()}), model(glm::mat4(1.0f)), normals{} {
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
        parameters.resize(28 * 4);
        cube_perturbations.resize(28 * 4);
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
    void restore_default_parameters() {
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
        set_parameters_default();
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
    void set_parameters_default() {
        /* Set the cube's parameters to its springs parameters, which initialize in a cube shape*/
        for(int i = 0; i < springs.size(); i++) {
            std::vector<double> spring_params = springs.at(i)->get_parameters();
            parameters.at(4*i + 0) = spring_params.at(0);
            parameters.at(4*i + 1) = spring_params.at(1);
            parameters.at(4*i + 2) = spring_params.at(2);
            parameters.at(4*i + 3) = spring_params.at(3);
            //std::cout << "Spring params: " << spring_params.at(0) << " " << spring_params.at(1)<<  " " << spring_params.at(2) << "\n";
            //std::cout << "Cube params: " << parameters.at(4*i + 0) << " " << parameters.at(4*i + 1)<<  " " <<parameters.at(4*i +2) << "\n";
        }
    // for(double i : parameters) {
    //     std::cout << i << " ";
    // }
    // std::cout << "\n";
    }
    void set_parameters(std::pair<std::vector<double>::iterator, std::vector<double>::iterator> &params) {
        int i = 0;
        //std::cout << "parameters size: " << params.second - params.first << std::endl;
        for(std::vector<double>::iterator it = params.first; it != params.second; it++) {
            parameters.at(i++) = *(it);
        }
    }
    std::vector<double> get_parameters() {
        std::vector<double> accumulated_perturbations;
        for(int i = 0; i < cube_perturbations.size(); i++) {
            accumulated_perturbations.push_back(parameters.at(i) - p0.at(i));
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
        int spring_parameters = 4;
        int i = 0;
        //std::cout << perturbations.second - perturbations.first << std::endl;
        for(std::vector<double>::iterator it = perturbations.first; it != perturbations.second; it++) {
            double a = *(it++);//perturbations.at(i/spring_parameters);
            double b = *(it++);//perturbations.at(i/spring_parameters + 1);
            double c = *(it++);//perturbations.at(i/spring_parameters + 2);
            double d = *(it);//perturbations.at(i/spring_parameters + 3);
            if (parameters.at(4*i + 3) + d > 2*3.1416 || parameters.at(4*i + 3) + d < 0) d = 0;
            //std::cout << "i: " << 4*i << ", " << 4*i+1 << ", " << 4*i+2 << ", " << 4*i+3 << std::endl;

            //std::cout << "a, b, c, d: " << a << ", " << b << ", " << c << ", " << d << std::endl;
            springs.at(i)->mutate_paramaters(a, b, c, d);
    
            parameters.at(4*i) += a;
            parameters.at(4*i + 1) += b;
            parameters.at(4*i + 2) += c;
            parameters.at(4*i + 3) += d;
            cube_perturbations.at(4*i) = a;
            cube_perturbations.at(4*i + 1) = b;
            cube_perturbations.at(4*i + 2) = c;
            cube_perturbations.at(4*i + 3) = d;
            if (parameters.at(4*i) < 0) parameters.at(4*i) *= -1;
            if (parameters.at(4*i + 1) < 0) parameters.at(4*i +1) *= -1;
            if (parameters.at(4*i + 2) < 1e-6) parameters.at(4*i + 2) = 1e-6;
            
            i++;
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
        vertices[i] = val.x;
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
    glm::dvec3 centroid;
    std::vector<std::vector<std::vector<std::shared_ptr<Cube>>>> elements;
    std::vector<std::shared_ptr<Cube>> elements_list;
    std::vector<std::shared_ptr<Spring>> connections;
    std::vector<double> parameters;
    
    Grid(int h, int l, int w, std::vector<double> &p) : height{h}, length{l}, width{w}, parameters{p}{
        //std::cout << "start of grid constructor\n";
        elements.resize(height);
        for(auto &slice : elements) {
            slice.resize(length);
            for(auto &row : slice) {
                row.resize(width);
            }
        }
        //std::cout << "end of grid constructor\n";
    }

    void make_dynamic() {
        for(auto &cube : elements_list) {
            cube-> make_dynamic();
        }
    }

    void restore_defaults() {
        for(int i = 0; i < elements_list.size(); i++) {
            elements_list.at(i)->restore_default_parameters();
        }
        set_parameters_default();

    }

    void set_parameters_default() {
        /* Sets cube's parameters equal to its springs. Use restore_defaults to restore springs to default parameters*/
        for(int i = 0; i < elements_list.size(); i++) {
            elements_list.at(i)->set_parameters_default();
        }
    }

    // void set_parameters() {
    //     int num_params = 28*4;
    //     for(int i = 0; i < elements_list.size(); i++) {
    //         std::pair<std::vector<double>::iterator, 
    //         std::vector<double>::iterator> a {parameters.begin() + i*num_params, parameters.begin() + (i+1) * num_params};
    //         //std::cout << "setting parameters\n";
    //         //elements_list.at(i)->set_parameters(a);
    //     }
    // }

    void fix_parameters() {
        for(int i = 0; i < parameters.size()/4; i++) {
            if (parameters.at(4*i) < 0) parameters.at(4*i) *= -1;
            if (parameters.at(4*i + 1) < 0) parameters.at(4*i +1) *= -1;
            if (parameters.at(4*i + 2) < 1e-6) parameters.at(4*i + 2) = 1e-6;
        }
    }


    void generate_perturbations(std::vector<double> &p) {
        int i = 0;
        int num_params = 28*4;
        //for(int i = 0; i < p.size(); i++) {std::cout << p.at(i) << " ";}
        //std::cout << "\nSize: " << p.size() << std::endl;
        parameters = p;
        for( auto &cube : elements_list) {
            std::pair<std::vector<double>::iterator, std::vector<double>::iterator> pi {p.begin() + i*num_params, p.begin() + (i+1) * num_params};
            //cube->set_parameters(pi);
            //std::cout << "Mutate\n";
            cube->set_parameters_default();
            cube->mutate(pi);
            //std::cout << "Done mutating\n";
            i++;
        }
    }

    std::vector<double> get_parameters() {
        std::vector<double> return_vec;
        return_vec.resize(parameters.size());
        for(int i = 0; i < elements_list.size(); i++) {
            std::vector<double> cube_params = elements_list.at(i)->get_parameters();
            for(int j = 0; j < cube_params.size(); j++) {
                //std::cout << "cube params: " << cube_params.at(j) << std::endl;
                return_vec.at(28*4*i + j) = cube_params.at(j);
            }
        }
        return return_vec;
    }
    void add(std::shared_ptr<Cube> c, int row, int column, int elevation) {
        double connection_strength = 3e7;
        if (row < 0 || row > length) throw std::runtime_error("Vector bounds exceeded for 'row' in Grid.add method");
        if (column < 0 || column > width) throw std::runtime_error("Vector bounds exceeded for 'column' in Grid.add method");
        if (elevation < 0 || elevation > height) throw std::runtime_error("Vector bounds exceeded for 'height' in Grid.add method");

        //std::cout << "Attempting to add cube at " << elevation << " " << row << " " << column << std::endl;
        elements_list.push_back(c);

        //Grid indexed by elevation, row, column
        elements.at(elevation).at(row).at(column) = c;
        //std::cout << "Added Cube to position: " << elevation << " " << row << " " << column << std::endl;
        //std::cout << "Checking Upstairs Neighbor\n";
        if (elevation < height-1 && elements.at(elevation+1).at(row).at(column) != nullptr) {
            std::shared_ptr<Cube> above = elements.at(elevation+1).at(row).at(column);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(2*i + 1), above->masses.at(2*i), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
        }
        //std::cout << "Checking Downstairs Neighbor\n";
        if (elevation > 0 && elements.at(elevation-1).at(row).at(column) != nullptr) {
            std::shared_ptr<Cube> below = elements.at(elevation-1).at(row).at(column);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(2*i), below->masses.at(2*i + 1), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
        }
        //std::cout << "Checking left Neighbor\n";
        if (column > 0 && elements.at(elevation).at(row).at(column-1) != nullptr) {
            std::shared_ptr<Cube> left = elements.at(elevation).at(row).at(column-1);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(i), left->masses.at(i+4), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
        }
        //std::cout << "Checking right Neighbor\n";

        if (column < length - 1 && elements.at(elevation).at(row).at(column+1) != nullptr) {
            std::shared_ptr<Cube> right = elements.at(elevation).at(row).at(column+1);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(i+4), right->masses.at(i), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
        }
        //std::cout << "Checking rear Neighbor (" << elevation << " " << row-1 << " " << column << ")\n";
        if (row > 0 && elements.at(elevation).at(row-1).at(column) != nullptr) {
            std::shared_ptr<Cube> behind = elements.at(elevation).at(row-1).at(column);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(2*i + 2 - (i%2)), behind->masses.at(2*i-(i%2)), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
        }
        //std::cout << "Checking front Neighbor\n";
        if (row < width - 1 && elements.at(elevation).at(row+1).at(column) != nullptr) {
            std::shared_ptr<Cube> front = elements.at(elevation).at(row+1).at(column);
            for(int i = 0; i < 4; i++) {
                std::shared_ptr<Spring> ns = Spring::create(c->masses.at(2*i-(i%2)), front->masses.at(2*i + 2 - (i%2)), 0.1, connection_strength);
                ns->set_resting_length(0);
                connections.push_back(ns);
            }
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

class RNG {
    public:
    std::vector<double> mu;
    std::vector<double> sigma;
    int num_params;
    std::default_random_engine generator;
    

    RNG(double m, double s, int n) : mu{m}, sigma{s}, num_params{n}, generator{} {
            
    };

    double snrv() {
        std::normal_distribution<double> z{0, 1};
        return z(generator);
    }
    std::vector<double> sample() {
        std::vector<double> p;
        for(int i = 0; i < num_params; i++) {
            std::normal_distribution<double> d {mu.at(i), sigma.at(i)};
            p.push_back(d(generator));
        }
        return p;
    }


    void set_params(std::vector<double> &m, std::vector<double> &s) {
        mu = m;
        sigma = s;
    }
};

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
    std::vector<double> initial_parameters;
    for(int i = 0; i < 28 * 2; i++) {
        initial_parameters.push_back(0.1);
        initial_parameters.push_back(0.01);
        initial_parameters.push_back(1.0);
        initial_parameters.push_back(0.0);
    }
    Grid g = Grid(3, 3, 3, initial_parameters);
    if (DEBUG) std::cout << "Before creating the cube!" << std::endl;
    std::shared_ptr<Cube> firstcube  {new Cube};
    std::shared_ptr<Cube> secondcube {new Cube};
    if (DEBUG) std::cout << "AFTER creating the cubes!" << std::endl;
    //std::cout << "Initializing cube 1's parameters" << std::endl;
    firstcube->_init_cube_location(cube1_params);
    //std::cout << "Initializing cube 2's parameters" << std::endl;
    secondcube->_init_cube_location(cube2_params);
    g.add(firstcube, 0, 0, 0);
    //std::cout << "Adding Second Cube\n";
    g.add(secondcube, 1, 0, 0);
    g.make_dynamic();
    g.set_parameters_default();
    initial_parameters = g.get_parameters();
    std::vector<double> initial_means;
    std::vector<double> initial_std;
    for(int i = 0; i < initial_parameters.size(); i++) {
        //std::cout << "Initial Parameter: " << initial_parameters.at(i);
        initial_std.push_back((initial_parameters.at(i) + 0.1)/10);
        //std::cout << "Std: " << initial_std.at(i) << std::endl;
        initial_means.push_back(0.0);
    }
    rng.set_params(initial_means, initial_std);
    // std::vector<double> perturbation_test1 = rng.sample();
    // std::vector<double> perturbation_test2 = rng.sample();
    // g.generate_perturbations(perturbation_test1);

    // std::cout << std::fixed << std::setprecision(2);
    // std::cout << "\nPerturbations Test 1: " << std::endl;
    // for(int i = 0; i < initial_parameters.size()/10; i++) {
    //     std::cout << perturbation_test1.at(i) << " ";
    // }
    // std::cout << "\nPerturbations Test 2: " << std::endl;
    // for(int i = 0; i < initial_parameters.size()/10; i++) {
    //     std::cout << perturbation_test2.at(i) << " ";
    // }
    // std::cout << "\ng After Test 1: " << std::endl;
    // std::vector<double> test_params = g.get_parameters();
    // for(int i = 0; i < initial_parameters.size()/10; i++) {
    //     std::cout << test_params.at(i) << " ";
    // }

    

    // std::cout << "\ng After Test 1 close to perturbations 1: " << std::endl;
    // test_params = g.get_parameters();
    // for(int i = 0; i < initial_parameters.size()/10; i++) {
    //     std::cout << (std::fabs(test_params.at(i) - perturbation_test1.at(i)) < 1e-7) << " ";
    // }

    // std::vector<double> sum_perturbs;
    // for(int i = 0; i < initial_parameters.size(); i++) {
    //     sum_perturbs.push_back(test_params.at(i) + perturbation_test2.at(i));
    // }
    
    // g.generate_perturbations(perturbation_test2);
    // std::cout << "\ng After Test 2: " << std::endl;
    // test_params = g.get_parameters();
    // for(int i = 0; i < initial_parameters.size()/10; i++) {
    //     std::cout << test_params.at(i) << " ";
    // }

    // std::cout << "\ng After Test 2 close to perturbation 1 + perturbation 2: " << std::endl;
    // test_params = g.get_parameters();
    // for(int i = 0; i < initial_parameters.size()/10; i++) {
    //     std::cout << (std::fabs(test_params.at(i) - (perturbation_test1.at(i) + perturbation_test2.at(i))) < 1e-7) << " ";
    // }

    // std::cout << "\ng After Test 2 close to perturbation 2: " << std::endl;
    // test_params = g.get_parameters();
    // for(int i = 0; i < initial_parameters.size()/10; i++) {
    //     std::cout << (std::fabs(test_params.at(i) - (perturbation_test2.at(i))) < 1e-7) << " ";
    // }

    // std::cout << "\ng After Test 2 close to perturbation 1: " << std::endl;
    // test_params = g.get_parameters();
    // for(int i = 0; i < initial_parameters.size()/10; i++) {
    //     std::cout << (std::fabs(test_params.at(i) - (perturbation_test1.at(i))) < 1e-7) << " ";
    // }

    // std::cout << "\ng Equivalent after restoring defaults and passing sum of perturbations" << std::endl;
    // g.restore_defaults();
    // g.generate_perturbations(sum_perturbs);
    // test_params = g.get_parameters();
    // for(int i = 0; i < initial_parameters.size()/10; i++) {
    //     std::cout << (std::fabs(test_params.at(i) - (perturbation_test1.at(i) + perturbation_test2.at(i))) < 1e-7) << " ";
    // }


    // return 1;

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
    Grid ng = Grid(3,3,3,initial_parameters);
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
        population.push_back(initial_means);

    }
    bool render = false;
    std::vector<double> best_params;
    std::vector<std::vector<double>> all_fitnesses;
    double best_fitness = -1e9;
    int max_generations = 1000000;
    int render_every_n_generations = 1000;
    int log_every_n_generations = 100;
    std::vector<double> fitnesses;
    std::vector<double> best_fitnesses;
    std::vector<double> old_mean (28 * 4 * 2);
    std::vector<double> best_average_velocities;
    std::vector<double> old_std = initial_std;
    while(generation < max_generations) {
        render = false;
        
        //if (generation % render_every_n_generations == 0) render = true;
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
            for(int j = 0; j < 5 + (5 - (i/7)); j++) { //BRING THIS BACK AFTER DEBUGGING
                if(generation == 0) fitnesses.push_back(-1e9);
                std::vector<double> perturbations = rng.sample();
                //std::cout << "got rng sample...\n";
                //.generate_perturbations(population_parameters.at(i+j));
                std::vector<double> new_params;
                for(int k = 0; k < parent_params.size(); k++) {
                    new_params.push_back(parent_params.at(k) + perturbations.at(k));
                }
                
                offspring.push_back(new_params);
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
            ng.generate_perturbations(offspring.at(i));
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
        std::vector<double> mean (28*4*2);
        std::vector<double> stdevs (28*4*2);
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
        Standard deviations should all be positive and not too close to zero or extremely large
        */
        std::cout << "Standard deviations\n";
        for(int i = 0; i < 28*4*2; i++) {
            stdevs.at(i) = std::exp((1/(double)std::sqrt(2*2*28*4))*rng.snrv())*old_std.at(i)*std::exp((1/(double)std::sqrt(2*std::sqrt(2*28*4)))*rng.snrv());
            std::cout << stdevs.at(i) << "\t";
        }
        std::cout << "\n";
        rng.set_params(initial_means, stdevs);
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
            g.restore_defaults();
            g.generate_perturbations(population.at(0));
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

