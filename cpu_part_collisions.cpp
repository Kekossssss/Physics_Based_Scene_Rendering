#include "cpu_part_collisions.hpp"
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <omp.h>

using namespace std;

// ═══════════════════════════════════════════════════════════════
//  CONSTANTES GLOBALES
// ═══════════════════════════════════════════════════════════════

double gravity = 9.81;

// ═══════════════════════════════════════════════════════════════
//  POINT3D - IMPLÉMENTATION
// ═══════════════════════════════════════════════════════════════

Point3D Point3D::operator+(const Point3D& p) const {
    return {x + p.x, y + p.y, z + p.z};
}

Point3D Point3D::operator-(const Point3D& p) const {
    return {x - p.x, y - p.y, z - p.z};
}

Point3D Point3D::operator*(double s) const {
    return {x * s, y * s, z * s};
}

double Point3D::dot(const Point3D& p) const {
    return x * p.x + y * p.y + z * p.z;
}

double Point3D::magnitude() const {
    return sqrt(x*x + y*y + z*z);
}

Point3D Point3D::normalize() const {
    double mag = magnitude();
    return mag > 0 ? (*this) * (1.0/mag) : Point3D{0,0,0};
}

// ═══════════════════════════════════════════════════════════════
//  FONCTIONS DE DISTANCE
// ═══════════════════════════════════════════════════════════════

double L1(const Point3D& a, const Point3D& b) {
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z);
}

double L2(const Point3D& a, const Point3D& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

// ═══════════════════════════════════════════════════════════════
//  SHAPE - IMPLÉMENTATION
// ═══════════════════════════════════════════════════════════════

Shape::Shape(Point3D c, double h, double L, double W, Point3D v, double m, double e)
    : massCenter(c), height(h), length(L), width(W), velocity(v), mass(m), restitution(e) {}

void Shape::update(double dt) {
    velocity.y += gravity * dt;
    massCenter = massCenter + velocity * dt;
}

void Shape::printPosition() const {
    cout << "Center: (" << massCenter.x << ", " << massCenter.y << ", " << massCenter.z << ")\n";
}

void Shape::translate(double dx, double dy, double dz) {
    massCenter.x += dx;
    massCenter.y += dy;
    massCenter.z += dz;
}

void Shape::applyImpulse(const Point3D& impulse) {
    velocity.x += impulse.x / mass;
    velocity.y += impulse.y / mass;
    velocity.z += impulse.z / mass;
}

Point3D Shape::getCenter() const { return massCenter; }
Point3D Shape::getVelocity() const { return velocity; }
double Shape::getMass() const { return mass; }
double Shape::getRestitution() const { return restitution; }
double Shape::getLength() const { return length; }
double Shape::getHeight() const { return height; }
double Shape::getWidth() const { return width; }

void Shape::setVelocity(const Point3D& v) { velocity = v; }
void Shape::setMass(double m) { mass = m; }

Shape::~Shape() {}

// ═══════════════════════════════════════════════════════════════
//  SPHERE - IMPLÉMENTATION
// ═══════════════════════════════════════════════════════════════

Sphere::Sphere(Point3D c, double diameter, Point3D v, double m, double e)
    : Shape(c, diameter, diameter, diameter, v, m, e), radius(diameter/2) {}

double Sphere::getRadius() const { return radius; }

double Sphere::distance(const Point3D& p) const {
    return L2(getCenter(), p);
}

void Sphere::update(double dt) {
    Shape::update(dt);
}

// ═══════════════════════════════════════════════════════════════
//  RIGIDBODY - IMPLÉMENTATION
// ═══════════════════════════════════════════════════════════════

RigidBody::RigidBody(Point3D c, double h, double L, double W, Point3D v, 
                     Point3D a, Point3D av, double m, double e)
    : Shape(c, h, L, W, v, m, e), angle(a), angularVelocity(av) {
    axes[0] = {1,0,0};
    axes[1] = {0,1,0};
    axes[2] = {0,0,1};
    updateAxes();
}

void RigidBody::updateAxes() {
    double cx = cos(angle.x), sx = sin(angle.x);
    double cy = cos(angle.y), sy = sin(angle.y);
    double cz = cos(angle.z), sz = sin(angle.z);
    
    axes[0] = {cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz};
    axes[1] = {cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz};
    axes[2] = {-sy, sx*cy, cx*cy};
}

double RigidBody::distance(const Point3D& p) const {
    return L1(getCenter(), p);
}

void RigidBody::rotate(double dax, double day, double daz) {
    angle.x += dax;
    angle.y += day;
    angle.z += daz;
    updateAxes();
}

void RigidBody::setAngularVelocity(const Point3D& av) {
    angularVelocity = av;
}

void RigidBody::update(double dt) {
    Shape::update(dt);
    angle = angle + angularVelocity * dt;
    updateAxes();
}

const Point3D& RigidBody::getAxis(int i) const { return axes[i]; }
Point3D RigidBody::getAngle() const { return angle; }
Point3D RigidBody::getAngularVelocity() const { return angularVelocity; }

void RigidBody::printAngle() const {
    cout << "Angles: (" << angle.x << ", " << angle.y << ", " << angle.z << ")\n";
}

// ═══════════════════════════════════════════════════════════════
//  CUBE ET RECTANGULAR PRISM
// ═══════════════════════════════════════════════════════════════

Cube::Cube(Point3D c, double side, Point3D v, Point3D a, Point3D av, double m, double e)
    : RigidBody(c, side, side, side, v, a, av, m, e) {}

RectangularPrism::RectangularPrism(Point3D c, double h, double L, double W, 
                                   Point3D v, Point3D a, Point3D av, double m, double e)
    : RigidBody(c, h, L, W, v, a, av, m, e) {}

// ═══════════════════════════════════════════════════════════════
//  DÉTECTION DE COLLISIONS
// ═══════════════════════════════════════════════════════════════

bool checkSphereCollision(const Sphere& s1, const Sphere& s2) {
    return s1.distance(s2.getCenter()) <= (s1.getRadius() + s2.getRadius());
}

bool checkRigidNoAngleCollision(const RigidBody& a, const RigidBody& b) {
    return (abs(a.getCenter().x - b.getCenter().x) <= (a.getLength()/2 + b.getLength()/2)) &&
           (abs(a.getCenter().y - b.getCenter().y) <= (a.getHeight()/2 + b.getHeight()/2)) &&
           (abs(a.getCenter().z - b.getCenter().z) <= (a.getWidth()/2 + b.getWidth()/2));
}

bool checkOBBCollision(const RigidBody& A, const RigidBody& B) {
    Point3D T = B.getCenter() - A.getCenter();
    double ra, rb, t;
    double R[3][3], AbsR[3][3];
    const double EPSILON = 1e-6;

    for(int i=0; i<3; i++) {
        const Point3D& Ai = A.getAxis(i);
        #pragma omp simd
        for(int j=0; j<3; j++) {
            const Point3D& Bj = B.getAxis(j);
            R[i][j] = Ai.dot(Bj);
            AbsR[i][j] = abs(R[i][j]) + EPSILON;
        }
    }

    double halfDimsA[3] = {A.getLength()/2, A.getHeight()/2, A.getWidth()/2};
    double halfDimsB[3] = {B.getLength()/2, B.getHeight()/2, B.getWidth()/2};
    
    for(int i=0; i<3; i++) {
        ra = halfDimsA[i];
        rb = halfDimsB[0]*AbsR[i][0] + halfDimsB[1]*AbsR[i][1] + halfDimsB[2]*AbsR[i][2];
        t = abs(T.dot(A.getAxis(i)));
        if(t > ra + rb) return false;
    }

    for(int i=0; i<3; i++) {
        ra = halfDimsA[0]*AbsR[0][i] + halfDimsA[1]*AbsR[1][i] + halfDimsA[2]*AbsR[2][i];
        rb = halfDimsB[i];
        t = abs(T.dot(B.getAxis(i)));
        if(t > ra + rb) return false;
    }

    for(int i=0; i<3; i++) {
        const Point3D& Ai = A.getAxis(i);
        for(int j=0; j<3; j++) {
            const Point3D& Bj = B.getAxis(j);
            Point3D axis = {Ai.y*Bj.z - Ai.z*Bj.y, Ai.z*Bj.x - Ai.x*Bj.z, Ai.x*Bj.y - Ai.y*Bj.x};
            ra = halfDimsA[(i+1)%3] * AbsR[(i+2)%3][j] + halfDimsA[(i+2)%3] * AbsR[(i+1)%3][j];
            rb = halfDimsB[(j+1)%3] * AbsR[i][(j+2)%3] + halfDimsB[(j+2)%3] * AbsR[i][(j+1)%3];
            t = abs(T.dot(axis));
            if(t > ra + rb) return false;
        }
    }
    return true;
}

Point3D closestPointOnOBB(const RigidBody& box, const Point3D& p) {
    Point3D d = p - box.getCenter();
    Point3D q = box.getCenter();
    double halfDims[3] = {box.getLength()/2, box.getHeight()/2, box.getWidth()/2};

    for(int i=0; i<3; i++) {
        double dist = d.dot(box.getAxis(i));
        dist = max(-halfDims[i], min(halfDims[i], dist));
        q = q + box.getAxis(i) * dist;
    }
    return q;
}

bool checkSphereRigidCollision(const Sphere& s, const RigidBody& b) {
    Point3D closest = closestPointOnOBB(b, s.getCenter());
    return s.distance(closest) <= s.getRadius();
}

// ═══════════════════════════════════════════════════════════════
//  RÉSOLUTION DE COLLISIONS
// ═══════════════════════════════════════════════════════════════

void resolveSphereSphereCollision(Sphere* s1, Sphere* s2) {
    Point3D normal = (s2->getCenter() - s1->getCenter()).normalize();
    Point3D relVel = s2->getVelocity() - s1->getVelocity();
    
    double velAlongNormal = relVel.dot(normal);
    if(velAlongNormal > 0) return;
    
    double e = min(s1->getRestitution(), s2->getRestitution());
    double j = -(1 + e) * velAlongNormal;
    j /= (1/s1->getMass() + 1/s2->getMass());
    
    Point3D impulse = normal * j;
    s1->applyImpulse(impulse * -1.0);
    s2->applyImpulse(impulse);
}

void resolveRigidRigidCollision(RigidBody* r1, RigidBody* r2) {
    // À implémenter
}

void resolveSphereRigidCollision(Sphere* s, RigidBody* r) {
    // À implémenter
}

// ═══════════════════════════════════════════════════════════════
//  FONCTIONS UTILITAIRES
// ═══════════════════════════════════════════════════════════════

double randDouble(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

void initRandom() {
    srand(time(0));
}

void logStep(int step, double time, string msg) {
    cout << "[Step " << step << " | t=" << fixed << setprecision(2) << time << "] " << msg << endl;
}

// ═══════════════════════════════════════════════════════════════
//  FONCTIONS DE TEST
// ═══════════════════════════════════════════════════════════════

void testSphereCollisionSequence() {
    cout << "\n--- TEST 1: Collision de Sphères en Mouvement ---\n";
    Sphere s1({-5, 0, 0}, 2.0, {2, 0, 0});
    Sphere s2({5, 0, 0}, 2.0, {-2, 0, 0});

    double dt = 0.1;
    for (int i = 0; i <= 20; i++) {
        s1.update(dt);
        s2.update(dt);
        bool collide = checkSphereCollision(s1, s2);
        if (abs(s1.getCenter().x) < 3.0) {
            cout << "S1: " << s1.getCenter().x << " | S2: " << s2.getCenter().x 
                 << " -> Collision: " << (collide ? "OUI" : "NON") << endl;
        }
    }
}

void testOBBRotation() {
    cout << "\n--- TEST 2: Cubes en Rotation (OBB) ---\n";
    Cube c1({0, 0, 0}, 2.0, {0, 0, 0}, {0, 0, 0}, {0, 0, 0});
    Cube c2({0, 5, 0}, 2.0, {0, -1, 0}, {0, 0, 0}, {0, 0, 1});

    double dt = 0.1;
    for (int i = 0; i < 40; i++) {
        c2.update(dt);
        if (checkOBBCollision(c1, c2)) {
            cout << "!! COLLISION à t=" << i * dt << " | C2 Y=" << c2.getCenter().y << endl;
            break;
        }
    }
}

void testMixedCollision() {
    cout << "\n--- TEST 3: Sphère vs Boîte Rigide ---\n";
    RectangularPrism sol({0, -2, 0}, 1, 10, 10, {0,0,0}, {0,0,0}, {0,0,0});
    Sphere s({0, 5, 0}, 2.0, {0, -2, 0});

    double dt = 0.1;
    for(int i=0; i<30; i++) {
        s.update(dt);
        if (checkSphereRigidCollision(s, sol)) {
            cout << "Impact à t=" << i*dt << " Hauteur=" << s.getCenter().y << endl;
            break;
        }
    }
}

void runFullSimulation(int N, int STEPS, double dt, bool resolveCollisions) {
    vector<Shape*> shapes;
    
    cout << "\n╔════════════════════════════════════════════╗" << endl;
    cout << "║  SIMULATION COMPLÈTE                       ║" << endl;
    cout << "╚════════════════════════════════════════════╝" << endl;
    cout << "Objets: " << N << " | Steps: " << STEPS << " | dt: " << dt << endl;
    
    for(int i=0; i<N; i++) {
        Point3D pos = {randDouble(-50,50), randDouble(0,100), randDouble(-50,50)};
        Point3D vel = {randDouble(-5,5), randDouble(-2,2), randDouble(-5,5)};
        
        if (i % 2 == 0) {
            shapes.push_back(new Sphere(pos, randDouble(1, 3), vel));
        } else {
            Point3D angle = {randDouble(0, 3.14), randDouble(0, 3.14), randDouble(0, 3.14)};
            Point3D angVel = {randDouble(-1, 1), randDouble(-1, 1), randDouble(-1, 1)};
            shapes.push_back(new Cube(pos, randDouble(1, 3), vel, angle, angVel));
        }
    }

    auto start = chrono::high_resolution_clock::now();
    long long collisions = 0;

    for(int step=0; step<STEPS; step++) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            shapes[i]->update(dt);
        }
        
        #pragma omp parallel for schedule(dynamic) reduction(+:collisions)
        for(int i=0; i<N; i++) {
            for(int j=i+1; j<N; j++) {
                Sphere* s1 = dynamic_cast<Sphere*>(shapes[i]);
                Sphere* s2 = dynamic_cast<Sphere*>(shapes[j]);
                
                if(s1 && s2 && checkSphereCollision(*s1, *s2)) {
                    collisions++;
                    if(resolveCollisions) resolveSphereSphereCollision(s1, s2);
                }
            }
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    
    cout << "Temps: " << duration.count() << "s | Collisions: " << collisions << endl;
    
    for(auto s : shapes) delete s;
}

void printSimulationResults(const SimulationResults& results) {
    cout << "\n╔════════════════════════════════════════════╗" << endl;
    cout << "║         RÉSULTATS DE SIMULATION            ║" << endl;
    cout << "╚════════════════════════════════════════════╝" << endl;
    cout << "Temps d'exécution : " << fixed << setprecision(3) << results.executionTime << " s" << endl;
    cout << "Collisions        : " << results.collisionsDetected << endl;
    cout << "Résolutions       : " << results.collisionsResolved << endl;
    cout << "Performance       : " << fixed << setprecision(2) << results.performanceMTestsPerSec << " MTests/s" << endl;
}

// ═══════════════════════════════════════════════════════════════
//  CLASSE COLLISIONSIMULATION
// ═══════════════════════════════════════════════════════════════

CollisionSimulation::CollisionSimulation(double timeStep)
    : dt(timeStep), currentStep(0), totalCollisions(0), totalResolutions(0) {}

void CollisionSimulation::addShape(Shape* shape) {
    shapes.push_back(shape);
}

void CollisionSimulation::clearShapes() {
    for(auto s : shapes) delete s;
    shapes.clear();
}

int CollisionSimulation::getShapeCount() const {
    return shapes.size();
}

void CollisionSimulation::step() {
    int N = shapes.size();
    
    #pragma omp parallel for
    for(int i=0; i<N; i++) {
        shapes[i]->update(dt);
    }
    
    long long stepCollisions = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:stepCollisions)
    for(int i=0; i<N; i++) {
        for(int j=i+1; j<N; j++) {
            Sphere* s1 = dynamic_cast<Sphere*>(shapes[i]);
            Sphere* s2 = dynamic_cast<Sphere*>(shapes[j]);
            
            if(s1 && s2 && checkSphereCollision(*s1, *s2)) {
                stepCollisions++;
            }
        }
    }
    
    totalCollisions += stepCollisions;
    currentStep++;
}

void CollisionSimulation::run(int steps, bool resolveCollisions) {
    for(int i=0; i<steps; i++) {
        step();
    }
}

long long CollisionSimulation::getTotalCollisions() const { return totalCollisions; }
long long CollisionSimulation::getTotalResolutions() const { return totalResolutions; }
int CollisionSimulation::getCurrentStep() const { return currentStep; }

void CollisionSimulation::setGravity(double g) { gravity = g; }
void CollisionSimulation::setTimeStep(double t) { dt = t; }

CollisionSimulation::~CollisionSimulation() {
    clearShapes();
}