#include "cpu_part.hpp"
#include <iomanip>
#include <omp.h> 

using namespace std;

double GRAVITY = 9.81; 

ostream& operator<<(ostream& os, const Point3D& p) {
    os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
    return os;
}

bool isValid(const Point3D& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}


Shape::Shape(Point3D c, double h, double L, double W, Point3D v, double m, double e, double g)
    : massCenter(c), height(h), length(L), width(W), velocity(v), mass(m), restitution(e), gravity(g) {}

double Shape::distance(const Point3D& point) const {
    return massCenter.distance(point);
}

void Shape::translate(double dx, double dy, double dz) {
    massCenter.x += dx; massCenter.y += dy; massCenter.z += dz;
}

void Shape::update(double dt) {
    velocity.y += gravity * dt;
    massCenter = massCenter + velocity * dt;
}

void Shape::applyImpulse(const Point3D& impulse) {
    if (isValid(impulse) && mass > 0.0001) {
        velocity = velocity + impulse * (1.0 / mass);
    }
}

// --- Sphere ---
Sphere::Sphere(Point3D c, double diameter, Point3D v, double m, double e, double g)
    : Shape(c, diameter, diameter, diameter, v, m, e, g), radius(diameter/2) {}

// --- RigidBody ---
RigidBody::RigidBody(Point3D c, double h, double L, double W, Point3D v, Point3D a, Point3D av, double m, double e, double g)
    : Shape(c, h, L, W, v, m, e, g), angle(a), angularVelocity(av) {
    updateAxes();
}

void RigidBody::updateAxes() {
    double cx = std::cos(angle.x), sx = std::sin(angle.x);
    double cy = std::cos(angle.y), sy = std::sin(angle.y);
    double cz = std::cos(angle.z), sz = std::sin(angle.z);
    
    axes[0] = {cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz};
    axes[1] = {cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz};
    axes[2] = {-sy,   sx*cy,            cx*cy};
}

void RigidBody::update(double dt) {
    Shape::update(dt);
    angle = angle + angularVelocity * dt;
    updateAxes();
}

// --- Cube & Prism ---
Cube::Cube(Point3D c, double side, Point3D v, Point3D a, Point3D av, double m, double e, double g)
    : RigidBody(c, side, side, side, v, a, av, m, e, g) {}

RectangularPrism::RectangularPrism(Point3D c, double h, double L, double W, Point3D v, Point3D a, Point3D av, double m, double e, double g)
    : RigidBody(c, h, L, W, v, a, av, m, e, g) {}


///////// Collision detection

Point3D closestPointOnOBB(const RigidBody& box, const Point3D& p) {
    Point3D d = p - box.getCenter();
    Point3D q = box.getCenter();
    double halfDims[3] = { box.getLength()/2, box.getHeight()/2, box.getWidth()/2 };

    for(int i=0; i<3; i++){
        double dist = d.dot(box.getAxis(i));
        if(dist > halfDims[i]) dist = halfDims[i];
        if(dist < -halfDims[i]) dist = -halfDims[i];
        q = q + box.getAxis(i) * dist;
    }
    return q;
}

bool checkSphereCollision(const Sphere& s1, const Sphere& s2) {
    return s1.distance(s2.getCenter()) <= (s1.getRadius() + s2.getRadius());
}

bool checkSphereRigidCollision(const Sphere& s, const RigidBody& b) {
    Point3D closest = closestPointOnOBB(b, s.getCenter());
    return s.getCenter().distance(closest) <= s.getRadius();
}

bool checkOBBCollision(const RigidBody& A, const RigidBody& B) {
    Point3D T = B.getCenter() - A.getCenter();
    double ra, rb, t;
    double R[3][3], AbsR[3][3];
    const double EPSILON = 1e-6;
    
    for(int i=0; i<3; i++) for(int j=0; j<3; j++) {
        R[i][j] = A.getAxis(i).dot(B.getAxis(j));
        AbsR[i][j] = std::abs(R[i][j]) + EPSILON;
    }

    // Test axes A
    for(int i=0; i<3; i++){
        ra = (i==0?A.getLength():(i==1?A.getHeight():A.getWidth()))/2;
        rb = B.getLength()/2 * AbsR[i][0] + B.getHeight()/2 * AbsR[i][1] + B.getWidth()/2 * AbsR[i][2];
        if(std::abs(T.dot(A.getAxis(i))) > ra + rb) return false;
    }

    // Test axes B
    for(int i=0; i<3; i++){
        ra = A.getLength()/2 * AbsR[0][i] + A.getHeight()/2 * AbsR[1][i] + A.getWidth()/2 * AbsR[2][i];
        rb = (i==0?B.getLength():(i==1?B.getHeight():B.getWidth()))/2;
        if(std::abs(T.dot(B.getAxis(i))) > ra + rb) return false;
    }

    return true; 
}

/////// resolve collision

double getEffectiveRadius(const RigidBody& rb, const Point3D& normal) {
    double r = 0.0;
    r += (rb.getLength() / 2.0) * std::abs(normal.dot(rb.getAxis(0)));
    r += (rb.getHeight() / 2.0) * std::abs(normal.dot(rb.getAxis(1)));
    r += (rb.getWidth()  / 2.0) * std::abs(normal.dot(rb.getAxis(2)));
    return r;
}

void resolveSphereSphereCollision(Sphere* s1, Sphere* s2) {
    Point3D delta = s2->getCenter() - s1->getCenter();
    double dist = delta.length();
    double overlap = (s1->getRadius() + s2->getRadius()) - dist;

    Point3D normal;
    if (dist < 1e-5) normal = {0, -1, 0}; 
    else normal = delta * (1.0 / dist);
    
    if (!isValid(normal)) normal = {0, -1, 0};

    Point3D relVel = s2->getVelocity() - s1->getVelocity();
    if(relVel.dot(normal) > 0) return;

    // CORRECTION: utiliser les masses directement
    double m1 = s1->getMass();
    double m2 = s2->getMass();
    double totalMass = m1 + m2;
    
    double e = std::min(s1->getRestitution(), s2->getRestitution());
    double j = -(1 + e) * relVel.dot(normal) / totalMass;
    
    if (std::isfinite(j)) {
        s1->applyImpulse(normal * -j);
        s2->applyImpulse(normal * j);
    }

    // Position correction avec les bonnes masses
    if(overlap > 0.01) {
        double ratio1 = m2 / totalMass;
        double ratio2 = m1 / totalMass;
        Point3D corr = normal * (overlap * 0.5); // Réduit de 0.8 à 0.5
        if (isValid(corr)) {
            s1->translate(-corr.x * ratio1, -corr.y * ratio1, -corr.z * ratio1);
            s2->translate( corr.x * ratio2,  corr.y * ratio2,  corr.z * ratio2);
        }
    }
}

void resolveRigidRigidCollision(RigidBody* r1, RigidBody* r2) {
    Point3D delta = r2->getCenter() - r1->getCenter();
    double dist = delta.length();
    
    Point3D normal;
    if (dist < 1e-5) normal = {0, -1, 0}; 
    else normal = delta * (1.0 / dist);
    if (!isValid(normal)) normal = {0, -1, 0};

    double r1_ext = getEffectiveRadius(*r1, normal);
    double r2_ext = getEffectiveRadius(*r2, normal);
    double overlap = (r1_ext + r2_ext) - dist;

    Point3D relVel = r2->getVelocity() - r1->getVelocity();
    if(relVel.dot(normal) > 0) return;
    
    // CORRECTION: utiliser les masses directement
    double m1 = r1->getMass();
    double m2 = r2->getMass();
    double totalMass = m1 + m2;
    
    double e = std::min(r1->getRestitution(), r2->getRestitution());
    double j = -(1 + e) * relVel.dot(normal) / totalMass;
    
    if (std::isfinite(j)) {
        r1->applyImpulse(normal * -j);
        r2->applyImpulse(normal * j);
    }

    // Position correction avec les bonnes masses
    if(overlap > 0.01) {
        double ratio1 = m2 / totalMass;
        double ratio2 = m1 / totalMass;
        Point3D corr = normal * (overlap * 0.5);
        if (isValid(corr)) {
            r1->translate(-corr.x * ratio1, -corr.y * ratio1, -corr.z * ratio1);
            r2->translate( corr.x * ratio2,  corr.y * ratio2,  corr.z * ratio2);
        }
    }
}

void resolveSphereRigidCollision(Sphere* s, RigidBody* r) {
    Point3D closest = closestPointOnOBB(*r, s->getCenter());
    Point3D delta = s->getCenter() - closest;
    double dist = delta.length();
    
    double overlap = s->getRadius() - dist;
    Point3D normal;
    
    if(dist < 1e-5) { 
        Point3D dir = s->getCenter() - r->getCenter();
        double centerDist = dir.length();
        if (centerDist < 1e-5) normal = {0, -1, 0};
        else normal = dir * (1.0 / centerDist);
        overlap = s->getRadius();
    } else {
        normal = delta * (1.0 / dist);
    }
    
    if (!isValid(normal)) normal = {0, -1, 0};

    Point3D relVel = s->getVelocity() - r->getVelocity();
    if(relVel.dot(normal) > 0) return;
    
    // CORRECTION: utiliser les masses directement
    double mS = s->getMass();
    double mR = r->getMass();
    double totalMass = mS + mR;
    
    double e = std::min(s->getRestitution(), r->getRestitution());
    double j = -(1 + e) * relVel.dot(normal) / totalMass;
    
    if (std::isfinite(j)) {
        s->applyImpulse(normal * j);
        r->applyImpulse(normal * -j);
    }

    // Position correction avec les bonnes masses
    if(overlap > 0.01) {
        double ratioS = mR / totalMass;
        double ratioR = mS / totalMass;
        Point3D corr = normal * (overlap * 0.5);
        if (isValid(corr)) {
            s->translate( corr.x * ratioS,  corr.y * ratioS,  corr.z * ratioS);
            r->translate(-corr.x * ratioR, -corr.y * ratioR, -corr.z * ratioR);
        }
    }
}


/////// simulation

void simulationStep(vector<Shape*>& shapes, double dt, long long& nbCollisions) {
    int N = shapes.size();
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) shapes[i]->update(dt);

    // Collisions
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Sphere* s1 = dynamic_cast<Sphere*>(shapes[i]);
            Sphere* s2 = dynamic_cast<Sphere*>(shapes[j]);
            RigidBody* r1 = dynamic_cast<RigidBody*>(shapes[i]);
            RigidBody* r2 = dynamic_cast<RigidBody*>(shapes[j]);

            bool hit = false;
            if (s1 && s2) { if (checkSphereCollision(*s1, *s2)) { resolveSphereSphereCollision(s1, s2); hit=true; } }
            else if (r1 && r2) { if (checkOBBCollision(*r1, *r2)) { resolveRigidRigidCollision(r1, r2); hit=true; } }
            else if (s1 && r2) { if (checkSphereRigidCollision(*s1, *r2)) { resolveSphereRigidCollision(s1, r2); hit=true; } }
            else if (r1 && s2) { if (checkSphereRigidCollision(*s2, *r1)) { resolveSphereRigidCollision(s2, r1); hit=true; } }
            if (hit) nbCollisions++;
        }
    }
}
/*
///// test 
int main() {
    srand(time(0));
    
    vector<Shape*> shapes;
    double dt = 0.01;
    int STEPS = 150; // 1.5 seconds

    cout << "physics engine test" << endl;
    cout << "1. Sphere vs Sphere (X=-10)" << endl;
    cout << "2. Cube vs Cube     (X=  0)" << endl;
    cout << "3. Sphere vs Floor  (X= 10)" << endl;
    cout << "4. Slab vs Sphere   (X= 20)" << endl;

    // 1: Sphere vs Sphere (Both moving)
    // Top sphere falling, Bottom sphere rising
    shapes.push_back(new Sphere({-10, -5, 0}, 2.0, {0, 5, 0}, 1.0, 0.8, 9.81));
    shapes.push_back(new Sphere({-10, 5, 0}, 2.0, {0, -5, 0}, 1.0, 0.8, 9.81));

    // 2: Cube vs Cube (Both moving)
    // Top cube falling, Bottom cube rising
    shapes.push_back(new Cube({0, -5, 0}, 4.0, {0, 5, 0}, {0,0,0}, {0,0,0}, 1.0, 0.8, 9.81));
    shapes.push_back(new Cube({0, 5, 0}, 4.0, {0, -5, 0}, {0,0,0}, {0,0,0}, 1.0, 0.8, 9.81));

    // 3: Sphere vs Static Cube (Floor)
    // Sphere falling (G=9.81) on fixed Cube (G=0.0, Mass=Huge)
    shapes.push_back(new Sphere({10, -5, 0}, 2.0, {0, 0, 0}, 1.0, 0.8, 9.81));
    shapes.push_back(new Cube({10, 5, 0}, 4.0, {0, 0, 0}, {0,0,0}, {0,0,0}, 1e10, 0.5, 0.0));

    // 4: Rectangular Prism (Slab) vs Static Sphere
    // Flat slab falling (G=9.81) on fixed Sphere (G=0.0)
    // Testing getEffectiveRadius logic
    shapes.push_back(new RectangularPrism({20, -5, 0}, 1.0, 8.0, 4.0, {0, 0, 0}, {0,0,0}, {0,0,0}, 1.0, 0.5, 9.81));
    shapes.push_back(new Sphere({20, 5, 0}, 4.0, {0, 0, 0}, 1e10, 0.5, 0.0));

    long long totalCollisions = 0;

    for(int step = 0; step < STEPS; step++) {
        simulationStep(shapes, dt, totalCollisions);

        if (step % 5 == 0) {
            cout << "t=" << fixed << setprecision(2) << step*dt << "s | ";
            
            cout << "S-S: " << setw(5) << shapes[0]->getCenter().y << " | ";
            cout << "C-C: " << setw(5) << shapes[2]->getCenter().y << " | ";
            cout << "S-F: " << setw(5) << shapes[4]->getCenter().y << " | ";
            cout << "P-S: " << setw(5) << shapes[6]->getCenter().y << endl;
        }
    }
    
    cout << "Total Collisions: " << totalCollisions << endl;

    for(auto s : shapes) delete s;
    return 0;
}*/