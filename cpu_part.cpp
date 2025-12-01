#include <iostream>
#include <cmath>
using namespace std;

double gravity = 0; 

struct Point3D {
    double x;
    double y;
    double z;
};

class Shape {
protected:
    Point3D massCenter;
    double height;
    double length;
    double width;
    Point3D velocity;

public:
    Shape(Point3D c, double h, double L, double W, Point3D v)
        : massCenter(c), height(h), length(L), width(W), velocity(v) {}

    virtual double distance(const Point3D& point) const = 0;

    virtual void printPosition() const {
        cout << "Center: (" << massCenter.x << ", "
             << massCenter.y << ", "
             << massCenter.z << ")\n";
    }

    void translate(double dx, double dy, double dz) {
        massCenter.x += dx;
        massCenter.y += dy;
        massCenter.z += dz;
    }

    virtual void update(double dt) {
        velocity.y += gravity * dt;
        massCenter.x += velocity.x * dt;
        massCenter.y += velocity.y * dt;
        massCenter.z += velocity.z * dt;
    }

    Point3D getCenter() const { 
        return massCenter; 
    }
    double getLength() const { 
        return length;
    }
    double getHeight() const { 
        return height; 
    }
    double getWidth()  const { 
        return width; 
    }

    virtual ~Shape() {}
};

//////Distances 
double L1(const Point3D& a, const Point3D& b) {
    return std::abs(a.x - b.x) +
           std::abs(a.y - b.y) +
           std::abs(a.z - b.z);
}


double L2(const Point3D& a, const Point3D& b) {
    return std::sqrt(
        std::pow(a.x - b.x, 2) +
        std::pow(a.y - b.y, 2) +
        std::pow(a.z - b.z, 2)
    );
}

//////Sphere
class Sphere : public Shape {
private:
    double radius;
public:
    Sphere(Point3D c, double diameter, Point3D v)
        : Shape(c, diameter, diameter, diameter, v), radius(diameter/2) {}

    double getRadius() const { 
        return radius; 
    }

    double distance(const Point3D& p) const override {
        return L2(getCenter(), p);
    }
    
    void update(double dt) override {
        Shape::update(dt);
    }
};


///////Rigid bodies
class RigidBody : public Shape {
protected:
    Point3D angle;
    Point3D angularVelocity;
    Point3D axes[3]; // local axes

public:
    RigidBody(Point3D c, double h, double L, double W, Point3D v, Point3D a, Point3D av)
        : Shape(c, h, L, W, v), angle(a), angularVelocity(av) {
        axes[0] = {1,0,0};
        axes[1] = {0,1,0};
        axes[2] = {0,0,1};
        updateAxes();
    }

    void printAngle() const {
        cout << "Angles: (" << angle.x << ", " << angle.y << ", " << angle.z << ")\n";
    }

    double distance(const Point3D& p) const override {
        return L1(getCenter(), p);
    }

    void rotate(double dax, double day, double daz) {
        angle.x += dax;
        angle.y += day;
        angle.z += daz;
    }

    void setAngularVelocity(const Point3D& av) {
        angularVelocity = av;
    }

    void updateAxes() {
        axes[0] = {cos(angle.y)*cos(angle.z), sin(angle.x)*sin(angle.y)*cos(angle.z)-cos(angle.x)*sin(angle.z), cos(angle.x)*sin(angle.y)*cos(angle.z)+sin(angle.x)*sin(angle.z)};
        axes[1] = {cos(angle.y)*sin(angle.z), sin(angle.x)*sin(angle.y)*sin(angle.z)+cos(angle.x)*cos(angle.z), cos(angle.x)*sin(angle.y)*sin(angle.z)-sin(angle.x)*cos(angle.z)};
        axes[2] = {-sin(angle.y), sin(angle.x)*cos(angle.y), cos(angle.x)*cos(angle.y)};
    } 

    virtual void update(double dt) override {
        Shape::update(dt);
        angle.x += angularVelocity.x * dt;
        angle.y += angularVelocity.y * dt;
        angle.z += angularVelocity.z * dt;
        updateAxes();
    }
    
    const Point3D& getAxis(int i) const { return axes[i]; }
};

class Cube : public RigidBody {
public:
    Cube(Point3D c, double side, Point3D v, Point3D a, Point3D av)
        : RigidBody(c, side, side, side, v, a, av) {}

};

class RectangularPrism : public RigidBody {
public:
    RectangularPrism(Point3D c, double h, double L, double W, Point3D v, Point3D a, Point3D av)
        : RigidBody(c, h, L, W, v, a, av) {}

};


///// Collision
//collision between spheres
bool checkSphereCollision(const Sphere& s1, const Sphere& s2) {
    return s1.distance(s2.getCenter()) <= (s1.getRadius() + s2.getRadius());
}

//collision between two Rigid bodies at angle = (0,0,0)  (for the beginning)
bool checkRigidNoAngleCollision(const RigidBody& a, const RigidBody& b) {
    return (abs(a.getCenter().x - b.getCenter().x) <= (a.getLength()/2 + b.getLength()/2)) &&
           (abs(a.getCenter().y - b.getCenter().y) <= (a.getHeight()/2 + b.getHeight()/2)) &&
           (abs(a.getCenter().z - b.getCenter().z) <= (a.getWidth()/2 + b.getWidth()/2));
}

//collision between two Rigid bodies in Oriented Bounding Box 
bool checkOBBCollision(const RigidBody& A, const RigidBody& B) {
    
    //vector AB
    Point3D T = {B.getCenter().x - A.getCenter().x,
                 B.getCenter().y - A.getCenter().y,
                 B.getCenter().z - A.getCenter().z};

    double ra, rb;
    double R[3][3];      // axes for projection 
    double AbsR[3][3];
    double t;

    const double EPSILON = 1e-6;

    // Projection matrix
    for(int i=0; i<3; i++) {
        const Point3D& Ai = A.getAxis(i);
        for(int j=0; j<3; j++) {
            const Point3D& Bj = B.getAxis(j);
            R[i][j] = Ai.x*Bj.x + Ai.y*Bj.y + Ai.z*Bj.z;
            AbsR[i][j] = std::abs(R[i][j]) + EPSILON;
        }
    }

    //Test axes of A
    for(int i=0; i<3; i++){
        const Point3D& Ai = A.getAxis(i);
        // ra = projection of A half-dimensions on its own axis i
        switch(i) {
            case 0: ra = A.getLength() / 2; break;
            case 1: ra = A.getHeight() / 2; break;
            case 2: ra = A.getWidth() / 2; break;
        }
        
        // rb = projection of B half-dimensions on A's axis i
        rb = B.getLength()/2 * AbsR[i][0] + B.getHeight()/2 * AbsR[i][1] + B.getWidth()/2 * AbsR[i][2];

        // t = distance between centers projected on A's axis i
        t = std::abs(T.x * Ai.x + T.y * Ai.y + T.z * Ai.z);
        
        if(t > ra + rb) return false; // Separation found, no collision
    }

    //Test axes of B
    for(int i=0; i<3; i++){
        const Point3D& Bi = B.getAxis(i);

        ra = A.getLength()/2 * AbsR[0][i] + A.getHeight()/2 * AbsR[1][i] + A.getWidth()/2 * AbsR[2][i];
        
        switch(i) {
            case 0: rb = B.getLength() / 2; break;
            case 1: rb = B.getHeight() / 2; break;
            case 2: rb = B.getWidth() / 2; break;
        };

        t = std::abs(T.x * Bi.x + T.y * Bi.y + T.z * Bi.z);
        if(t > ra + rb) return false; // Separation found
    }

    //Test cross products of axes (9 tests)
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            const Point3D& Ai = A.getAxis(i);
            const Point3D& Bj = B.getAxis(j);

            // ra = projection of A on cross product axis
            switch(i) {
                case 0: ra = A.getHeight()/2 * AbsR[1][j] + A.getWidth()/2 * AbsR[2][j]; break;
                case 1: ra = A.getLength()/2 * AbsR[2][j] + A.getWidth()/2 * AbsR[0][j]; break;
                case 2: ra = A.getHeight()/2 * AbsR[0][j] + A.getLength()/2 * AbsR[1][j]; break;
            };

            // rb = projection of B on cross product axis
            switch(i) {
                case 0: rb = B.getHeight()/2 * AbsR[i][1] + B.getWidth()/2 * AbsR[i][2]; break;
                case 1: rb = B.getLength()/2 * AbsR[i][2] + B.getWidth()/2 * AbsR[i][0]; break;
                case 2: rb = B.getHeight()/2 * AbsR[i][0] + B.getLength()/2 * AbsR[i][1]; break;
            };

            // t = distance between centers projected onto cross product axis
            Point3D axis = { 
                Ai.y*Bj.z - Ai.z*Bj.y,
                Ai.z*Bj.x - Ai.x*Bj.z,
                Ai.x*Bj.y - Ai.y*Bj.x
            };

            t = std::abs(T.x * axis.x + T.y * axis.y + T.z * axis.z);
            if(t > ra + rb) return false; // Separation found
        }
    }

    // No separating axis found = collision
    return true;
}

//collision between sphere and moving rigid body
Point3D closestPointOnOBB(const RigidBody& box, const Point3D& p) {
    Point3D d = {p.x - box.getCenter().x, p.y - box.getCenter().y, p.z - box.getCenter().z};
    Point3D q = box.getCenter();

    double halfDims[3] = { box.getLength()/2, box.getHeight()/2, box.getWidth()/2 };

    for(int i=0; i<3; i++){
        const Point3D& axis = box.getAxis(i);
        double dist = d.x*axis.x + d.y*axis.y + d.z*axis.z;
        if(dist > halfDims[i]) dist = halfDims[i];
        if(dist < -halfDims[i]) dist = -halfDims[i];

        q.x += dist*axis.x;
        q.y += dist*axis.y;
        q.z += dist*axis.z;
    }
    return q;
}

bool checkSphereRigidCollision(const Sphere& s, const RigidBody& b) {
    Point3D closest = closestPointOnOBB(b, s.getCenter());
    return s.distance(closest) <= s.getRadius();
}

int main() {
    Sphere s1({0,0,0}, 2, {0,0,0});
    Sphere s2({1,0,0}, 2, {0,0,0});

    if (checkSphereCollision(s1, s2))
        cout << "Collision detected!\n";
    else
        cout << "No collision.\n";

    Cube c1({0,0,0}, 2, {0,0,0}, {0,0,0}, {0,0,0});
    Cube c2({3,0,0}, 2, {0,0,0}, {0,0,0}, {0,0,0});

    if (checkOBBCollision(c1, c2))
        cout << "Cube collision detected!\n";
    else
        cout << "No cube collision.\n";

    return 0;
}
