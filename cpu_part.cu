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

    double getRadius() const { return radius; }

    double distance(const Point3D& p) const override {
        return L2(massCenter, p);
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
        : Shape(c, h, L, W, v), angle(a), angularVelocity(av) {}

    void printAngle() const {
        cout << "Angles: (" << angle.x << ", " << angle.y << ", " << angle.z << ")\n";
    }

    double distance(const Point3D& p) const override {
        return L1(massCenter, p);
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
    return s1.distance(s2.massCenter) <= (s1.getRadius() + s2.getRadius());
}

//collision between two Rigid bodies at angle = (0,0,0)  (for the beginning)
bool checkRigidNoAngleCollision(const RigidBody& a, const RigidBody& b) {
    return (abs(a.massCenter.x - b.massCenter.x) <= (a.length/2 + b.length/2)) &&
           (abs(a.massCenter.y - b.massCenter.y) <= (a.height/2 + b.height/2)) &&
           (abs(a.massCenter.z - b.massCenter.z) <= (a.width/2 + b.width/2));
}

//collision between two Rigid bodies in Oriented Bounding Box 
bool checkOBBCollision(const RigidBody& A, const RigidBody& B) {
    
    //vector AB
    Point3D T = {B.massCenter.x - A.massCenter.x,
                 B.massCenter.y - A.massCenter.y,
                 B.massCenter.z - A.massCenter.z};

    double ra, rb;
    double R[3][3];      // axes for projection 
    double AbsR[3][3];

    const double EPSILON = 1e-6;

    //Matrix of projection
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            R[i][j] = A.axes[i].x*B.axes[j].x +
                      A.axes[i].y*B.axes[j].y +
                      A.axes[i].z*B.axes[j].z;
            AbsR[i][j] = std::abs(R[i][j]) + EPSILON;
        }
    }

    //Test axes of A
    for(int i=0; i<3; i++){
        // ra = projection of A half-dimensions on its own axis i
        switch(i) {
            case 0: ra = A.length / 2; break;
            case 1: ra = A.height / 2; break;
            case 2: ra = A.width / 2; break;
        }
        
        // rb = projection of B half-dimensions on A's axis i
        rb = B.length/2 * AbsR[i][0] + B.height/2 * AbsR[i][1] + B.width/2 * AbsR[i][2];

        // t = distance between centers projected on A's axis i
        t = std::abs(T.x*A.axes[i].x + T.y*A.axes[i].y + T.z*A.axes[i].z);

        if(t > ra + rb) return false; // Separation found, no collision
    }

    //Test axes of B
    for(int i=0; i<3; i++){
        ra = A.length/2 * AbsR[0][i] + A.height/2 * AbsR[1][i] + A.width/2 * AbsR[2][i];
        
        switch(i) {
            case 0: rb = B.length / 2; break;
            case 1: rb = B.height / 2; break;
            case 2: rb = B.width / 2; break;
        };

        t = std::abs(T.x*B.axes[i].x + T.y*B.axes[i].y + T.z*B.axes[i].z);
        if(t > ra + rb) return false; // Separation found
    }

    //Test cross products of axes (9 tests)
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            // ra = projection of A on cross product axis
            switch(i) {
                case 0: ra = A.height/2 * AbsR[1][j] + A.width/2 * AbsR[2][j]; break;
                case 1: ra = A.length/2 * AbsR[2][j] + A.width/2 * AbsR[0][j]; break;
                case 2: ra = A.height/2 * AbsR[0][j] + A.length/2 * AbsR[1][j]; break;
            };

            // rb = projection of B on cross product axis
            switch(i) {
                case 0: rb = B.height/2 * AbsR[i][1] + B.width/2 * AbsR[i][2]; break;
                case 1: rb = B.length/2 * AbsR[i][2] + B.width/2 * AbsR[i][0]; break;
                case 2: rb = B.height/2 * AbsR[i][0] + B.length/2 * AbsR[i][1]; break;
            };

            // t = distance between centers projected onto cross product axis
            Point3D axis = { 
                A.axes[i].y*B.axes[j].z - A.axes[i].z*B.axes[j].y,
                A.axes[i].z*B.axes[j].x - A.axes[i].x*B.axes[j].z,
                A.axes[i].x*B.axes[j].y - A.axes[i].y*B.axes[j].x
            };

            t = std::abs(T.x*axis.x + T.y*axis.y + T.z*axis.z);
            if(t > ra + rb) return false; // Separation found
        }
    }

    // No separating axis found = collision
    return true;
}

//collision between sphere and moving rigid body
Point3D closestPointOnOBB(const RigidBody& box, const Point3D& p) {
    Point3D d = {p.x - box.massCenter.x, p.y - box.massCenter.y, p.z - box.massCenter.z};
    Point3D q = box.massCenter;

    double halfDims[3] = { box.length/2, box.height/2, box.width/2 };

    for(int i=0; i<3; i++){
        double dist = d.x*box.axes[i].x + d.y*box.axes[i].y + d.z*box.axes[i].z;
        if(dist > halfDims[i]) dist = halfDims[i];
        if(dist < -halfDims[i]) dist = -halfDims[i];

        q.x += dist*box.axes[i].x;
        q.y += dist*box.axes[i].y;
        q.z += dist*box.axes[i].z;
    }
    return q;
}

bool checkSphereRigidCollision(const Sphere& s, const RigidBody& b) {
    Point3D closest = closestPointOnOBB(b, s.massCenter);
    return s.distance(closest) <= s.getRadius();
}


//////// example 
int main() {
    double dt = 0.01;
    int N = 1000;

    // Crée les objets
    Shape* s1 = new Cube({0,0,0}, 2, {0,0,0}, {0,0,0}, {0,0,0});
    Shape* s2 = new RectangularPrism({5,0,0}, 2, 3, 1, {0,0,0}, {0,0,0}, {0,0,0});

    std::vector<Shape*> objects = {s1, s2};

    for (int i = 0; i < N; i++) {
        for (auto obj : objects) {
            obj->update(dt);
            obj->printPosition();
        }

        // Exemple collision simple pour RigidBody
        RigidBody* r1 = dynamic_cast<RigidBody*>(s1);
        RigidBody* r2 = dynamic_cast<RigidBody*>(s2);
        if(r1 && r2 && checkRigidNoAngleCollision(*r1, *r2)) {
            cout << "Collision Rigid Bodies!\n";
        }

        cout << "----------\n";
    }

    delete s1;
    delete s2;

    return 0;
}

