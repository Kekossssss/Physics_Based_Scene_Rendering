#ifndef CPU_PART_HPP
#define CPU_PART_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

extern double GRAVITY;

/**
 * @brief Simple 3D Vector structure used for position, velocity, and dimensions.
 * Mathematical operators are inlined for performance.
 */
struct Point3D {
    double x, y, z;

    Point3D() : x(0), y(0), z(0) {} 
    Point3D(double x0, double y0, double z0) : x(x0), y(y0), z(z0) {}

    // Operator overloads
    Point3D operator+(const Point3D& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Point3D operator-(const Point3D& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Point3D operator*(double scalar) const { return {x * scalar, y * scalar, z * scalar}; }

    // Dot product
    double dot(const Point3D& other) const { return x * other.x + y * other.y + z * other.z; }

    // Vector magnitude (length)
    double length() const { return std::sqrt(x*x + y*y + z*z); }

    // Returns a normalized version of the vector (unit vector)
    Point3D normalize() {
        double l = length();
        if (l > 1e-8) return {x / l, y / l, z / l};
        return {0, 0, 0};
    }

    // Euclidean distance to another point
    double distance(const Point3D& point) const {
        double dx = point.x - x;
        double dy = point.y - y;
        double dz = point.z - z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

// Stream operator for easy printing: cout << point;
std::ostream& operator<<(std::ostream& os, const Point3D& p);

bool isValid(const Point3D& p);

/**
 * @brief Base class for all physical entities.
 * Handles kinematics (velocity, gravity) and basic properties.
 */
class Shape {
protected:
    Point3D massCenter;
    double height, length, width;
    Point3D velocity;
    double mass;
    double restitution; 
    double gravity; 
public:
    Shape(Point3D c, double h, double L, double W, Point3D v, double m = 500, double e = 0.5, double g = GRAVITY);
    virtual ~Shape() {}

    virtual double distance(const Point3D& point) const;
    
    // Moves the object manually
    void translate(double dx, double dy, double dz);
    
    // Physics update step (integration)
    virtual void update(double dt);
    
    // Apply a force impulse to change velocity
    void applyImpulse(const Point3D& impulse);

    // Getters
    Point3D getCenter() const { return massCenter; }
    Point3D getVelocity() const { return velocity; }
    double getLength() const { return length; }
    double getHeight() const { return height; }
    double getWidth()  const { return width; }
    double getMass() const { return mass; }
    double getRestitution() const { return restitution; }
    double getGravity() const { return gravity; }
};

/**
 * @brief Specialized Shape: Sphere
 */
class Sphere : public Shape {
private:
    double radius;
public:
    Sphere(Point3D c, double diameter, Point3D v, double m = 500, double e = 0.5, double g = GRAVITY);
    double getRadius() const { return radius; }
    double getDiameter() const { return 2*radius; }
};

/**
 * @brief Base class for oriented boxes (Cubes, Prisms).
 * Handles rotation logic (Euler angles and Axes).
 */
class RigidBody : public Shape {
protected:
    Point3D angle;           // Euler angles (x, y, z)
    Point3D angularVelocity;
    Point3D axes[3];         // Local orthonormal axes (Right, Up, Forward)

public:
    RigidBody(Point3D c, double h, double L, double W, Point3D v, Point3D a, Point3D av, double m = 500, double e = 0.5, double g = GRAVITY);
    
    const Point3D& getAxis(int i) const { return axes[i]; }
    Point3D getAngle() const { return angle; }
    Point3D getAngularVelocity() const { return angularVelocity; }
    
    // Recomputes local axes based on current angles
    void updateAxes();
    
    virtual void update(double dt) override;
};

/**
 * @brief Perfect Cube (Length = Width = Height)
 */
class Cube : public RigidBody {
public:
    Cube(Point3D c, double side, Point3D v, Point3D a, Point3D av, double m, double e, double g);
};

/**
 * @brief Rectangular Prism (Wall, Slab, Plank, etc.)
 */
class RectangularPrism : public RigidBody {
public:
    RectangularPrism(Point3D c, double h, double L, double W, Point3D v, Point3D a, Point3D av, double m, double e, double g);
};

//// Collision Detection

// Finds the closest point on an OBB surface to a given point p
Point3D closestPointOnOBB(const RigidBody& box, const Point3D& p);

// Narrow phase checks
bool checkSphereCollision(const Sphere& s1, const Sphere& s2);
bool checkSphereRigidCollision(const Sphere& s, const RigidBody& b);
bool checkOBBCollision(const RigidBody& A, const RigidBody& B);


//// Collision Resolution 

// Calculates effective radius of a box projected onto a normal vector
double getEffectiveRadius(const RigidBody& rb, const Point3D& normal);

// Impulse solvers with positional correction and NaN safety
void resolveSphereSphereCollision(Sphere* s1, Sphere* s2);
void resolveRigidRigidCollision(RigidBody* r1, RigidBody* r2);
void resolveSphereRigidCollision(Sphere* s, RigidBody* r);


// Simulation Loop
void simulationStep(std::vector<Shape*>& shapes, double dt, long long& nbCollisions);
void simulationStepNoParallel(std::vector<Shape*>& shapes, double dt, long long& nbCollisions);

#endif
