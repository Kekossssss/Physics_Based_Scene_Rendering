#ifndef CPU_OBJECTS_HPP
#define CPU_OBJECTS_HPP

#include <cmath>
#include <iostream>
#include <vector>

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

extern double gravity; // Global gravity constant (can be set in main)

// ============================================================================
// BASIC STRUCTURES
// ============================================================================

/**
 * 3D Point structure for positions, velocities, and angles
 */
struct Point3D
{
    double x;
    double y;
    double z;

    // Constructors
    Point3D() : x(0), y(0), z(0) {}
    Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    // Operators
    Point3D operator+(const Point3D &p) const
    {
        return Point3D(x + p.x, y + p.y, z + p.z);
    }

    Point3D operator-(const Point3D &p) const
    {
        return Point3D(x - p.x, y - p.y, z - p.z);
    }

    Point3D operator*(double scalar) const
    {
        return Point3D(x * scalar, y * scalar, z * scalar);
    }

    Point3D &operator+=(const Point3D &p)
    {
        x += p.x;
        y += p.y;
        z += p.z;
        return *this;
    }

    Point3D &operator-=(const Point3D &p)
    {
        x -= p.x;
        y -= p.y;
        z -= p.z;
        return *this;
    }

    // Utility functions
    double magnitude() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    double squareMagnitude() const
    {
        return x * x + y * y + z * z;
    }

    Point3D normalized() const
    {
        double mag = magnitude();
        if (mag > 1e-10)
        {
            return Point3D(x / mag, y / mag, z / mag);
        }
        return Point3D(0, 0, 0);
    }

    double dot(const Point3D &p) const
    {
        return x * p.x + y * p.y + z * p.z;
    }

    Point3D cross(const Point3D &p) const
    {
        return Point3D(
            y * p.z - z * p.y,
            z * p.x - x * p.z,
            x * p.y - y * p.x);
    }
};

// ============================================================================
// DISTANCE FUNCTIONS
// ============================================================================

/**
 * L1 (Manhattan) distance between two points
 */
inline double L1(const Point3D &a, const Point3D &b)
{
    return std::abs(a.x - b.x) +
           std::abs(a.y - b.y) +
           std::abs(a.z - b.z);
}

/**
 * L2 (Euclidean) distance between two points
 */
inline double L2(const Point3D &a, const Point3D &b)
{
    return std::sqrt(
        std::pow(a.x - b.x, 2) +
        std::pow(a.y - b.y, 2) +
        std::pow(a.z - b.z, 2));
}

/**
 * Squared L2 distance (faster, no sqrt)
 */
inline double L2Squared(const Point3D &a, const Point3D &b)
{
    return std::pow(a.x - b.x, 2) +
           std::pow(a.y - b.y, 2) +
           std::pow(a.z - b.z, 2);
}

// ============================================================================
// BASE SHAPE CLASS
// ============================================================================

/**
 * Abstract base class for all 3D shapes
 */
class Shape
{
protected:
    Point3D massCenter; // Center of mass position
    double height;      // Height dimension
    double length;      // Length dimension
    double width;       // Width dimension
    Point3D velocity;   // Linear velocity

public:
    // Constructor
    Shape(Point3D c, double h, double L, double W, Point3D v)
        : massCenter(c), height(h), length(L), width(W), velocity(v) {}

    // Virtual destructor
    virtual ~Shape() {}

    // Pure virtual functions (must be implemented by derived classes)
    virtual double distance(const Point3D &point) const = 0;
    virtual int getType() const = 0; // 0=Sphere, 1=Cube, 2=RectangularPrism

    // Virtual functions with default implementations
    virtual void printPosition() const
    {
        std::cout << "Center: (" << massCenter.x << ", "
                  << massCenter.y << ", "
                  << massCenter.z << ")\n";
    }

    virtual void printInfo() const
    {
        std::cout << "Type: " << getType() << "\n";
        printPosition();
        std::cout << "Velocity: (" << velocity.x << ", "
                  << velocity.y << ", "
                  << velocity.z << ")\n";
        std::cout << "Dimensions: L=" << length
                  << " H=" << height
                  << " W=" << width << "\n";
    }

    // Translation
    void translate(double dx, double dy, double dz)
    {
        massCenter.x += dx;
        massCenter.y += dy;
        massCenter.z += dz;
    }

    void translate(const Point3D &delta)
    {
        massCenter += delta;
    }

    // Physics update
    virtual void update(double dt)
    {
        velocity.y += gravity * dt;
        massCenter.x += velocity.x * dt;
        massCenter.y += velocity.y * dt;
        massCenter.z += velocity.z * dt;
    }

    // Getters
    Point3D getCenter() const { return massCenter; }
    double getLength() const { return length; }
    double getHeight() const { return height; }
    double getWidth() const { return width; }
    Point3D getVelocity() const { return velocity; }

    // Setters
    void setCenter(const Point3D &c) { massCenter = c; }
    void setVelocity(const Point3D &v) { velocity = v; }

    // Virtual getters for rotation (default: no rotation)
    virtual void getRotation(Point3D &angle) const
    {
        angle = Point3D(0, 0, 0);
    }

    virtual void getAngularVelocity(Point3D &angVel) const
    {
        angVel = Point3D(0, 0, 0);
    }

    virtual const Point3D *getAxes() const
    {
        return nullptr;
    }
};

// ============================================================================
// SPHERE CLASS
// ============================================================================

/**
 * Sphere shape (no rotation, defined by radius)
 */
class Sphere : public Shape
{
private:
    double radius;

public:
    // Constructor
    Sphere(Point3D c, double diameter, Point3D v)
        : Shape(c, diameter, diameter, diameter, v),
          radius(diameter / 2.0) {}

    // Type identifier
    int getType() const override { return 0; }

    // Distance to point
    double distance(const Point3D &p) const override
    {
        return L2(getCenter(), p);
    }

    // Update (no rotation for sphere)
    void update(double dt) override
    {
        Shape::update(dt);
    }

    // Getters
    double getRadius() const { return radius; }
    double getDiameter() const { return radius * 2.0; }

    // Print info
    void printInfo() const override
    {
        std::cout << "=== SPHERE ===\n";
        Shape::printInfo();
        std::cout << "Radius: " << radius << "\n";
    }
};

// ============================================================================
// RIGID BODY CLASS (Base for rotating objects)
// ============================================================================

/**
 * Rigid body with rotation capabilities
 */
class RigidBody : public Shape
{
protected:
    Point3D angle;           // Euler angles (x, y, z rotations)
    Point3D angularVelocity; // Angular velocity
    Point3D axes[3];         // Local coordinate axes (updated by rotation)

public:
    // Constructor
    RigidBody(Point3D c, double h, double L, double W, Point3D v,
              Point3D a, Point3D av)
        : Shape(c, h, L, W, v), angle(a), angularVelocity(av)
    {
        // Initialize axes to identity
        axes[0] = Point3D(1, 0, 0);
        axes[1] = Point3D(0, 1, 0);
        axes[2] = Point3D(0, 0, 1);
        updateAxes();
    }

    // Distance (using L1 for rigid bodies)
    double distance(const Point3D &p) const override
    {
        return L1(getCenter(), p);
    }

    // Print rotation info
    void printAngle() const
    {
        std::cout << "Angles: (" << angle.x << ", "
                  << angle.y << ", "
                  << angle.z << ")\n";
    }

    void printInfo() const override
    {
        std::cout << "=== RIGID BODY ===\n";
        Shape::printInfo();
        printAngle();
        std::cout << "Angular Velocity: (" << angularVelocity.x << ", "
                  << angularVelocity.y << ", "
                  << angularVelocity.z << ")\n";
    }

    // Rotation methods
    void rotate(double dax, double day, double daz)
    {
        angle.x += dax;
        angle.y += day;
        angle.z += daz;
        updateAxes();
    }

    void rotate(const Point3D &deltaAngle)
    {
        angle += deltaAngle;
        updateAxes();
    }

    void setAngularVelocity(const Point3D &av)
    {
        angularVelocity = av;
    }

    /**
     * Update local axes based on Euler angles (ZYX rotation order)
     * Uses rotation matrices: R = Rz(θz) * Ry(θy) * Rx(θx)
     */
    void updateAxes()
    {
        double cx = std::cos(angle.x);
        double sx = std::sin(angle.x);
        double cy = std::cos(angle.y);
        double sy = std::sin(angle.y);
        double cz = std::cos(angle.z);
        double sz = std::sin(angle.z);

        // X-axis in world coordinates
        axes[0].x = cy * cz;
        axes[0].y = sx * sy * cz - cx * sz;
        axes[0].z = cx * sy * cz + sx * sz;

        // Y-axis in world coordinates
        axes[1].x = cy * sz;
        axes[1].y = sx * sy * sz + cx * cz;
        axes[1].z = cx * sy * sz - sx * cz;

        // Z-axis in world coordinates
        axes[2].x = -sy;
        axes[2].y = sx * cy;
        axes[2].z = cx * cy;
    }

    // Physics update with rotation
    void update(double dt) override
    {
        Shape::update(dt);
        angle.x += angularVelocity.x * dt;
        angle.y += angularVelocity.y * dt;
        angle.z += angularVelocity.z * dt;
        updateAxes();
    }

    // Getters
    const Point3D &getAxis(int i) const { return axes[i]; }
    const Point3D *getAxes() const override { return axes; }
    Point3D getAngle() const { return angle; }
    Point3D getAngularVelocity() const { return angularVelocity; }

    void getRotation(Point3D &a) const override
    {
        a = angle;
    }

    void getAngularVelocity(Point3D &av) const override
    {
        av = angularVelocity;
    }

    // Setters
    void setAngle(const Point3D &a)
    {
        angle = a;
        updateAxes();
    }
};

// ============================================================================
// CUBE CLASS
// ============================================================================

/**
 * Cube (all sides equal)
 */
class Cube : public RigidBody
{
public:
    // Constructor
    Cube(Point3D c, double side, Point3D v, Point3D a, Point3D av)
        : RigidBody(c, side, side, side, v, a, av) {}

    // Type identifier
    int getType() const override { return 1; }

    // Getters
    double getSide() const { return length; }

    // Print info
    void printInfo() const override
    {
        std::cout << "=== CUBE ===\n";
        RigidBody::printInfo();
        std::cout << "Side: " << length << "\n";
    }
};

// ============================================================================
// RECTANGULAR PRISM CLASS
// ============================================================================

/**
 * Rectangular prism (box with different dimensions)
 */
class RectangularPrism : public RigidBody
{
public:
    // Constructor
    RectangularPrism(Point3D c, double h, double L, double W,
                     Point3D v, Point3D a, Point3D av)
        : RigidBody(c, h, L, W, v, a, av) {}

    // Type identifier
    int getType() const override { return 2; }

    // Print info
    void printInfo() const override
    {
        std::cout << "=== RECTANGULAR PRISM ===\n";
        RigidBody::printInfo();
    }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Create a random Point3D within specified ranges
 */
inline Point3D randomPoint3D(double minX, double maxX,
                             double minY, double maxY,
                             double minZ, double maxZ)
{
    return Point3D(
        minX + (double)rand() / RAND_MAX * (maxX - minX),
        minY + (double)rand() / RAND_MAX * (maxY - minY),
        minZ + (double)rand() / RAND_MAX * (maxZ - minZ));
}

/**
 * Create a random double within range
 */
inline double randomDouble(double min, double max)
{
    return min + (double)rand() / RAND_MAX * (max - min);
}

/**
 * Delete all shapes in a vector
 */
inline void deleteShapes(std::vector<Shape *> &shapes)
{
    for (auto shape : shapes)
    {
        delete shape;
    }
    shapes.clear();
}

/**
 * Print all shapes in a vector
 */
inline void printAllShapes(const std::vector<Shape *> &shapes)
{
    std::cout << "\n=== SCENE: " << shapes.size() << " objects ===\n";
    for (size_t i = 0; i < shapes.size(); i++)
    {
        std::cout << "\nObject " << i << ":\n";
        shapes[i]->printInfo();
    }
}

/**
 * Update all shapes in a vector
 */
inline void updateAllShapes(std::vector<Shape *> &shapes, double dt)
{
#pragma omp parallel for
    for (size_t i = 0; i < shapes.size(); i++)
    {
        shapes[i]->update(dt);
    }
}

bool checkRigidNoAngleCollision(const RigidBody &a, const RigidBody &b);

#endif // CPU_OBJECTS_HPP