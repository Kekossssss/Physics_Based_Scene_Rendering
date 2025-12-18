#ifndef COLLISION_SIM_H
#define COLLISION_SIM_H

#include <iostream>
#include <cmath>
#include <vector>

// ═══════════════════════════════════════════════════════════════
//  CONSTANTES GLOBALES
// ═══════════════════════════════════════════════════════════════

extern double gravity;

// ═══════════════════════════════════════════════════════════════
//  STRUCTURE POINT3D
// ═══════════════════════════════════════════════════════════════

struct Point3D {
    double x, y, z;
    
    // Opérateurs
    Point3D operator+(const Point3D& p) const;
    Point3D operator-(const Point3D& p) const;
    Point3D operator*(double s) const;
    
    // Méthodes utilitaires
    double dot(const Point3D& p) const;
    double magnitude() const;
    Point3D normalize() const;
    
    // Constructeur par défaut
    Point3D(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
};

// ═══════════════════════════════════════════════════════════════
//  FONCTIONS DE DISTANCE
// ═══════════════════════════════════════════════════════════════

double L1(const Point3D& a, const Point3D& b);
double L2(const Point3D& a, const Point3D& b);

// ═══════════════════════════════════════════════════════════════
//  CLASSE ABSTRAITE SHAPE
// ═══════════════════════════════════════════════════════════════

class Shape {
protected:
    Point3D massCenter;
    double height, length, width;
    Point3D velocity;
    double mass;
    double restitution;

public:
    // Constructeur
    Shape(Point3D c, double h, double L, double W, Point3D v, 
          double m = 1.0, double e = 0.8);
    
    // Méthode virtuelle pure
    virtual double distance(const Point3D& point) const = 0;
    
    // Méthodes virtuelles
    virtual void update(double dt);
    virtual void printPosition() const;
    
    // Méthodes physiques
    void applyImpulse(const Point3D& impulse);
    void translate(double dx, double dy, double dz);
    
    // Getters
    Point3D getCenter() const;
    Point3D getVelocity() const;
    double getMass() const;
    double getRestitution() const;
    double getLength() const;
    double getHeight() const;
    double getWidth() const;
    
    // Setters
    void setVelocity(const Point3D& v);
    void setMass(double m);
    
    // Destructeur virtuel
    virtual ~Shape();
};

// ═══════════════════════════════════════════════════════════════
//  CLASSE SPHERE
// ═══════════════════════════════════════════════════════════════

class Sphere : public Shape {
private:
    double radius;
    
public:
    // Constructeur
    Sphere(Point3D c, double diameter, Point3D v, 
           double m = 1.0, double e = 0.8);
    
    // Getter
    double getRadius() const;
    
    // Override
    double distance(const Point3D& p) const override;
    void update(double dt) override;
};

// ═══════════════════════════════════════════════════════════════
//  CLASSE RIGIDBODY
// ═══════════════════════════════════════════════════════════════

class RigidBody : public Shape {
protected:
    Point3D angle;
    Point3D angularVelocity;
    Point3D axes[3];
    
    void updateAxes();

public:
    // Constructeur
    RigidBody(Point3D c, double h, double L, double W, Point3D v, 
              Point3D a, Point3D av, double m = 1.0, double e = 0.7);
    
    // Méthodes de rotation
    void rotate(double dax, double day, double daz);
    void setAngularVelocity(const Point3D& av);
    
    // Getters
    const Point3D& getAxis(int i) const;
    Point3D getAngle() const;
    Point3D getAngularVelocity() const;
    
    // Affichage
    void printAngle() const;
    
    // Override
    double distance(const Point3D& p) const override;
    void update(double dt) override;
};

// ═══════════════════════════════════════════════════════════════
//  CLASSES DÉRIVÉES DE RIGIDBODY
// ═══════════════════════════════════════════════════════════════

class Cube : public RigidBody {
public:
    Cube(Point3D c, double side, Point3D v, Point3D a, Point3D av,
         double m = 1.0, double e = 0.7);
};

class RectangularPrism : public RigidBody {
public:
    RectangularPrism(Point3D c, double h, double L, double W, 
                     Point3D v, Point3D a, Point3D av,
                     double m = 1.0, double e = 0.7);
};

// ═══════════════════════════════════════════════════════════════
//  FONCTIONS DE DÉTECTION DE COLLISIONS
// ═══════════════════════════════════════════════════════════════

// Collision entre deux sphères
bool checkSphereCollision(const Sphere& s1, const Sphere& s2);

// Collision entre deux corps rigides (OBB - Oriented Bounding Box)
bool checkOBBCollision(const RigidBody& A, const RigidBody& B);

// Collision entre sphère et corps rigide
bool checkSphereRigidCollision(const Sphere& s, const RigidBody& b);

// Collision entre corps rigides sans rotation (simplifiée)
bool checkRigidNoAngleCollision(const RigidBody& a, const RigidBody& b);

// Fonction auxiliaire pour OBB
Point3D closestPointOnOBB(const RigidBody& box, const Point3D& p);

// ═══════════════════════════════════════════════════════════════
//  FONCTIONS DE RÉSOLUTION DE COLLISIONS
// ═══════════════════════════════════════════════════════════════

// Résolution élastique entre deux sphères
void resolveSphereSphereCollision(Sphere* s1, Sphere* s2);

// Résolution entre corps rigides (à implémenter)
void resolveRigidRigidCollision(RigidBody* r1, RigidBody* r2);

// Résolution entre sphère et corps rigide (à implémenter)
void resolveSphereRigidCollision(Sphere* s, RigidBody* r);

// ═══════════════════════════════════════════════════════════════
//  FONCTIONS UTILITAIRES
// ═══════════════════════════════════════════════════════════════

// Génération de nombres aléatoires
double randDouble(double min, double max);

// Initialisation du générateur aléatoire
void initRandom();

// Logging pour les tests
void logStep(int step, double time, std::string msg);

// ═══════════════════════════════════════════════════════════════
//  FONCTIONS DE TEST
// ═══════════════════════════════════════════════════════════════

// Test de collision entre sphères en mouvement
void testSphereCollisionSequence();

// Test de rotation et collision OBB
void testOBBRotation();

// Test de collision mixte (sphère vs corps rigide)
void testMixedCollision();

// Test complet de la simulation
void runFullSimulation(int N, int STEPS, double dt, bool resolveCollisions = true);

// ═══════════════════════════════════════════════════════════════
//  CLASSE SIMULATION (Optionnelle - pour encapsuler la logique)
// ═══════════════════════════════════════════════════════════════

class CollisionSimulation {
private:
    std::vector<Shape*> shapes;
    double dt;
    int currentStep;
    long long totalCollisions;
    long long totalResolutions;
    
public:
    // Constructeur
    CollisionSimulation(double timeStep = 0.01);
    
    // Gestion des objets
    void addShape(Shape* shape);
    void clearShapes();
    int getShapeCount() const;
    
    // Simulation
    void step();
    void run(int steps, bool resolveCollisions = true);
    
    // Statistiques
    long long getTotalCollisions() const;
    long long getTotalResolutions() const;
    int getCurrentStep() const;
    
    // Configuration
    void setGravity(double g);
    void setTimeStep(double dt);
    
    // Destructeur
    ~CollisionSimulation();
};

// ═══════════════════════════════════════════════════════════════
//  STRUCTURES POUR RÉSULTATS DE SIMULATION
// ═══════════════════════════════════════════════════════════════

struct SimulationResults {
    double executionTime;
    long long collisionsDetected;
    long long collisionsResolved;
    int totalSteps;
    int objectCount;
    double performanceMTestsPerSec;
};

// Fonction pour afficher les résultats
void printSimulationResults(const SimulationResults& results);

#endif // COLLISION_SIM_H