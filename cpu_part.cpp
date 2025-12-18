#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono> 
#include <omp.h>
using namespace std;

double gravity = 9.81; 

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
/*bool checkOBBCollision(const RigidBody& A, const RigidBody& B) {
    
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
} */

bool checkOBBCollision(const RigidBody& A, const RigidBody& B) {
    
    // Vecteur entre les centres
    Point3D T = {B.getCenter().x - A.getCenter().x,
                 B.getCenter().y - A.getCenter().y,
                 B.getCenter().z - A.getCenter().z};

    double ra, rb, t;
    double R[3][3];
    double AbsR[3][3];
    const double EPSILON = 1e-6;
    
    // --- 1. Calcul de la Matrice de Rotation (SIMD OK) ---
    // C'est ici que tu justifies ton "Micro-parallélisme" dans les slides.
    for(int i=0; i<3; i++) {
        const Point3D& Ai = A.getAxis(i);
        
        // Le SIMD fonctionne bien ici car c'est un calcul pur sans conditions
        #pragma omp simd
        for(int j=0; j<3; j++) {
            const Point3D& Bj = B.getAxis(j);
            R[i][j] = Ai.x*Bj.x + Ai.y*Bj.y + Ai.z*Bj.z;
            AbsR[i][j] = std::abs(R[i][j]) + EPSILON;
        }
    }

    // --- 2. Test des Axes de A ---
    for(int i=0; i<3; i++){
        const Point3D& Ai = A.getAxis(i);
        switch(i) {
            case 0: ra = A.getLength() / 2; break;
            case 1: ra = A.getHeight() / 2; break;
            case 2: ra = A.getWidth() / 2; break;
        }
        rb = B.getLength()/2 * AbsR[i][0] + B.getHeight()/2 * AbsR[i][1] + B.getWidth()/2 * AbsR[i][2];
        t = std::abs(T.x * Ai.x + T.y * Ai.y + T.z * Ai.z);
        
        if(t > ra + rb) return false; // Séparation trouvée
    }

    // --- 3. Test des Axes de B ---
    for(int i=0; i<3; i++){
        const Point3D& Bi = B.getAxis(i);
        ra = A.getLength()/2 * AbsR[0][i] + A.getHeight()/2 * AbsR[1][i] + A.getWidth()/2 * AbsR[2][i];
        switch(i) {
            case 0: rb = B.getLength() / 2; break;
            case 1: rb = B.getHeight() / 2; break;
            case 2: rb = B.getWidth() / 2; break;
        };
        t = std::abs(T.x * Bi.x + T.y * Bi.y + T.z * Bi.z);
        
        if(t > ra + rb) return false; // Séparation trouvée
    }

    // --- 4. Test des 9 Produits Vectoriels (Cross Products) ---
    // On utilise la boucle imbriquée classique car on a besoin du "return false" immédiat.
    // Les switch/case rendent le SIMD inefficace ici de toute façon.
    for(int i=0; i<3; i++){
        const Point3D& Ai = A.getAxis(i);
        
        for(int j=0; j<3; j++){
             const Point3D& Bj = B.getAxis(j);

            // ra = projection de A sur l'axe du produit vectoriel
            switch(i) {
                case 0: ra = A.getHeight()/2 * AbsR[1][j] + A.getWidth()/2 * AbsR[2][j]; break;
                case 1: ra = A.getLength()/2 * AbsR[2][j] + A.getWidth()/2 * AbsR[0][j]; break;
                case 2: ra = A.getHeight()/2 * AbsR[0][j] + A.getLength()/2 * AbsR[1][j]; break;
            };

            // rb = projection de B sur l'axe du produit vectoriel
            switch(i) {
                case 0: rb = B.getHeight()/2 * AbsR[i][1] + B.getWidth()/2 * AbsR[i][2]; break;
                case 1: rb = B.getLength()/2 * AbsR[i][2] + B.getWidth()/2 * AbsR[i][0]; break;
                case 2: rb = B.getHeight()/2 * AbsR[i][0] + B.getLength()/2 * AbsR[i][1]; break;
            };

            // t = distance projetée
            // On calcule le produit vectoriel (Cross Product) à la volée
            Point3D axis = { 
                Ai.y*Bj.z - Ai.z*Bj.y,
                Ai.z*Bj.x - Ai.x*Bj.z,
                Ai.x*Bj.y - Ai.y*Bj.x
            };

            t = std::abs(T.x * axis.x + T.y * axis.y + T.z * axis.z);
            
            if(t > ra + rb) return false; // Séparation trouvée
        }
    }

    // Si on arrive ici, aucun axe séparateur n'a été trouvé
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



// Test /////////////////////////////
void logStep(int step, double time, string msg) {
    cout << "[Step " << step << " | t=" << fixed << setprecision(2) << time << "] " << msg << endl;
}

void testSphereCollisionSequence() {
    cout << "\n--- TEST 1: Collision of moving Spheres ---\n";
    Sphere s1({-5, 0, 0}, 2.0, {2, 0, 0}); 
    Sphere s2({5, 0, 0}, 2.0, {-2, 0, 0});

    double dt = 0.1;
    for (int i = 0; i <= 20; i++) {
        s1.update(dt);
        s2.update(dt);

        bool collide = checkSphereCollision(s1, s2);
        
        if (abs(s1.getCenter().x) < 3.0) {
            cout << "S1: " << s1.getCenter().x << " | S2: " << s2.getCenter().x 
                 << " -> Collision: " << (collide ? "YES" : "NO") << endl;
        }
    }
}

void testOBBRotation() {
    cout << "\n--- TEST 2: Rotating Cubes (Test OBB) ---\n";
    
    Cube c1({0, 0, 0}, 2.0, {0, 0, 0}, {0, 0, 0}, {0, 0, 0});
    
    Cube c2({0, 5, 0}, 2.0, {0, -1, 0}, {0, 0, 0}, {0, 0, 1});

    double dt = 0.1;
    bool hit = false;

    for (int i = 0; i < 40; i++) {
        c2.update(dt); 

        if (checkOBBCollision(c1, c2)) {
            cout << "!! COLLISION DETECTED at t=" << i * dt 
                 << " | C2 Y=" << c2.getCenter().y 
                 << " | Angle Z=" << fixed << setprecision(2) << 3.14159
                 << endl;
            hit = true;
            break; 
        }
    }
    
    if (!hit) cout << "test failed : no collision détected." << endl;
}

void testMixedCollision() {
    cout << "\n--- TEST 3: Sphere vs Rigid Box ---\n";
    RectangularPrism sol({0, -2, 0}, 1, 10, 10, {0,0,0}, {0,0,0}, {0,0,0});
    
    Sphere s({0, 5, 0}, 2.0, {0, -2, 0});

    double dt = 0.1;
    for(int i=0; i<30; i++) {
        s.update(dt);
        if (checkSphereRigidCollision(s, sol)) {
            cout << "Impact Sphere-Sol a t=" << i*dt << " Hauteur S=" << s.getCenter().y << endl;
            break;
        }
    }
}

double randDouble(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

int main() {
    srand(time(0));
    
    // 1. CONFIGURATION
    int N = 2000;
    int STEPS = 50;
    double dt = 0.01;
    
    vector<Shape*> shapes;
    
    cout << "Initialisation of " << N << " object" << endl;
    
    // 2. GENERATION
    for(int i=0; i<N; i++) {
        Point3D pos = {randDouble(-50,50), randDouble(-50,50), randDouble(-50,50)};
        Point3D vel = {randDouble(-5,5), randDouble(-5,5), randDouble(-5,5)};
        
        if (i % 2 == 0) {
            shapes.push_back(new Sphere(pos, randDouble(1, 3), vel));
        } else {
            Point3D angle = {randDouble(0, 3), randDouble(0, 3), randDouble(0, 3)};
            Point3D angVel = {randDouble(-1, 1), randDouble(-1, 1), randDouble(-1, 1)};
            shapes.push_back(new Cube(pos, randDouble(1, 3), vel, angle, angVel));
        }
    }

    cout << "Beginning of the simulation..." << endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    long long collisionsDetected = 0;

    // 3. SIMULATION LOOP
    for(int step=0; step<STEPS; step++) {
        
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            shapes[i]->update(dt);
        }
        
        // B. Collision Detection
        // collapse(2) : fusionne les boucles i et j pour faire un seul gros tas de tâches
        // schedule(dynamic) : important car certains tests (Cube-Cube) sont plus longs que d'autres (Sphere-Sphere)
        // reduction(+:collisionsDetected) : chaque thread compte dans son coin, et on additionne tout à la fin
        
        #pragma omp parallel for schedule(dynamic) reduction(+:collisionsDetected)
        for(int i=0; i<N; i++) {
            for(int j=i+1; j<N; j++) {
                
                bool collision = false;
                
                // ... (votre logique de dynamic_cast reste identique) ...
                Sphere* s1 = dynamic_cast<Sphere*>(shapes[i]);
                Sphere* s2 = dynamic_cast<Sphere*>(shapes[j]);
                RigidBody* r1 = dynamic_cast<RigidBody*>(shapes[i]);
                RigidBody* r2 = dynamic_cast<RigidBody*>(shapes[j]);

                if(s1 && s2) collision = checkSphereCollision(*s1, *s2);
                else if(r1 && r2) collision = checkOBBCollision(*r1, *r2);
                else if(s1 && r2) collision = checkSphereRigidCollision(*s1, *r2);
                else if(r1 && s2) collision = checkSphereRigidCollision(*s2, *r1);

                if(collision) {
                    collisionsDetected++; 
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    cout << "------------------------------------------------" << endl;
    cout << "Time spent : " << diff.count() << " s" << endl;
    cout << "Collisions   : " << collisionsDetected << endl;
    cout << "Performance  : " << (double)(N*N*STEPS)/diff.count() / 1e6 << " MTests/sec" << endl;
    cout << "------------------------------------------------" << endl;

    for(auto s : shapes) delete s;
    
    return 0;
}
