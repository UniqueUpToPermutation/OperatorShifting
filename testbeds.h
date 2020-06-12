#ifndef OPERATORAUGMENTATION_TESTBEDS_H
#define OPERATORAUGMENTATION_TESTBEDS_H

enum PerturbType {
    PERTURB_TYPE_DISCRETE,
    PERTURB_TYPE_GAMMA,
};

void dgnGridLaplacian1D(int argc, char** argv);
void dgnGridLaplacian2D(int argc, char** argv);
void dgnMeshLaplacian(int argc, char** argv);
void dgnGraphEdgePerturb(int argc, char** argv);
void dgnGraphEdgeDrop(int argc, char** argv);

#endif //OPERATORAUGMENTATION_TESTBEDS_H
