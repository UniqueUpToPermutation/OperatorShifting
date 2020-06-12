#include <functional>
#include <iostream>
#include <vector>
#include "testbeds.h"

#define DEFAULT dgnGraphEdgePerturb

int main(int argc, char** argv)
{
    if (argc == 1)
        DEFAULT(0, nullptr);
    else if (argc > 1)
    {
        char* strMethod = argv[1];
        if (strcmp(strMethod, "GridLaplacian1D") == 0)
            dgnGridLaplacian1D(argc - 2, &argv[2]);
        else if (strcmp(strMethod, "GridLaplacian2D") == 0)
            dgnGridLaplacian2D(argc - 2, &argv[2]);
        else if (strcmp(strMethod, "GraphEdgePerturb") == 0)
            dgnGraphEdgePerturb(argc - 2, &argv[2]);
        else if (strcmp(strMethod, "MeshLaplacian") == 0)
            dgnMeshLaplacian(argc - 2, &argv[2]);
        else if (strcmp(strMethod, "GraphEdgeDrop") == 0)
            dgnGraphEdgeDrop(argc - 2, &argv[2]);
        else
            DEFAULT(argc - 2, &argv[2]);
    }
}