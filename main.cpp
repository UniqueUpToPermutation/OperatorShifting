#include <functional>
#include <iostream>
#include <vector>

void dgnGridLaplacian1D();
void dgnGridLaplacian2D();
void dgnMeshLaplacian();
void dgnGraphEdgePerturb();
void dgnGraphEdgeDrop();

#define DEFAULT dgnGraphEdgePerturb

int main(int argc, char** argv)
{
    if (argc == 1)
        DEFAULT();
    else if (argc > 1)
    {
        char* strMethod = argv[1];
        if (strcmp(strMethod, "GridLaplacian1D") == 0)
            dgnGridLaplacian1D();
        else if (strcmp(strMethod, "GridLaplacian2D") == 0)
            dgnGridLaplacian2D();
        else if (strcmp(strMethod, "GraphEdgePerturb") == 0)
            dgnGraphEdgePerturb();
        else if (strcmp(strMethod, "MeshLaplacian") == 0)
            dgnMeshLaplacian();
        else if (strcmp(strMethod, "GraphEdgeDrop") == 0)
            dgnGraphEdgeDrop();
        else
            DEFAULT();
    }
}