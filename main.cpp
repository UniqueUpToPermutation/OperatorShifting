#include <functional>
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>

#include "testbeds.h"

#define DEFAULT dgnGraphEdgePerturb
#define CONFIG_SRC "config.json"

nlohmann::json globalConfig;
bool globalConfigAvailable;

int main(int argc, char** argv)
{
    // Load configuration if possible
    std::ifstream f(CONFIG_SRC);
    if (f.is_open()) {
        globalConfigAvailable = true;
        f >> globalConfig;
        f.close();
    }

    std::string strMethod;

    if (argc == 1) {
        if (globalConfigAvailable)
            strMethod = globalConfig["default"].get<std::string>();
        else
            DEFAULT(0, nullptr);
    }
    else
        strMethod = argv[1];

    if (argc > 1)
    {
        if (strMethod == "GridLaplacian1D")
            dgnGridLaplacian1D(argc - 2, &argv[2]);
        else if (strMethod == "GridLaplacian2D")
            dgnGridLaplacian2D(argc - 2, &argv[2]);
        else if (strMethod == "GraphEdgePerturb")
            dgnGraphEdgePerturb(argc - 2, &argv[2]);
        else if (strMethod == "MeshLaplacian")
            dgnMeshLaplacian(argc - 2, &argv[2]);
        else if (strMethod == "GraphEdgeDrop")
            dgnGraphEdgeDrop(argc - 2, &argv[2]);
        else
            DEFAULT(argc - 2, &argv[2]);
    }
}