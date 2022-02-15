#pragma once

#include <opshift/json.hpp>

#include <fstream>

#define CONFIG_SRC "config.json"

enum PerturbType {
    PERTURB_TYPE_DISCRETE,
    PERTURB_TYPE_GAMMA,
};

nlohmann::json globalConfig;
bool globalConfigAvailable;

void loadConfig() {
     // Load configuration if possible
    std::ifstream f(CONFIG_SRC);
    if (f.is_open()) {
        globalConfigAvailable = true;
        f >> globalConfig;
        f.close();
    }
}