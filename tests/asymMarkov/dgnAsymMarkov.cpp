#include <opshift/opshift.h>
#include <opshift/diagnostics.h>
#include <opshift/graphlap.h>
#include <opshift/matutil.h>
#include <opshift/tests.h>

#include <random>
#include <set>
#include <cstring>

#define _USE_MATH_DEFINES
#include <math.h>

#define DEFAULT_THREAD_COUNT 4
#define DEFAULT_NUM_SUB_RUNS_NAIVE 10000
#define DEFAULT_NUM_SUB_RUNS 100
#define DEFAULT_HISTOGRAPH_SAMPLE_COUNT 32
#define SAMPLES_PER_SUB_RUN 100
#define DEFAULT_PRESET 0
#define DEFAULT_GRID_SIZE 8
#define DEFAULT_DISCOUNT_FACTOR 0.9

enum class GraphPreset {
    P1D_1d4_1d4,
    P1D_1d6_2d6,
    P1D_0_1d2,
    P1D_1d8_1d8_1d8_1d8,
    P1D_complete,
    P2D_unif,
    P2D_nonunif,
    P3D_unif,
    P3D_nonunif
};

using namespace opshift;
using namespace lemon;

std::default_random_engine dgnAsymMarkovRnd(
    std::chrono::system_clock::now().time_since_epoch().count());

// Treat the outgoing edges of a node as a histogram
void histogramSample(
    const ListDigraph* graph,
    const ListDigraph::ArcMap<double>* probabilities,
    const int sampleCount,
    ListDigraph::ArcMap<double>* output) {

    std::vector<int> sampleBuckets;
    std::vector<double> bucketProbabilities;
   
    int vertexCount = lemon::countNodes(*graph);

    sampleBuckets.reserve(vertexCount);
    bucketProbabilities.reserve(vertexCount);

    for (ListDigraph::NodeIt node(*graph); node != INVALID; ++node) {
        sampleBuckets.clear();
        bucketProbabilities.clear();

        // Read in probabilities
        for (ListDigraph::OutArcIt e(*graph, node); e != INVALID; ++e) {
            bucketProbabilities.emplace_back(probabilities->operator[](e));
            sampleBuckets.emplace_back(0);
        }

        // Sample histogram from probabilities
        std::discrete_distribution<> dist(
            bucketProbabilities.begin(), 
            bucketProbabilities.end());

        for (int sampleId = 0; sampleId < sampleCount; ++sampleId) {
            sampleBuckets[dist(dgnAsymMarkovRnd)]++;
        }

        // Set new graph weights to histogram
        size_t index = 0;
        for (ListDigraph::OutArcIt e(*graph, node); e != INVALID; ++e, ++index) {
            output->operator[](e) = ((double)sampleBuckets[index] / (double)sampleCount);
        }
    }
}

typedef std::function<std::tuple<double, double, double, double>(int, int)>
    grid2DProbabilityFunc;
typedef std::function<std::tuple<double, double, double, double, double, double>(int, int, int)>
    grid3DProbabilityFunc;

typedef std::tuple<
    std::unique_ptr<ListDigraph>, 
    std::unique_ptr<ListDigraph::ArcMap<double>>> 
    PresetResults;

PresetResults generate1DGridChain(
    int nodeCount,
    double leftProbability,
    double rightProbability,
    double leftLeftProbability = 0.0,
    double rightRightProbability = 0.0) {

    std::vector<ListDigraph::Node> nodes;
    std::vector<ListDigraph::Arc> rightRightArcs;
    std::vector<ListDigraph::Arc> rightArcs;
    std::vector<ListDigraph::Arc> selfArcs;
    std::vector<ListDigraph::Arc> leftArcs;
    std::vector<ListDigraph::Arc> leftLeftArcs;
    nodes.reserve(nodeCount);

    double stayProbability = 1.0 
        - leftProbability 
        - rightProbability 
        - leftLeftProbability 
        - rightRightProbability;

    assert(stayProbability >= 0.0);

    auto idx = [nodeCount](int i) {
        return (i + nodeCount) % nodeCount;
    };
    
    auto graph = std::make_unique<ListDigraph>();

    for (int i = 0; i < nodeCount; ++i) {
        nodes.emplace_back(graph->addNode());
    }

    for (int i = 0; i < nodeCount; ++i) {
        if (leftLeftProbability > 0.0)
            leftLeftArcs.emplace_back(graph->addArc(nodes[idx(i)], nodes[idx(i - 2)]));
        
        leftArcs.emplace_back(graph->addArc(nodes[idx(i)], nodes[idx(i - 1)]));
        selfArcs.emplace_back(graph->addArc(nodes[idx(i)], nodes[idx(i)]));
        rightArcs.emplace_back(graph->addArc(nodes[idx(i)], nodes[idx(i + 1)]));
        
        if (rightRightProbability > 0.0)
            rightRightArcs.emplace_back(graph->addArc(nodes[idx(i)], nodes[idx(i + 2)]));
    }

    auto probabilities = std::make_unique<ListDigraph::ArcMap<double>>(*graph);

    for (int i = 0; i < nodeCount; ++i) {
        if (leftLeftProbability > 0.0)
            (*probabilities)[leftLeftArcs[i]] = leftLeftProbability;

        (*probabilities)[leftArcs[i]] = leftProbability;
        (*probabilities)[selfArcs[i]] = stayProbability;
        (*probabilities)[rightArcs[i]] = rightProbability;
        
        if (rightRightProbability > 0.0)
            (*probabilities)[rightRightArcs[i]] = rightRightProbability;
    }

    return std::make_tuple(std::move(graph), std::move(probabilities));
}

PresetResults generate2DGridChain(
    int nodeCountX,
    int nodeCountY,
    const grid2DProbabilityFunc& func) {
    
    std::vector<ListDigraph::Node> nodes;
    std::vector<ListDigraph::Arc> rightArcs;
    std::vector<ListDigraph::Arc> selfArcs;
    std::vector<ListDigraph::Arc> leftArcs;
    std::vector<ListDigraph::Arc> upArcs;
    std::vector<ListDigraph::Arc> downArcs;

    auto idx = [nodeCountX, nodeCountY](int x, int y) {
        x = (x + nodeCountX) % nodeCountX;
        y = (y + nodeCountY) % nodeCountY;
        return y * nodeCountX + x;
    };

    int nodeCount = nodeCountX * nodeCountY;
    nodes.reserve(nodeCount);

    auto graph = std::make_unique<ListDigraph>();

    for (int i = 0; i < nodeCount; ++i) {
        nodes.emplace_back(graph->addNode());
    }

    for (int y = 0; y < nodeCountY; ++y) {
        for (int x = 0; x < nodeCountX; ++x) {
            leftArcs.emplace_back(graph->addArc(
                nodes[idx(x, y)], nodes[idx(x - 1, y)]));
            rightArcs.emplace_back(graph->addArc(
                nodes[idx(x, y)], nodes[idx(x + 1, y)]));
            upArcs.emplace_back(graph->addArc(
                nodes[idx(x, y)], nodes[idx(x, y - 1)]));
            downArcs.emplace_back(graph->addArc(
                nodes[idx(x, y)], nodes[idx(x, y + 1)]));
            selfArcs.emplace_back(graph->addArc(
                nodes[idx(x, y)], nodes[idx(x, y)]));
        }
    }

    auto probabilities = std::make_unique<ListDigraph::ArcMap<double>>(*graph);

    for (int y = 0; y < nodeCountY; ++y) {
        for (int x = 0; x < nodeCountX; ++x) {
            int i = idx(x, y);

            auto [leftProbability, rightProbability, upProbability, downProbability] = func(x, y);
            double stayProbability = 1.0 
                - leftProbability 
                - rightProbability 
                - upProbability 
                - downProbability;

            assert(stayProbability >= 0.0);

            (*probabilities)[leftArcs[i]] = leftProbability;
            (*probabilities)[rightArcs[i]] = rightProbability;
            (*probabilities)[upArcs[i]] = upProbability;
            (*probabilities)[downArcs[i]] = downProbability;
            (*probabilities)[selfArcs[i]] = stayProbability;
        } 
    }

    return std::make_tuple(std::move(graph), std::move(probabilities));
}

PresetResults generate3DGridChain(
    int nodeCountX,
    int nodeCountY,
    int nodeCountZ,
    const grid3DProbabilityFunc& func) {
    
    std::vector<ListDigraph::Node> nodes;
    std::vector<ListDigraph::Arc> rightArcs;
    std::vector<ListDigraph::Arc> selfArcs;
    std::vector<ListDigraph::Arc> leftArcs;
    std::vector<ListDigraph::Arc> upArcs;
    std::vector<ListDigraph::Arc> downArcs;
    std::vector<ListDigraph::Arc> forwardArcs;
    std::vector<ListDigraph::Arc> backwardArcs;

    auto idx = [nodeCountX, nodeCountY, nodeCountZ](int x, int y, int z) {
        x = (x + nodeCountX) % nodeCountX;
        y = (y + nodeCountY) % nodeCountY;
        z = (z + nodeCountZ) % nodeCountZ;
        return z * nodeCountX * nodeCountY + y * nodeCountX + x;
    };

    int nodeCount = nodeCountX * nodeCountY;
    nodes.reserve(nodeCount);

    auto graph = std::make_unique<ListDigraph>();

    for (int i = 0; i < nodeCount; ++i) {
        nodes.emplace_back(graph->addNode());
    }

    for (int z = 0; z < nodeCountZ; ++z) {
        for (int y = 0; y < nodeCountY; ++y) {
            for (int x = 0; x < nodeCountX; ++x) {
                leftArcs.emplace_back(graph->addArc(
                    nodes[idx(x, y, z)], nodes[idx(x - 1, y, z)]));
                rightArcs.emplace_back(graph->addArc(
                    nodes[idx(x, y, z)], nodes[idx(x + 1, y, z)]));
                upArcs.emplace_back(graph->addArc(
                    nodes[idx(x, y, z)], nodes[idx(x, y - 1, z)]));
                downArcs.emplace_back(graph->addArc(
                    nodes[idx(x, y, z)], nodes[idx(x, y + 1, z)]));
                forwardArcs.emplace_back(graph->addArc(
                    nodes[idx(x, y, z)], nodes[idx(x, y, z - 1)]));
                backwardArcs.emplace_back(graph->addArc(
                    nodes[idx(x, y, z)], nodes[idx(x, y, z + 1)]));
                selfArcs.emplace_back(graph->addArc(
                    nodes[idx(x, y, z)], nodes[idx(x, y, z)]));
            }
        }
    }

    auto probabilities = std::make_unique<ListDigraph::ArcMap<double>>(*graph);

    for (int z = 0; z < nodeCountZ; ++z) {
        for (int y = 0; y < nodeCountY; ++y) {
            for (int x = 0; x < nodeCountX; ++x) {
                int i = idx(x, y, z);

                auto [leftProbability, 
                    rightProbability, 
                    upProbability, 
                    downProbability, 
                    forwardProbability, 
                    backwardProbability] = func(x, y, z);
                double stayProbability = 1.0 
                    - leftProbability 
                    - rightProbability 
                    - upProbability 
                    - downProbability
                    - forwardProbability
                    - backwardProbability;

                assert(stayProbability >= 0.0);

                (*probabilities)[leftArcs[i]] = leftProbability;
                (*probabilities)[rightArcs[i]] = rightProbability;
                (*probabilities)[upArcs[i]] = upProbability;
                (*probabilities)[downArcs[i]] = downProbability;
                (*probabilities)[forwardArcs[i]] = forwardProbability;
                (*probabilities)[backwardArcs[i]] = backwardProbability;
                (*probabilities)[selfArcs[i]] = stayProbability;
            } 
        }
    }

    return std::make_tuple(std::move(graph), std::move(probabilities));
}

PresetResults generateComplete(
    int nodeCount) {
    
    std::vector<ListDigraph::Node> nodes;
    std::vector<ListDigraph::Arc> arcs;
    nodes.reserve(nodeCount);
    arcs.reserve(nodeCount * nodeCount);

    double prob = 1.0 / (double)nodeCount;
    
    auto graph = std::make_unique<ListDigraph>();

    for (int i = 0; i < nodeCount; ++i) {
        nodes.emplace_back(graph->addNode());
    }

    for (int i = 0; i < nodeCount; ++i) {
        for (int j = 0; j < nodeCount; ++j) {
            arcs.emplace_back(graph->addArc(nodes[i], nodes[j]));
        }
    }

    auto probabilities = std::make_unique<ListDigraph::ArcMap<double>>(*graph);

    for (int i = 0; i < nodeCount; ++i) {
        (*probabilities)[arcs[i]] = prob;
    }

    return std::make_tuple(std::move(graph), std::move(probabilities));
}

PresetResults generatePreset(GraphPreset present, int gridSize) {
    switch (present) {
        case GraphPreset::P1D_0_1d2:
            return generate1DGridChain(gridSize, 0.0, 0.5);
        case GraphPreset::P1D_1d4_1d4:
            return generate1DGridChain(gridSize, 0.25, 0.25);
        case GraphPreset::P1D_1d6_2d6:
            return generate1DGridChain(gridSize, 1.0 / 6.0, 2.0 / 6.0);
        case GraphPreset::P1D_1d8_1d8_1d8_1d8:
            return generate1DGridChain(gridSize, 0.125, 0.125, 0.125, 0.125);
        case GraphPreset::P1D_complete:
            return generateComplete(gridSize);
        case GraphPreset::P2D_unif:
        {
            auto probFunc = [](int x, int y) {
                return std::make_tuple(0.25, 0.25, 0.25, 0.25);
            };
            return generate2DGridChain(gridSize, gridSize, probFunc);
        }
        case GraphPreset::P2D_nonunif:
        {
            auto probFunc = [gridSize](int x, int y) {
                auto dhoriz = std::sin(2.0 * M_PI * (double)x / (double)gridSize) / 8.0;
                auto dvert = std::sin(2.0 * M_PI * (double)y / (double)gridSize) / 8.0;
                return std::make_tuple(0.25 - dhoriz, 0.25 + dhoriz, 0.25 - dvert, 0.25 + dvert);
            };
            return generate2DGridChain(gridSize, gridSize, probFunc);
        }
        case GraphPreset::P3D_unif:
        {
            auto probFunc = [](int x, int y, int z) {
                auto unifProb = 1.0 / 6.0;
                return std::make_tuple(
                    unifProb, 
                    unifProb, 
                    unifProb, 
                    unifProb, 
                    unifProb, 
                    unifProb);
            };
            return generate3DGridChain(gridSize, gridSize, gridSize, probFunc);
        }
        case GraphPreset::P3D_nonunif:
        {
            auto probFunc = [gridSize](int x, int y, int z) {
                auto dhoriz = std::sin(2.0 * M_PI * (double)x / (double)gridSize) / 8.0;
                auto dvert = std::sin(2.0 * M_PI * (double)y / (double)gridSize) / 8.0;
                auto ddepth = std::sin(2.0 * M_PI * (double)z / (double)gridSize) / 8.0;
                auto unifProb = 1.0 / 6.0;
                return std::make_tuple(
                    unifProb - dhoriz, 
                    unifProb + dhoriz, 
                    unifProb - dvert, 
                    unifProb + dvert, 
                    unifProb - ddepth, 
                    unifProb + ddepth);
            };
            return generate3DGridChain(gridSize, gridSize, gridSize, probFunc);
        }
        default:
            throw std::runtime_error("Not implemeneted!");
    }
}

struct MarkovSampleParameters {
    std::shared_ptr<ListDigraph::ArcMap<double>> probabilities;
};
struct MarkovSampleHyperparameters {
    ListDigraph* graph;
    int samples;
    double discountFactor;
};

typedef MatrixParameterDistribution<
    MarkovSampleParameters, 
    MarkovSampleHyperparameters> DistributionBase;
typedef ProblemDefinition<
    MarkovSampleParameters, 
    MarkovSampleHyperparameters> ProblemDefType;

class MarkovHistogramSampleDistribution : public DistributionBase {
private:
    const size_t dimension;

public:
    void drawParameters(
        MarkovSampleParameters* output) const override {
        auto newProbabilities = std::make_shared<
            ListDigraph::ArcMap<double>>(*hyperparameters.graph);

        histogramSample(hyperparameters.graph, 
            parameters.probabilities.get(), 
            hyperparameters.samples, 
            newProbabilities.get());

        output->probabilities = std::move(newProbabilities);
    }

    std::shared_ptr<IInvertibleMatrixOperator> convert(
        const MarkovSampleParameters& params) const override {
        Eigen::SparseMatrix<double> matrix(getDimension(), getDimension());
        
        markovGenerator(
            hyperparameters.graph, 
            params.probabilities.get(), 
            &matrix, 
            hyperparameters.discountFactor);

        return std::make_shared<SparseMatrixSampleSquare>(matrix);
    }

    size_t getDimension() const override {
        return dimension;
    }

    bool isSPD() const override {
        return false;
    }

    MarkovHistogramSampleDistribution(
        MarkovSampleParameters& parameters, 
        MarkovSampleHyperparameters& hyperparameters) :
        DistributionBase(parameters, hyperparameters), 
        dimension(lemon::countNodes(*hyperparameters.graph)) {
    }
};

typedef Diagnostics<MarkovSampleParameters, MarkovSampleHyperparameters, MarkovHistogramSampleDistribution>
        DiagnosticsType;
typedef ProblemRun<MarkovSampleParameters, MarkovSampleHyperparameters>
        ProblemRunType;

void dgnAsymMarkov(int argc, char** argv) {

    int histogramSamples = DEFAULT_HISTOGRAPH_SAMPLE_COUNT;
    int threadCount = DEFAULT_THREAD_COUNT;
    int numSubRunsNaive = DEFAULT_NUM_SUB_RUNS_NAIVE;
    int numSubRuns = DEFAULT_NUM_SUB_RUNS;
    int samplesPerSubRun = SAMPLES_PER_SUB_RUN;
    int gridSize = DEFAULT_GRID_SIZE;
    double discountFactor = DEFAULT_DISCOUNT_FACTOR;
    int preset = DEFAULT_PRESET;

    std::cout << "################ AsymMarkov ################" << std::endl << std::endl;

    // Read global configuration from file
    if (globalConfigAvailable) {
        auto config = globalConfig["AsymMarkov"];

        preset = config["preset"].get<int>();
        histogramSamples = config["histogramSamples"].get<int>();
        threadCount = config["threadCount"].get<int>();
        numSubRunsNaive = config["numSubRunsNaive"].get<int>();
        numSubRuns = config["numSubRuns"].get<int>();
        samplesPerSubRun = config["samplesPerSubRun"].get<int>();
        discountFactor = config["discountFactor"].get<double>();
        gridSize = config["gridSize"].get<int>();

        std::cout << "Configuration:" << std::endl;
        std::cout << std::setw(4) << config << std::endl << std::endl;
    }

    if (argc > 0)
        preset = std::stoi(argv[0]);

    auto [graph, probabilities] = generatePreset(
        (GraphPreset)preset, gridSize);

    MarkovSampleParameters params;
    params.probabilities = std::move(probabilities);

    MarkovSampleHyperparameters hyperparams;
    hyperparams.graph = graph.get();
    hyperparams.samples = histogramSamples;
    hyperparams.discountFactor = discountFactor;

    auto true_mat_dist = std::make_shared<
        MarkovHistogramSampleDistribution>(params, hyperparams);
    auto problem_def = std::make_shared<
        ProblemDefType>(true_mat_dist);
    auto diagnostics = DiagnosticsType(problem_def);

    // Naive run
    auto run = std::shared_ptr<ProblemRunType>(
            new NaiveRun<
                MarkovSampleParameters, 
                MarkovSampleHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRunsNaive;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new ResidualOpshiftRun<
                MarkovSampleParameters, 
                MarkovSampleHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new ResidualOpshiftTruncatedRun<
                MarkovSampleParameters, 
                MarkovSampleHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    diagnostics.run(threadCount);
    diagnostics.printResults();
    diagnostics.printLatexTable();
}

int main(int argc, char *argv[]) {
    loadConfig();
    dgnAsymMarkov(argc - 1, &argv[1]);
}