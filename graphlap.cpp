#include "graphlap.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;
using namespace lemon;

void loadGraphUnweighted(const string& filename, ListGraph* output) {
    ifstream f(filename);

    if (!f.is_open())
        throw "LoadGraph: file not found!";

    std::vector<std::pair<int, int>> edges;

    for (std::string line; std::getline(f, line); ) {
        if (line[0] == '%') // Ignore comments
            continue;

        std::replace_if(std::begin(line), std::end(line), [] (char x) { return std::ispunct(x); }, ' ');
        stringstream ss(line);

        int u, v;
        if (ss >> u >> v)
            edges.emplace_back(std::make_pair(u, v));
    }

    int max_id = -1;
    for (auto& e : edges)
        max_id = std::max(max_id, std::max(e.first, e.second));

    for (int i = 0; i <= max_id; ++i)
        output->addNode();

    for (auto& e : edges)
        output->addEdge(output->nodeFromId(e.first), output->nodeFromId(e.second));
}

void loadGraphWeighted(const std::string& filename, lemon::ListGraph* output,
        lemon::ListGraph::EdgeMap<double>* weights) {
    ifstream f(filename);

    if (!f.is_open())
        throw "LoadGraph: file not found!";

    std::vector<std::pair<std::pair<int, int>, double>> edges;

    for (std::string line; std::getline(f, line); ) {
        if (line[0] == '%') // Ignore comments
            continue;

        std::replace_if(std::begin(line), std::end(line), [] (char x) { return std::ispunct(x); }, ' ');
        stringstream ss(line);

        int u, v;
        double w;
        if (ss >> u >> v >> w)
            edges.emplace_back(std::make_pair(std::make_pair(u, v), w));
    }

    int max_id = -1;
    for (auto& e : edges)
        max_id = std::max(max_id, std::max(e.first.first, e.first.second));

    for (int i = 0; i <= max_id; ++i)
        output->addNode();

    for (auto& e : edges) {
        auto eInGraph = output->addEdge(output->nodeFromId(e.first.first), output->nodeFromId(e.first.second));
        (*weights)[eInGraph] = e.second;
    }
}

void enumGraphLaplacian(const lemon::ListGraph* graph, const lemon::ListGraph::EdgeMap<double>* weights,
                        std::vector<Eigen::Triplet<double>>* output) {
    int edgeCount = countEdges(*graph);
    int vertexCount = countNodes(*graph);
    output->reserve(edgeCount + vertexCount);
    for (ListGraph::NodeIt u(*graph); u != INVALID; ++u) {
        double degree = 0.0;
        for (ListGraph::IncEdgeIt e(*graph, u); e != INVALID; ++e) {
            degree += weights->operator[](e);
            auto v = graph->oppositeNode(u, e);
            output->emplace_back(Eigen::Triplet<double>(graph->id(u), graph->id(v), -weights->operator[](e)));
        }
        output->emplace_back(Eigen::Triplet<double>(graph->id(u), graph->id(u), degree));
    }
}

void enumGraphLaplacian(const lemon::ListGraph* graph,
                        std::vector<Eigen::Triplet<double>>* output) {
    int edgeCount = countEdges(*graph);
    int vertexCount = countNodes(*graph);
    output->reserve(edgeCount + vertexCount);
    for (ListGraph::NodeIt u(*graph); u != INVALID; ++u) {
        double degree = 0.0;
        for (ListGraph::IncEdgeIt e(*graph, u); e != INVALID; ++e) {
            degree += 1.0;
            auto v = graph->oppositeNode(u, e);
            output->emplace_back(Eigen::Triplet<double>(graph->id(u), graph->id(v), -1.0));
        }
        output->emplace_back(Eigen::Triplet<double>(graph->id(u), graph->id(u), degree));
    }
}

void graphLaplacian(const lemon::ListGraph* graph, const lemon::ListGraph::EdgeMap<double>* weights,
                    Eigen::SparseMatrix<double>* output) {
    int count = countNodes(*graph);
    output->resize(count, count);
    std::vector<Eigen::Triplet<double>> nzEntries;
    enumGraphLaplacian(graph, weights, &nzEntries);
    output->setFromTriplets(nzEntries.begin(), nzEntries.end());
}

void graphLaplacian(const lemon::ListGraph* graph,
                    Eigen::SparseMatrix<double>* output) {
    int count = countNodes(*graph);
    output->resize(count, count);
    std::vector<Eigen::Triplet<double>> nzEntries;
    enumGraphLaplacian(graph, &nzEntries);
    output->setFromTriplets(nzEntries.begin(), nzEntries.end());
}