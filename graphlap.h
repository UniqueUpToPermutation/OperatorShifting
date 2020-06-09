#ifndef OPERATORAUGMENTATION_GRAPHLAP_H
#define OPERATORAUGMENTATION_GRAPHLAP_H

#include <string>
#include <lemon/list_graph.h>
#include <Eigen/Sparse>

void loadGraphUnweighted(const std::string& filename, lemon::ListGraph* output);
void loadGraphWeighted(const std::string& filename, lemon::ListGraph* output, lemon::ListGraph::EdgeMap<double>* weights);
void enumGraphLaplacian(const lemon::ListGraph* graph, const lemon::ListGraph::EdgeMap<double>* weights,
        std::vector<Eigen::Triplet<double>>* output);
void enumGraphLaplacian(const lemon::ListGraph* graph,
                        std::vector<Eigen::Triplet<double>>* output);
void graphLaplacian(const lemon::ListGraph* graph, const lemon::ListGraph::EdgeMap<double>* weights,
        Eigen::SparseMatrix<double>* output);
void graphLaplacian(const lemon::ListGraph* graph,
                    Eigen::SparseMatrix<double>* output);

#endif
