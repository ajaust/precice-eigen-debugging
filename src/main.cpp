#include<iostream>
#include<vector>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>
#include <unsupported/Eigen/SparseExtra>

#include "precice/SolverInterface.hpp"

int main(int argc, char* argv[])
{

  Eigen::SparseMatrix<double> A;
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;

  precice::SolverInterface precice( "EigenPrecice", 0, 1 );
  precice.configure( "precice-config.xml" );

  precice.initialize();

  constexpr int matrixSize = 3;
  A.resize( matrixSize, matrixSize );
  A.reserve( matrixSize );
  std::vector<Eigen::Triplet<double>> matrixData(matrixSize);
  Eigen::VectorXd b( matrixSize );
  for (int i = 0; i < matrixSize; ++i ) {
    matrixData[i] = ( Eigen::Triplet<double>( i, i, (double) (i+1) ) );
    b[i] = 1.;
  }
  A.setFromTriplets( matrixData.begin(), matrixData.end() );
  A.makeCompressed();

  // Solve system
  solver.compute(A);
  Eigen::VectorXd x = solver.solve( b );

  for (int i = 0; i < matrixSize; ++i) {
    std::cout << "x[" << i << "] = " << x[i] << std::endl;
  }
  
  precice.finalize();

  return 0;
}
