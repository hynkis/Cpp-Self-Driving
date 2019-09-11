/* ====================================================
TODO:
  - (V) Quadratic Objective (P)
  - (V) Linear Objective (q)
  - Equality Constraint (lb, ub)
  - Augmented Matrix (A)








====================================================== */
#include "osqp.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "iostream"

#define NX 6
#define NU 2
#define N 40

// Quadratic Objective (P)
void castMPCToQPHessian(const Eigen::DiagonalMatrix<double, NX> &Q, const Eigen::DiagonalMatrix<double, NX> &QN, Eigen::DiagonalMatrix<double, NU> &R, int mpcWindow,
                        Eigen::SparseMatrix<double> &hessianMatrix)
{
  hessianMatrix.resize(NX*(mpcWindow+1) + NU*mpcWindow, NX*(mpcWindow+1) + NU*mpcWindow);

  // Populate hessian matrix
  for (int i=0; i<NX*(mpcWindow+1) + NU*mpcWindow; i++)
  {
    if (i < NX*(mpcWindow)) // Q weight matrix
    {
      int posQ = i%NX;
      float value = Q.diagonal()[posQ];
      if (value != 0)
        hessianMatrix.insert(i,i) = value;
    }
    else if (i >= NX*(mpcWindow) && i < NX*(mpcWindow+1)) // QN weight matrix
    {
      int posQN = i%NX;
      float value = QN.diagonal()[posQN];
      if (value != 0)
        hessianMatrix.insert(i,i) = value;
    }
    else // i >= NX*(mpcWindow+1) : R weight matrix
    {
      int posR = i%NU;
      float value = R.diagonal()[posR];
      if (value != 0)
        hessianMatrix.insert(i,i) = value;
    }
  }
}

// Linear Objective (q)
void castMPCToQPGradient(const Eigen::DiagonalMatrix<double, NX> &Q, const Eigen::DiagonalMatrix<double, NX> &QN, const Eigen::Matrix<double, NX, N> &xRef, int mpcWindow,
                        Eigen::VectorXd &gradient)
{
  // Populate the gradient vector
  gradient = Eigen::VectorXd::Zero(NX*(mpcWindow+1) + NU*mpcWindow, 1);
  Eigen::Matrix<double, NX, 1> Qx_ref;

  for (int i=0; i<NX*(mpcWindow+1); i++)
  {
    if (i < NX*(mpcWindow)) // Q weight matrix
    {
      if (i%NX == 0) // Calculate -Qx_ref at every each x_ref.
      {
        int posxRef = i%NX;
        Qx_ref = Q * -xRef.row(i%NX);
      }
      int posQ = i%NX;
      float value = Qx_ref(posQ,0);
      gradient(i,0) = value;
    }
    else if (i >= NX*(mpcWindow) && i < NX*(mpcWindow+1)) // QN weight matrix
    {
      if (i%NX == 0) // Calculate -Qx_ref at every each x_ref.
      {
        int posxRef = i%NX;
        Qx_ref = QN * -xRef.row(i%NX);
      }
      int posQ = i%NX;
      float value = Qx_ref(posQ,0);
      gradient(i,0) = value;
    }
  }
}

int main(int argc, char **argv)
{

  // Define Weight matrix
  Eigen::SparseMatrix<double> Q(NX,NX);
  Eigen::SparseMatrix<double> R(NU,NU);
  
  // Dynamics model list
  std::vector<Eigen::SparseMatrix<double>> Q_list;
  Eigen::SparseMatrix<double> Q1(3,3);
  Q1.diagonal() << 1,2,3;
  Eigen::SparseMatrix<double> Q2(3,3);
  Q2.diagonal() << 3,6,9;
  Q_list.push_back(Q1);
  Q_list.push_back(Q2);

  std::cout << "Q1 : " << Q1.diagonal() << " Q2 : " << Q2.diagonal() << std::endl;
  std::cout << "Q1(1,1) : " << Q1.coeff(1,1) << std::endl;

  /*
  // Convert to sparse matrix
  Eigen::SparseMatrix<double> sparse = dense.sparseView();

  // Extract the needed values
  sparse.makeCompressed();
  int cols = sparse.outerSize();
  int rows = sparse.innerSize();
  int nz_max = sparse.nonZeros();


  std::cout << "nz_max : " << nz_max << std::endl;
  std::cout << "m (rows) : " << rows << std::endl;
  std::cout << "n (cols) : " << cols << std::endl;
  std::cout << "*p : " << sparse.outerIndexPtr()[0] << sparse.outerIndexPtr()[1] << sparse.outerIndexPtr()[2] << std::endl;
  std::cout << "sparse.outerSize() + 1 : " << sparse.outerSize() + 1 << std::endl;
  std::cout << "*i : " << *(sparse.innerIndexPtr()) << *(sparse.innerIndexPtr()+1) << *(sparse.innerIndexPtr()+2) << std::endl;
  std::cout << "*x : " << *(sparse.valuePtr()+1) << std::endl; // nonzero 개수만큼 얻어오기

  c_int *outerIndexPtr = new c_int[cols+1];
  c_int *innerIndexPtr = new c_int[nz_max];

  for (int i = 0;i<cols+1;i++)
  {
    outerIndexPtr[i] = *(sparse.outerIndexPtr() + i);
  }

  for (int i=0; i<nz_max; i++)
  {
    innerIndexPtr[i] = *(sparse.innerIndexPtr() + i);
  }

  delete outerIndexPtr;

  // Load problem data
  c_float P_x[3] = { 4.0, 1.0, 2.0, };
  c_int   P_nnz  = 3;
  c_int   P_i[3] = { 0, 0, 1, };
  c_int   P_p[3] = { 0, 1, 3, };
  c_float q[2]   = { 1.0, 1.0, };
  c_float A_x[4] = { 1.0, 1.0, 1.0, 1.0, };
  c_int   A_nnz  = 4;
  c_int   A_i[4] = { 0, 1, 0, 2, };
  c_int   A_p[3] = { 0, 2, 4, };
  c_float l[3]   = { 1.0, 0.0, 0.0, };
  c_float u[3]   = { 1.0, 0.7, 0.7, };
  c_int n = 2;
  c_int m = 3;

  // Exitflag
  c_int exitflag = 0;

  // Workspace structures
  OSQPWorkspace *work;
  OSQPSettings  *settings   = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  OSQPData      *data       = (OSQPData *)c_malloc(sizeof(OSQPData));
  OSQPData      *data_eigen = (OSQPData *)c_malloc(sizeof(OSQPData));

  // Populate data
  if (data) {
    data->n = n;
    data->m = m;
    data->P = csc_matrix(rows, cols, nz_max, sparse.valuePtr(), innerIndexPtr, outerIndexPtr);
    data->q = q;
    data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
    data->l = l;
    data->u = u;
  }

  // Define solver settings as default
  if (settings) osqp_set_default_settings(settings);

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Solve Problem
  osqp_solve(work);

  // Clean workspace
  osqp_cleanup(work);
  if (data) {
    if (data->A) c_free(data->A);
    if (data->P) c_free(data->P);
    c_free(data);
  }
  if (settings)  c_free(settings);

  return exitflag;
  */
  return 0;
}