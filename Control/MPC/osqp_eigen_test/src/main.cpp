#include "osqp.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "iostream"

int main(int argc, char **argv)
{
  // Eigen Matrix
  Eigen::MatrixXd dense(4,6);
  dense << 11,   0,   0,  14,   0,  16,
            0,  22,   0,   0,  25,  26,
            0,   0,  33,  34,   0,  36,
          41,   0,  43,  44,   0,  46;

  // Convert to sparse matrix
  Eigen::SparseMatrix<double> sparse = dense.sparseView();

  // Extract the needed values
  sparse.makeCompressed();
  std::cout << "nz_max : " << sparse.nonZeros() << std::endl;
  std::cout << "m (rows) : " << sparse.innerSize() << std::endl;
  std::cout << "n (cols) : " << sparse.outerSize() << std::endl;
  std::cout << "*p : " << sparse.outerIndexPtr()[0] << sparse.outerIndexPtr()[1] << sparse.outerIndexPtr()[2] << std::endl;
  std::cout << "sparse.outerSize() + 1 : " << sparse.outerSize() + 1 << std::endl;
  std::cout << "*i : " << *(sparse.innerIndexPtr()) << *(sparse.innerIndexPtr()+1) << *(sparse.innerIndexPtr()+2) << std::endl;
  std::cout << "*x : " << *(sparse.valuePtr()+1) << std::endl; // nonzero 개수만큼 얻어오기

  // // Load problem data
  // c_float P_x[4] = {4.00, 1.00, 1.00, 2.00, };
  // c_int P_nnz = 4;
  // c_int P_p[3] = {0, 2, 4, };
  // c_int P_i[4] = {0, 1, 0, 1, };
  
  // c_float q[2] = {1.00, 1.00, };
  // c_float A_x[4] = {1.00, 1.00, 1.00, 1.00, };
  // c_int A_nnz = 4;
  // c_int A_i[4] = {0, 1, 0, 2, };
  // c_int A_p[3] = {0, 2, 4, };
  // c_float l[3] = {1.00, 0.00, 0.00, };
  // c_float u[3] = {1.00, 0.70, 0.70, };
  // c_int n = 2;
  // c_int m = 3;

  // // Exitflag
  // c_int exitflag = 0;

  // // Workspace structures
  // OSQPWorkspace *work;
  // OSQPSettings  *settings   = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  // OSQPData      *data       = (OSQPData *)c_malloc(sizeof(OSQPData));
  // OSQPData      *data_eigen = (OSQPData *)c_malloc(sizeof(OSQPData));

  // // Populate data
  // if (data) {
  //   data->n = n;
  //   data->m = m;
  //   data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
  //   data->q = q;
  //   data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
  //   data->l = l;
  //   data->u = u;
  // }

  // // if (data_eigen) {
  // //   data->n = sparse.outerSize();
  // //   data->m = sparse.innerSize();
  // //   data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
  // //   data->q = q;
  // //   data->A = csc_matrix(data->m, data->n, sparse.nonZeros(), A_x, A_i, A_p);
  // //   data->l = l;
  // //   data->u = u;
  // // }

  // // Define solver settings as default
  // if (settings) osqp_set_default_settings(settings);

  // // Setup workspace
  // exitflag = osqp_setup(&work, data, settings);

  // // Solve Problem
  // osqp_solve(work);

  // // Clean workspace
  // osqp_cleanup(work);
  // if (data) {
  //   if (data->A) c_free(data->A);
  //   if (data->P) c_free(data->P);
  //   c_free(data);
  // }
  // if (settings)  c_free(settings);

  // return exitflag;
}