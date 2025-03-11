/*
    Maps problem slugs to their python modules
*/

export const slugToPythonModuleMap: Record<string, string> = {
  'conv-1d': 'conv_1d',
  'conv-2d': 'conv_2d',
  'gemm-relu': 'gemm_relu',
  'leaky-relu': 'leaky_relu',
  'matrix-multiplication': 'matrix_multiplication',
  'matrix-vector': 'matrix_vector',
  'relu': 'relu',
  'square-matmul': 'square_matmul',
  'vector-addition': 'vector_addition'
};