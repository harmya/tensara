from pathlib import Path

from utils import Tensor, TestCase
from cuda_tester import CUDATester
from vecadd_test import VectorAddTest, create_vector_add_tests, vector_add_ref

def checker_t4():
    import pycuda.autoinit
    import pycuda.driver as cuda
    import numpy as np
    from pycuda.compiler import SourceModule

    cuda_source = """
    extern "C" {
        __global__ void vector_add(float *a, float *b, float *c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    }
    """

    tester = CUDATester(
        cuda_source=cuda_source,
        kernel_name="vector_add",
        reference_func=vector_add_ref
    )

    # Run tests
    test_cases = create_vector_add_tests()
    results = []
    
    for test_case in test_cases:
        result = tester.run_test(test_case)
        results.append(result)
        print(f"Test {result['name']} {'PASSED' if result['passed'] else 'FAILED'}")
        print(f"  Max diff: {result.get('max_diff', 'N/A')}")
        print(f"  Execution time: {result.get('execution_time', 'N/A'):.6f} seconds")
    
    return results
    
def main():
    checker_t4()    