from utils import TestCase
from utils import Tensor
import numpy as np


class VectorAddTest(TestCase):
    def __init__(self, size, dtype=np.float32):
        super().__init__()
        self.vec_size = size
        self.dtype = dtype
        self.name = f"VectorAdd-{size}"
        
        self.inputs = [
            Tensor([size], dtype), 
            Tensor([size], dtype)  
        ]
        self.outputs = [
            Tensor([size], dtype)  
        ]
    
    def prepare_data(self, host_inputs, host_outputs):
        """Prepare input data for the test"""
        host_inputs[0][:] = np.arange(self.vec_size, dtype=self.dtype)
        host_inputs[1][:] = np.arange(self.vec_size, dtype=self.dtype)[::-1]  # Reversed
    
    def get_name(self):
        """Get the name of the test case"""
        return self.name
    
    def calculate_flops(self):
        """Calculate the number of floating point operations for this test"""
        return self.vec_size 
    
    def get_sizes(self):
        """Get the sizes for this test case"""
        return [self.vec_size]
    
    def launch_kernel(self, inputs, outputs, sizes, kernel_func):
        """Launch the CUDA kernel"""
        block_size = 256
        grid_size = (sizes[0] + block_size - 1) // block_size
        
        # Launch kernel
        kernel_func(
            inputs[0], inputs[1], outputs[0],
            np.int32(sizes[0]),
            block=(block_size, 1, 1),
            grid=(grid_size, 1, 1)
        )


def create_vector_add_tests(dtype=np.float32):
    """Create a set of vector addition test cases with different sizes"""
    return [
        VectorAddTest(100000, dtype),
        VectorAddTest(1000000, dtype),
        VectorAddTest(10000000, dtype)
    ]

def vector_add_ref(a, b):
    return a + b