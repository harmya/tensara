from abc import ABC, abstractmethod
import numpy as np

class Tensor:
    """Wrapper class for tensors"""
    def __init__(self, shape, dtype=np.float32):
        self.shape_tuple = tuple(shape)
        self.data = np.empty(self.shape_tuple, dtype=dtype)
        self.dtype = dtype
    
    def shape(self):
        return self.shape_tuple
    
    def size(self):
        return int(np.prod(self.shape_tuple))
    
    def nbytes(self):
        return self.data.nbytes


class TestCase(ABC):
    """Abstract base class for test cases"""
    
    def __init__(self):
        self.name = None
        self.problem_size = 0
        self.inputs = []
        self.outputs = []
    
    @abstractmethod
    def prepare_data(self, inputs, outputs):
        """Prepare input data for the test"""
        pass
    
    @abstractmethod
    def get_name(self):
        """Get the name of the test case"""
        return self.name
    
    @abstractmethod
    def calculate_flops(self):
        """Calculate the number of floating point operations for this test"""
        pass
    
    @abstractmethod
    def get_sizes(self):
        """Get the sizes for this test case"""
        pass
    
    @abstractmethod
    def launch_kernel(self, inputs, outputs, sizes, kernel_func):
        """Launch the CUDA kernel"""
        pass

    def input_shapes(self):
        """Get the shapes of the input tensors"""
        return self.inputs
    
    def output_shapes(self):
        """Get the shapes of the output tensors"""
        return self.outputs