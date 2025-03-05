import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
from typing import Optional, Callable, List

class CUDATester:
    """Generic tester for CUDA test cases"""
    
    def __init__(self, 
                 cuda_source: str, 
                 kernel_name: str, 
                 reference_func: Optional[Callable] = None,
    ):
        """
        Initialize the test runner
        
        Args:
            cuda_source: String containing CUDA kernel code
            kernel_name: Name of the kernel function in the CUDA code
            reference_func: PyTorch function for reference implementation
        """
        self.cuda_source = cuda_source
        self.kernel_name = kernel_name
        self.reference_func = reference_func
        
        # Compile the source code
        self.module = SourceModule(cuda_source)
        
        # Get the kernel function
        self.kernel_func = self.module.get_function(kernel_name)
    
    def run_test(self, test_case, tolerance=1e-7):
        """Run a test case and check results"""
        # Create host input and output arrays
        host_inputs = []
        for inp_tensor in test_case.input_shapes():
            host_inputs.append(np.empty(inp_tensor.shape(), dtype=inp_tensor.dtype))
        
        host_outputs = []
        for out_tensor in test_case.output_shapes():
            host_outputs.append(np.empty(out_tensor.shape(), dtype=out_tensor.dtype))
        
        # Prepare data
        test_case.prepare_data(host_inputs, host_outputs)
        sizes = test_case.get_sizes()
        
        # Create device inputs
        device_inputs = []
        for host_inp in host_inputs:
            device_inp = cuda.mem_alloc(host_inp.nbytes)
            cuda.memcpy_htod(device_inp, host_inp)
            device_inputs.append(device_inp)
        
        # Create device outputs
        device_outputs = []
        reference_outputs = []
        for host_out in host_outputs:
            device_out = cuda.mem_alloc(host_out.nbytes)
            reference_outputs.append(np.empty_like(host_out))
            device_outputs.append(device_out)
        
        # Launch the kernel
        start_time = time.time()
        test_case.launch_kernel(device_inputs, device_outputs, sizes, self.kernel_func)
        cuda.Context.synchronize()
        elapsed_time = time.time() - start_time
        
        # Copy results back to host
        solution_outputs = []
        for i, device_out in enumerate(device_outputs):
            solution_out = np.empty_like(host_outputs[i])
            cuda.memcpy_dtoh(solution_out, device_out)
            solution_outputs.append(solution_out)
        
        # Run reference implementation if provided
        if self.reference_func is not None:
            # Convert NumPy arrays to PyTorch tensors
            torch_inputs = [torch.from_numpy(inp).to('cuda') for inp in host_inputs]
            
            # Run reference implementation
            torch_result = self.reference_func(*torch_inputs)
            
            # Convert result to list if it's a single tensor
            if not isinstance(torch_result, tuple) and not isinstance(torch_result, list):
                torch_result = [torch_result]
            
            # Convert back to NumPy for comparison
            for i, torch_out in enumerate(torch_result):
                reference_outputs[i] = torch_out.cpu().numpy()
            
            # Check results
            passed = True
            max_diff = 0.0
            for i, (solution, reference) in enumerate(zip(solution_outputs, reference_outputs)):
                diff = np.abs(solution - reference).max()
                max_diff = max(max_diff, diff)
                if diff > tolerance:
                    passed = False
                    break
        else:
            # If no reference function is provided, set passed to False
            passed = False
            max_diff = 0.0
        
        # Free memory
        for device_inp in device_inputs:
            device_inp.free()
        for device_out in device_outputs:
            device_out.free()
        
        return {
            'passed': passed,
            'name': test_case.get_name(),
            'max_diff': max_diff,
            'execution_time': elapsed_time
        }