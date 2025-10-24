"""
CUDA-accelerated EVM components

GPU kernels for <10ms latency video magnification.
Falls back gracefully to CPU implementation if CUDA not available.
"""

from .gpu_pipeline import GPUEVMPipeline, test_gpu_pipeline

__all__ = ['GPUEVMPipeline', 'test_gpu_pipeline']
