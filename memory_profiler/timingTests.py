import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule


class MultiplicationTimeTest:
  @profile
  def __init__(self, numElements=400):
    mod = SourceModule("""
    __global__ void multiply_them(float *dest, float *a, float *b)
    {
      const int i = threadIdx.x;
      dest[i] = a[i] * b[i];
    }
    """)

    multiply_them = mod.get_function("multiply_them")

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)

    dest = numpy.zeros_like(a)
    multiply_them(
            drv.Out(dest), drv.In(a), drv.In(b),
            block=(400,1,1))


x = MultiplicationTimeTest()