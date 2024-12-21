"""
Check resources requested by kernel
"""

import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.autoinit

kernel_path = './flashforward.cpp'
with open(kernel_path, 'r') as f:
    kernel = f.read()

mod = compiler.SourceModule(kernel, options=["-Xptxas -v"])
print("Loaded mod successfully!")
