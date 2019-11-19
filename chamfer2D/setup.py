from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer2D',
    ext_modules=[
        CUDAExtension('chamfer2D', [
            'chamfer_cuda.cpp',
            'chamfer2D.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })