from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qkv_kernel",
    ext_modules=[
        CUDAExtension(
            name="qkv_kernel",
            sources=[
                "qkv_kernel.cpp",
                "qkv_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
