from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="xielu",
    version="0.1.0",
    packages=["xielu"],
    package_dir={"xielu": "."},
    ext_modules=[
        CUDAExtension(
            name="_xielu",
            sources=["src/binding.cpp", "src/xielu.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
