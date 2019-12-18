from setuptools import setup, Extension, find_packages

setup(
    name="gancon", 
    version="1.0.0", 
    description="GANcon: protein contact map prediction with deep generative adversarial network", 
    author="HIlab", 
    py_modules=['gancon.gancon'],
    packages=find_packages(),
    package_data={
            'gancon.lib': [
                'gancon/lib/alnstats'
            ]
        },
    include_package_data=True,
    zip_safe=False
)
