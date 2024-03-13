from setuptools import setup, find_packages

setup(
    name='optibeam',
    version='0.1.7',
    author='Andrew Xu',
    author_email='qiyuanxu95@gmail.com',
    description='Python modules for image processing, data analysis and machine learning in accelerator physics',
    url='https://github.com/Andrew-XQY/OptiBeam',
    packages=find_packages(),
    install_requires=[
        # List project's dependencies here.
        'numpy>=1.18.1'
    ],
    classifiers=[
        # Choose license as wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.6',
)
