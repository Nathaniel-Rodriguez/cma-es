from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='cma-es',
    version='0.1',
    description='Implements CMA-ES for python2.7 and python3.5',
    author='Nathaniel Rodriguez',
    packages=['cmaes'],
    url='https://github.com/Nathaniel-Rodriguez/cma-es.git',
    install_requires=[
          'numpy',
          'matplotlib',
          'joblib'
      ],
    include_package_data=True,
    zip_safe=False)