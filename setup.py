import io
from setuptools import setup, find_packages

def readme():
    with io.open('README.md', encoding='utf-8') as f:
        return f.read()

setup(name='tensorstock',
      version='0.1.0',
      install_requires=['linora>=1.0.0'],
      description='',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/Hourout/tensorseries',
      author='JinQing Lee',
      author_email='hourout@163.com',
      keywords=['stock'],
      license='Apache License Version 2.0',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'
      ],
      packages=find_packages(),
      zip_safe=False)
