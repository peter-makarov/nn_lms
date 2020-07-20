from setuptools import setup

setup(name='nn_lm',
      version='0.1',
      description='Word and character neural network language models with flair and pytorch',
      url='',
      author='Peter Makarov & Simon Clematide',
      author_email='makarov@cl.uzh.ch',
      license='MIT',
      packages=['nn_lm'],
      install_requires=[
          "flair==0.4.1",
          "numpy==1.16.3",
          "sacred==0.7.4",
          "torch==1.0.1.post2",
      ],
      zip_safe=True)

