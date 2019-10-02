from setuptools import setup
import setuptools
print(setuptools.find_packages(exclude=['doc', 'data']))
setup(
    name='pysubgroup',
    version='0.6.1',
    packages=['pysubgroup'],
    package_data={'pysubgroup':['data/credit-g.arff','data/titanic.csv']},
    url='http://florian.lemmerich.net/pysubgroup',
    license='',
    author='Florian Lemmerich',
    author_email='florian@lemmerich.net',
    description='pysubgroup is a Python library for the data analysis task of subgroup discovery.',
    install_requires=[
              'pandas', 'scipy', 'numpy', 'matplotlib'
          ],
    tests_require=['pytest'],
    python_requires='>=3.6'
)
