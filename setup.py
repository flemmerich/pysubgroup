from setuptools import setup

setup(
    name='pysubgroup',
    version='0.5',
    packages=['', ''],
    package_dir={'': 'pysubgroup', 'test': "pysubgroup.test"},
    url='http://florian.lemmerich.net/pysubgroup',
    license='',
    author='Florian Lemmerich',
    author_email='florian@lemmerich.net',
    description='pysubgroup is a Python library for the data analysis task of subgroup discovery.',
    install_requires=[
              'pandas','scipy','numpy','matplotlib'
          ]
)
