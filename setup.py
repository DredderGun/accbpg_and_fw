from setuptools import setup, find_packages

setup(
    name='accbpg',
    version='0.2',
    packages=find_packages(exclude=['frank_wolfe_wtih_rs*']),
    license='MIT',
    description='Accelerated Bregman proximal gradient (ABPG) and Frank-Wolfe with Bregman divergence methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'scipy'],
    url='https://github.com/DredderGun/accbpg_and_fw',
    author='Lin Xiao, Vyguzov Aleksandr',
    author_email='lin.xiao@gmail.com, al.vyguzov@yandex.ru'
)
