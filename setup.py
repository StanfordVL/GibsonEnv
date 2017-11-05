from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
from distutils.command.build_py import build_py as _build_py
import sys, os.path

'''
class PostInstallCommand(install):
		"""Post-installation for installation mode."""
		def run(self):
				print('post installation')
				check_call("bash realenv/envs/build.sh".split())
				install.run(self)
'''

setup(name='realenv',
	version='0.0.1',
	description='Real Environment Developed by Stanford University',
	url='https://github.com/fxia22/realenv',
	author='Stanford University',
	zip_safe=False,
	install_requires=[
			'numpy>=1.10.4',
			#'go-vncdriver>=0.4.19',
			'pyglet>=1.2.0',
			'gym>=0.9.2',
			'Pillow>=3.3.0',
          	'PyYAML>=3.12',
          	'numpy>=1.13',
          	'pybullet>=1.5.6',
          	'transforms3d>=0.3.1',
          	#'progressbar>=2.3',
            'tqdm >= 4',
          	'pyzmq>=16.0.2',
          	'Pillow>=4.2.1',
          	'matplotlib>=2.1.0',
            'mpi4py>=2.0.0',
            'cloudpickle>=0.4.1'
	],
	tests_require=[],
	# cmdclass={
	#	'install': PostInstallCommand
	#}
)


# check_call("bash realenv/envs/build.sh".split())
