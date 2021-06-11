import setuptools
from setuptools import setup, find_packages, Extension
import subprocess
from distutils.command.build_ext import build_ext
from distutils.extension import Extension

from setuptools import setup, find_packages, Extension
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
from distutils.command.build_py import build_py as _build_py
import sys, os.path
from codecs import open as copen


from gibson.assets.assets_manager import AssetsManager


# Get the long description from the relevant file
here = os.path.abspath(os.path.dirname(__file__))
with copen(os.path.join(here, 'long_description.rst'), encoding='utf-8') as f:
    long_description = f.read()


class CMakeExtension(Extension):
    def __init__(self, name, source_dir=''):
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' +
            os.path.join(ext_dir, 'channels'),
            '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' +
            os.path.join(ext_dir, 'channels', 'build'),
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['-j8']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.source_dir], cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


'''
class PostInstallCommand(install):
        """Post-installation for installation mode."""
        def run(self):
                print('post installation')
                check_call("bash realenv/envs/build.sh".split())
                install.run(self)
'''
# Read requirements.txt
with open(os.path.abspath('requirements.txt'), mode='r') as f:
    requirements = [line.rstrip() for line in f]

setup(name='gibson',
    version='0.7.5',
    description='Real Environment Developed by Stanford University',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/fxia22/realenv',
    author='Stanford University',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'
      ],
    zip_safe=False,
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    tests_require=[],
    entry_points={
          'console_scripts': [
              'gibson-set-assets-path = gibson.assets.assets_actions:set_assets_path',
              'gibson-download-assets-core = gibson.assets.assets_actions:download_assets_core',
              'gibson-download-dataset = gibson.assets.assets_actions:download_dataset',
          ],
    },
    ext_modules=[CMakeExtension('GibsonChannel', source_dir='gibson/core')],
    cmdclass=dict(build_ext=CMakeBuild),
    python_requires='>2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*, < 4',
)

# check_call("bash realenv/envs/build.sh".split())
