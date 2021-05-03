import subprocess
from distutils.command.build_ext import build_ext
from distutils.extension import Extension

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
from distutils.command.build_py import build_py as _build_py
import sys, os.path

from gibson.assets.assets_manager import AssetsManager

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
        build_args += ['-j2']

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

setup(name='gibson',
    version='0.4.0',
    description='Real Environment Developed by Stanford University',
    url='https://github.com/fxia22/realenv',
    author='Stanford University',
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
            'numpy>=1.10.4',
            'pyglet>=1.2.0',
            'gym==0.9.4',
            'Pillow>=3.3.0',
            'PyYAML>=3.12',
            'numpy>=1.13',
            'pybullet==1.9.4',
            'transforms3d>=0.3.1',
            'tqdm >= 4',
            'pyzmq>=16.0.2',
            'Pillow>=4.2.1',
            'matplotlib>=2.1.0',
            'mpi4py>=2.0.0',
            'cloudpickle>=0.4.1',
            'pygame==1.9.6',
            'opencv-python',
            'torchvision==0.2.0',
            'aenum',
            'imageio',
    ],
    include_package_data=True,
    tests_require=[],
    entry_points={
          'console_scripts': [
              'gibson-set-assets-path = gibson.assets.assets_actions:set_assets_path',
              'gibson-download-assets-core = gibson.assets.assets_actions:download_assets_core',
          ],
    },
    ext_modules=[CMakeExtension('gibson/core/channels', source_dir='gibson/core/channels')],
    cmdclass=dict(build_ext=CMakeBuild),
    # cmdclass={
    #    'install': PostInstallCommand
    #}
)

# check_call("bash realenv/envs/build.sh".split())
