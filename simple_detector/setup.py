from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'simple_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         (os.path.join('lib', 'python3.10', 'site-packages', 'simple_detector', 'models'), 
         glob(os.path.join('simple_detector', 'models', '*.pt'))),
    ],
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhb',
    maintainer_email='zhb@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simple_detector_node = simple_detector.simple_detector_node:main',
            'recognize_node = simple_detector.recognize_two:main',
            'calibration_node = simple_detector.calibration2:main',
        ],
    },
)