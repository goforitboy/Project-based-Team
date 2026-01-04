from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'servo_controller'

# 用glob递归匹配kortex_api下的所有文件（替代**/*通配符，解决兼容问题）
def get_kortex_api_files():
    file_paths = []
    # 递归遍历kortex_api目录下的所有文件
    for root, dirs, files in os.walk(os.path.join(package_name, 'kortex_api')):
        for file in files:
            # 转换为包内相对路径
            rel_path = os.path.relpath(os.path.join(root, file), package_name)
            file_paths.append(rel_path)
    # 去重并返回
    return list(set(file_paths))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),  # 自动识别所有子包
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rpp',
    maintainer_email='rpp@todo.todo',
    description='Servo controller via Kinova Interconnect I2C',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'servo_controller_node = servo_controller.servo_controller_node:main',
        ],
    },
    package_data={
        # 用实际匹配的文件路径替代通配符，避免语法错误
        package_name: get_kortex_api_files() + ['*.py'],
    },
    include_package_data=True,
)