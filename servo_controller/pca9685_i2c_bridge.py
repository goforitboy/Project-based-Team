#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed under the
# terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

###
# 适配PCA9685的Gen3 I2C桥接示例
# 功能：初始化PCA9685、设置PWM频率、输出指定PWM值
###

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time

from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.InterconnectConfigClientRpc import InterconnectConfigClient
from kortex_api.autogen.messages import Common_pb2, InterconnectConfig_pb2
from kortex_api.Exceptions import KServerException

class I2CBridge:
    def __init__(self, router):
        self.router = router
        self.device_manager = DeviceManagerClient(self.router)
        self.interconnect_config = InterconnectConfigClient(self.router)
        self.interconnect_device_id = self.GetDeviceIdFromDevType(Common_pb2.INTERCONNECT, 0)
        if self.interconnect_device_id is None:
            print("Could not find the Interconnect in the device list, exiting...")
            sys.exit(0)

    def GetDeviceIdFromDevType(self, device_type, device_index = 0):
        devices = self.device_manager.ReadAllDevices()
        current_index = 0
        for device in devices.device_handle:
            if device.device_type == device_type:
                if current_index == device_index:
                    print(f"Found the Interconnect on device identifier {device.device_identifier}")
                    return device.device_identifier
                current_index += 1
        return None

    def WriteValue(self, device_address, data, timeout_ms):
        i2c_write_parameter = InterconnectConfig_pb2.I2CWriteParameter()
        i2c_write_parameter.device = InterconnectConfig_pb2.I2C_DEVICE_EXPANSION
        i2c_write_parameter.device_address = device_address
        bytesData = bytes(data)
        i2c_write_parameter.data.data = bytesData
        i2c_write_parameter.data.size = len(bytesData)
        i2c_write_parameter.timeout = timeout_ms
        return self.interconnect_config.I2CWrite(i2c_write_parameter, deviceId=self.interconnect_device_id)

    def ReadValue(self, device_address, bytes_to_read, timeout_ms):
        i2c_read_request = InterconnectConfig_pb2.I2CReadParameter()
        i2c_read_request.device = InterconnectConfig_pb2.I2C_DEVICE_EXPANSION
        i2c_read_request.device_address = device_address
        i2c_read_request.size = bytes_to_read
        i2c_read_request.timeout = timeout_ms
        read_result = self.interconnect_config.I2CRead(i2c_read_request, deviceId=self.interconnect_device_id)
        data = read_result.data
        print(f"We were supposed to read {bytes_to_read} bytes and we read {read_result.size} bytes.")
        print(f"The data is : {ord(data):b}")
        return read_result.data

    def Configure(self, is_enabled, mode, addressing):
        I2CConfiguration = InterconnectConfig_pb2.I2CConfiguration()
        I2CConfiguration.device = InterconnectConfig_pb2.I2C_DEVICE_EXPANSION
        I2CConfiguration.enabled = is_enabled
        I2CConfiguration.mode = mode
        I2CConfiguration.addressing = addressing
        self.interconnect_config.SetI2CConfiguration(I2CConfiguration, deviceId=self.interconnect_device_id)

# ========== PCA9685专用操作函数 (已修正) ==========

def set_pca9685_pwm_freq(bridge, slave_addr, freq_hz):
    """
    设置PCA9685的PWM输出频率。
    """
    # 计算预分频值
    prescale_val = 25000000.0 / (4096.0 * freq_hz) - 1
    prescale = int(round(prescale_val))
    
    # 1. 进入休眠模式
    bridge.WriteValue(slave_addr, [0x00], 100)
    time.sleep(0.005)
    old_mode = bridge.ReadValue(slave_addr, 1, 100)
    new_mode = (ord(old_mode) & 0x7F) | 0x10  # 设置SLEEP位
    bridge.WriteValue(slave_addr, [0x00, new_mode], 100)
    time.sleep(0.005)
    
    # 2. 设置预分频器
    bridge.WriteValue(slave_addr, [0xFE, prescale], 100)
    time.sleep(0.005)
    
    # 3. 唤醒并开启自动地址递增 (AI = 1)
    bridge.WriteValue(slave_addr, [0x00, 0x20], 100)  # MODE1 = 0x20 (AI=1, SLEEP=0)
    time.sleep(0.005)

    # 4. 设置MODE2寄存器
    bridge.WriteValue(slave_addr, [0x01, 0x04], 100)
    time.sleep(0.005)
    
    print(f"PCA9685 PWM频率已设置为 {freq_hz}Hz (预分频值：{prescale})")

def set_pca9685_channel_pwm(bridge, slave_addr, channel, pwm_value):
    """
    设置PCA9685指定通道的PWM脉冲宽度。
    :param pwm_value: PWM脉冲宽度对应的计数值 (0-4095)
    """
    # 确保pwm_value在有效范围内
    pwm_value = max(0, min(4095, pwm_value))
    
    # 计算通道寄存器基地址
    reg_base = 0x06 + 4 * channel
    
    # 设置ON为0，OFF为目标值
    write_data = [
        reg_base,             # 指向ON_L寄存器
        0x00,                 # ON_L = 0
        0x00,                 # ON_H = 0
        pwm_value & 0xFF,     # OFF_L = 低8位
        (pwm_value >> 8) & 0x0F # OFF_H = 高4位
    ]
    bridge.WriteValue(slave_addr, write_data, 100)
    # 为了和你之前的日志格式保持一致，这里保留print
    duty_cycle_percent = (pwm_value / 4095.0) * 100
    print(f"PCA9685通道{channel}占空比已设置为 {duty_cycle_percent:.1f}% (PWM值：{pwm_value})")

# ... main函数保持不变 ...
def main():
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        bridge = I2CBridge(router)
        slave_address = 0x40
        bridge.Configure(True, InterconnectConfig_pb2.I2C_MODE_FAST, InterconnectConfig_pb2.I2C_DEVICE_ADDRESSING_7_BITS)
        time.sleep(1)
        print("I2C bridge object initialized")

        print("\nReading MODE1 register from PCA9685...")
        try:
            bridge.WriteValue(slave_address, [0x00], 100)
            time.sleep(0.5)
            bridge.ReadValue(slave_address, 1, 100)
            time.sleep(0.5)
        except Exception as ex:
            print(f"Error : {ex}")
                
        print("\nConfiguring PCA9685 PWM frequency...")
        try:
            set_pca9685_pwm_freq(bridge, slave_address, 50)
            time.sleep(0.5)
            # 注意：这里调用时传入的是PWM计数值，不是占空比
            set_pca9685_channel_pwm(bridge, slave_address, 0, 2048) # 50% 占空比
            time.sleep(0.5)
        except Exception as ex:
            print(f"Error : {ex}")

        print("\nReading MODE1 register from PCA9685 after configuration...")
        try:
            bridge.WriteValue(slave_address, [0x00], 100)
            time.sleep(0.5)
            bridge.ReadValue(slave_address, 1, 100)
            time.sleep(0.5)
        except Exception as ex:
            print(f"Error : {ex}")

if __name__ == "__main__":
    main()
