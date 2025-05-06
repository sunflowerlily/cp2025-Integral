import numpy as np
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生的实现
from template import ring_element, distance, potential, electric_field

def test_ring_element():
    """测试圆环参数方程的实现"""
    # 测试特殊角度
    x, y, z = ring_element(0)
    assert np.allclose([x, y, z], [0.1, 0, 0])
    
    x, y, z = ring_element(np.pi/2)
    assert np.allclose([x, y, z], [0, 0.1, 0])
    
    x, y, z = ring_element(np.pi)
    assert np.allclose([x, y, z], [-0.1, 0, 0])
    
    # 测试数组输入
    theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    x, y, z = ring_element(theta)
    expected_x = np.array([0.1, 0, -0.1, 0])
    expected_y = np.array([0, 0.1, 0, -0.1])
    expected_z = np.zeros_like(theta)
    assert np.allclose([x, y, z], [expected_x, expected_y, expected_z])

def test_distance():
    """测试距离计算的实现"""
    # 测试圆环上的点到原点的距离
    assert abs(distance(0, 0, 0, 0) - 0.1) < 1e-10
    assert abs(distance(0, 0, 0, np.pi) - 0.1) < 1e-10
    
    # 测试轴上点的距离
    assert abs(distance(0, 0, 0.1, 0) - np.sqrt(0.02)) < 1e-10
    
    # 测试一般点的距离
    d = distance(0.1, 0.1, 0.1, 0)
    expected = np.sqrt(0.01 + 0.01 + 0.01)
    assert abs(d - expected) < 1e-10

def test_potential():
    """测试电势计算的实现"""
    # 测试对称性
    v1 = potential(0.2, 0, 0)
    v2 = potential(0, 0.2, 0)
    assert abs(v1 - v2) < 1e-10  # 由于对称性，这两点电势应相等
    
    # 测试距离依赖性
    v_near = potential(0.15, 0, 0)
    v_far = potential(0.3, 0, 0)
    assert v_near > v_far  # 近处电势应大于远处
    
    # 测试轴上点
    v_axis = potential(0, 0, 0.2)
    assert np.isfinite(v_axis)  # 轴上点应该有有限的电势值

def test_electric_field():
    """测试电场计算的实现"""
    # 测试对称性
    Ex1, Ey1, Ez1 = electric_field(0.2, 0, 0)
    Ex2, Ey2, Ez2 = electric_field(0, 0.2, 0)
    assert abs(Ex1 - Ey2) < 1e-10  # 由于对称性，这两个分量应相等
    
    # 测试轴上点
    Ex, Ey, Ez = electric_field(0, 0, 0.2)
    assert abs(Ex) < 1e-10 and abs(Ey) < 1e-10  # 轴上点的径向场应为0
    assert Ez != 0  # 轴上点应该有z方向的场
    
    # 测试场强的距离依赖性
    E1 = np.sqrt(sum(x**2 for x in electric_field(0.15, 0, 0)))
    E2 = np.sqrt(sum(x**2 for x in electric_field(0.3, 0, 0)))
    assert E1 > E2  # 近处场强应大于远处

def test_field_consistency():
    """测试电势和电场的一致性"""
    # 选择测试点
    x, y, z = 0.2, 0.1, 0.1
    h = 1e-6  # 数值微分步长
    
    # 计算电场
    Ex, Ey, Ez = electric_field(x, y, z)
    
    # 用数值微分计算梯度
    Ex_num = -(potential(x+h, y, z) - potential(x-h, y, z))/(2*h)
    Ey_num = -(potential(x, y+h, z) - potential(x, y-h, z))/(2*h)
    Ez_num = -(potential(x, y, z+h) - potential(x, y, z-h))/(2*h)
    
    # 比较结果
    assert abs(Ex - Ex_num) < 1e-5
    assert abs(Ey - Ey_num) < 1e-5
    assert abs(Ez - Ez_num) < 1e-5

if __name__ == '__main__':
    pytest.main([__file__])