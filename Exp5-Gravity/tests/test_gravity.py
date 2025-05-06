import numpy as np
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生的实现
from template import GravityPlate

@pytest.fixture
def plate():
    """创建测试用的薄片实例"""
    return GravityPlate(length=1.0, width=1.0, density=1.0)

def test_initialization(plate):
    """测试薄片初始化"""
    assert plate.length == 1.0
    assert plate.width == 1.0
    assert plate.density == 1.0

def test_gauss_quadrature(plate):
    """测试高斯积分的实现"""
    # 测试简单函数的积分
    f = lambda x: x**2
    result = plate.gauss_quadrature(f, 0, 1, 10)
    assert abs(result - 1/3) < 1e-10
    
    # 测试三角函数的积分
    f = lambda x: np.sin(x)
    result = plate.gauss_quadrature(f, 0, np.pi, 20)
    assert abs(result - 2) < 1e-10

def test_gravity_element(plate):
    """测试单个面元的引力计算"""
    # 测试对称性
    g1 = plate.gravity_element(0.1, 0.1, 0.1)
    g2 = plate.gravity_element(-0.1, 0.1, 0.1)
    g3 = plate.gravity_element(0.1, -0.1, 0.1)
    g4 = plate.gravity_element(-0.1, -0.1, 0.1)
    
    # 由于对称性，这些点的引力应该相等
    assert abs(g1 - g2) < 1e-10
    assert abs(g1 - g3) < 1e-10
    assert abs(g1 - g4) < 1e-10
    
    # 测试距离依赖性
    g_near = plate.gravity_element(0, 0, 0.1)
    g_far = plate.gravity_element(0, 0, 0.2)
    assert g_near > g_far  # 近处引力应大于远处

def test_total_gravity(plate):
    """测试总引力的计算"""
    # 测试不同高度的引力
    heights = [0.1, 0.2, 0.4, 0.8]
    forces = [plate.total_gravity(z) for z in heights]
    
    # 验证引力随距离衰减
    for i in range(len(forces)-1):
        assert forces[i] > forces[i+1]
    
    # 验证远场极限
    z_far = 10.0  # 足够远的距离
    F = plate.total_gravity(z_far)
    # 计算等效质点的引力
    m = plate.density * plate.length * plate.width
    F_point = plate.G * m / z_far**2
    # 远场应接近点质量的引力
    assert abs(F/F_point - 1) < 0.01

def test_numerical_accuracy(plate):
    """测试数值计算的精度"""
    # 测试不同高斯点数的结果
    z = 0.5
    F1 = plate.total_gravity(z, n_points=10)
    F2 = plate.total_gravity(z, n_points=20)
    F3 = plate.total_gravity(z, n_points=40)
    
    # 结果应该随点数增加而收敛
    diff12 = abs(F2 - F1)
    diff23 = abs(F3 - F2)
    assert diff23 < diff12  # 高阶结果差异应更小

def test_symmetry_properties(plate):
    """测试引力场的对称性"""
    # 在不同位置计算引力
    z = 0.5
    x_vals = [-0.2, 0.0, 0.2]
    y_vals = [-0.2, 0.0, 0.2]
    
    forces = np.zeros((3, 3))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            forces[i,j] = plate.gravity_field(x, y, z)
    
    # 验证中心对称性
    assert abs(forces[0,0] - forces[2,2]) < 1e-10
    assert abs(forces[0,2] - forces[2,0]) < 1e-10
    
    # 验证轴对称性
    assert abs(forces[0,1] - forces[2,1]) < 1e-10
    assert abs(forces[1,0] - forces[1,2]) < 1e-10

if __name__ == '__main__':
    pytest.main([__file__])