import numpy as np
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生的实现
from template import HelmholtzCoil

@pytest.fixture
def coil():
    """创建测试用的线圈实例"""
    return HelmholtzCoil(radius=0.1, current=1.0, n_turns=100)

def test_initialization(coil):
    """测试线圈初始化"""
    assert coil.radius == 0.1
    assert coil.current == 1.0
    assert coil.n_turns == 100
    # 线圈间距应等于半径
    assert abs(coil.z1 + coil.z2) < 1e-10  # 中心对称
    assert abs(coil.z2 - coil.z1 - coil.radius) < 1e-10  # 间距等于半径

def test_k_parameter(coil):
    """测试椭圆积分参数k的计算"""
    # 中心点
    k = coil.k_parameter(0, 0, coil.z1)
    assert 0 <= k <= 1  # k应该在[0,1]范围内
    
    # 远处的点
    k = coil.k_parameter(1.0, 1.0, coil.z1)
    assert 0 <= k <= 1
    assert k < 0.1  # 远处k应该很小

def test_single_coil_field(coil):
    """测试单个线圈的磁场计算"""
    # 测试轴上点
    Br, Bz = coil.single_coil_field(0, 0, coil.z1)
    assert abs(Br) < 1e-10  # 轴上径向场应为0
    assert Bz > 0  # 轴上应有z方向的场
    
    # 测试对称性
    Br1, Bz1 = coil.single_coil_field(0.05, 0, coil.z1)
    Br2, Bz2 = coil.single_coil_field(-0.05, 0, coil.z1)
    assert abs(Br1 + Br2) < 1e-10  # 径向场反对称
    assert abs(Bz1 - Bz2) < 1e-10  # 轴向场对称

def test_total_field(coil):
    """测试总磁场的计算"""
    # 测试中心点
    Br, Bz = coil.total_field(0, 0)
    assert abs(Br) < 1e-10  # 中心点径向场应为0
    assert Bz > 0  # 中心点应有z方向的场
    
    # 测试对称性
    Br1, Bz1 = coil.total_field(0.05, 0)
    Br2, Bz2 = coil.total_field(-0.05, 0)
    assert abs(Br1 + Br2) < 1e-10  # 径向场反对称
    assert abs(Bz1 - Bz2) < 1e-10  # 轴向场对称
    
    # 测试远场衰减
    B1 = np.sqrt(sum(x**2 for x in coil.total_field(0, 0.2)))
    B2 = np.sqrt(sum(x**2 for x in coil.total_field(0, 0.4)))
    assert B1 > B2  # 场强应随距离衰减

def test_field_uniformity(coil):
    """测试中心区域的磁场均匀性"""
    # 计算中心点的场强
    B0 = coil.total_field(0, 0)[1]  # 只取z分量
    
    # 在中心区域的几个点检查场强变化
    test_points = [
        (0.01, 0),    # 略微偏离轴
        (0, 0.01),    # 轴上偏移
        (0.01, 0.01)  # 一般位置
    ]
    
    for r, z in test_points:
        Br, Bz = coil.total_field(r, z)
        B = np.sqrt(Br**2 + Bz**2)
        # 场强变化应小于1%
        assert abs(B - B0)/B0 < 0.01

def test_field_scaling(coil):
    """测试磁场与电流和匝数的关系"""
    # 记录原始参数下的场强
    B0 = np.sqrt(sum(x**2 for x in coil.total_field(0, 0)))
    
    # 测试电流加倍
    coil2 = HelmholtzCoil(radius=0.1, current=2.0, n_turns=100)
    B1 = np.sqrt(sum(x**2 for x in coil2.total_field(0, 0)))
    assert abs(B1/B0 - 2) < 0.01  # 场强应该加倍
    
    # 测试匝数加倍
    coil3 = HelmholtzCoil(radius=0.1, current=1.0, n_turns=200)
    B2 = np.sqrt(sum(x**2 for x in coil3.total_field(0, 0)))
    assert abs(B2/B0 - 2) < 0.01  # 场强应该加倍

if __name__ == '__main__':
    pytest.main([__file__])