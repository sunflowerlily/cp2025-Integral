import numpy as np
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入解决方案实现
from solution.helmholtz_solution import Helmholtz_coils
#from helmholtz import Helmholtz_coils

@pytest.fixture
def helmholtz_coils():
    """创建测试用的亥姆霍兹线圈实例"""
    return Helmholtz_coils

def test_helmholtz_coils_output_shape(helmholtz_coils):
    """测试输出形状"""
    Y, Z, By, Bz = helmholtz_coils(0.5, 0.5, 0.5)
    assert Y.shape == (25, 25, 20)
    assert Z.shape == (25, 25, 20)
    assert By.shape == (25, 25)
    assert Bz.shape == (25, 25)

def test_helmholtz_coils_symmetry(helmholtz_coils):
    """测试对称性"""
    Y, Z, By, Bz = helmholtz_coils(0.5, 0.5, 0.5)
    
    # 测试y方向对称性 - 确保比较相同大小的数组
    mid = By.shape[1] // 2
    left_half = By[:, :mid]
    right_half = By[:, mid+1:] if By.shape[1] % 2 != 0 else By[:, mid:]
    assert np.allclose(left_half, -right_half[:, ::-1])
    
    # 测试z方向对称性
    mid = Bz.shape[0] // 2
    top_half = Bz[:mid]
    bottom_half = Bz[mid+1:] if Bz.shape[0] % 2 != 0 else Bz[mid:]
    assert np.allclose(top_half, bottom_half[::-1])

def test_helmholtz_coils_center_field(helmholtz_coils):
    """测试中心点磁场"""
    Y, Z, By, Bz = helmholtz_coils(0.5, 0.5, 0.5)
    
    # 中心点应该在网格中间
    center_y = Y.shape[1] // 2
    center_z = Z.shape[0] // 2
    
    # 中心点径向场应为0
    assert abs(By[center_z, center_y]) < 1e-10
    # 中心点轴向场应大于0
    assert Bz[center_z, center_y] > 0

def test_helmholtz_coils_field_scaling(helmholtz_coils):
    """测试磁场与半径的关系"""
    Y1, Z1, By1, Bz1 = helmholtz_coils(0.5, 0.5, 0.5)
    Y2, Z2, By2, Bz2 = helmholtz_coils(1.0, 1.0, 1.0)
    
    # 磁场强度应与半径成反比
    assert np.allclose(By1 * 2, By2, atol=1e-2)
    assert np.allclose(Bz1 * 2, Bz2, atol=1e-2)

if __name__ == '__main__':
    pytest.main([__file__])