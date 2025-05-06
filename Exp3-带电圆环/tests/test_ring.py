import numpy as np
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入解决方案实现
#from solution.charged_ring_solution import calculate_potential_on_grid, calculate_electric_field_on_grid
from charged_ring import calculate_potential_on_grid, calculate_electric_field_on_grid

def test_potential_calculation():
    """测试电势计算函数"""
    y_coords = np.linspace(-2, 2, 5)
    z_coords = np.linspace(-2, 2, 5)
    V, y_grid, z_grid = calculate_potential_on_grid(y_coords, z_coords)
    
    # 检查返回值的形状和类型
    assert isinstance(V, np.ndarray)
    assert V.shape == (5, 5)
    assert isinstance(y_grid, np.ndarray)
    assert isinstance(z_grid, np.ndarray)
    
    # 检查电势值的合理性
    assert np.all(V >= 0)  # 电势应为非负值
    assert V.max() > V.min()  # 电势应有变化

def test_electric_field_calculation():
    """测试电场计算函数"""
    y_coords = np.linspace(-2, 2, 5)
    z_coords = np.linspace(-2, 2, 5)
    V, y_grid, z_grid = calculate_potential_on_grid(y_coords, z_coords)
    Ey, Ez = calculate_electric_field_on_grid(V, y_coords, z_coords)
    
    # 检查返回值的形状和类型
    assert isinstance(Ey, np.ndarray)
    assert isinstance(Ez, np.ndarray)
    assert Ey.shape == (5, 5)
    assert Ez.shape == (5, 5)
    
    # 检查中心点的电场
    center_idx = 2  # 中心点索引
    assert abs(Ey[center_idx, center_idx]) < 1e-10  # 中心点y方向场应为0
    assert abs(Ez[center_idx, center_idx]) >= 0  # 中心点z方向场强

if __name__ == '__main__':
    pytest.main([__file__])