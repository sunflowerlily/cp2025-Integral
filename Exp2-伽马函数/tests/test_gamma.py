import numpy as np
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生的实现
from template import original_integrand, transformed_integrand, find_optimal_c, gamma

def test_original_integrand():
    """测试原始被积函数的实现"""
    # 测试单个值
    assert abs(original_integrand(1.0, 2.0) - 0.36787944) < 1e-7
    
    # 测试数组输入
    x = np.array([1.0, 2.0, 3.0])
    a = 2.0
    expected = np.array([0.36787944, 0.27067057, 0.14872127])
    assert np.allclose(original_integrand(x, a), expected, rtol=1e-7)
    
    # 测试不同的a值
    assert abs(original_integrand(2.0, 3.0) - 0.54134113) < 1e-7
    assert abs(original_integrand(3.0, 4.0) - 0.44617586) < 1e-7

def test_find_optimal_c():
    """测试最优c值的计算"""
    # 对于a=2，峰值应该在x=1处
    c = find_optimal_c(2.0)
    assert abs(c - 1.0) < 1e-7
    
    # 对于a=3，峰值应该在x=2处
    c = find_optimal_c(3.0)
    assert abs(c - 2.0) < 1e-7
    
    # 对于a=4，峰值应该在x=3处
    c = find_optimal_c(4.0)
    assert abs(c - 3.0) < 1e-7

def test_transformed_integrand():
    """测试变换后的被积函数"""
    # 测试z=0.5时的值（应该接近峰值）
    a = 2.0
    c = find_optimal_c(a)
    result = transformed_integrand(0.5, a, c)
    assert result > transformed_integrand(0.3, a, c)
    assert result > transformed_integrand(0.7, a, c)
    
    # 测试边界条件
    assert np.isfinite(transformed_integrand(0.001, a, c))
    assert np.isfinite(transformed_integrand(0.999, a, c))

def test_gamma_special_values():
    """测试特殊值的伽马函数计算"""
    # 测试Γ(3/2)
    result = gamma(1.5)
    expected = 0.886226925
    assert abs(result - expected) < 1e-6
    
    # 测试整数值（应该等于阶乘）
    test_values = [(3, 2), (6, 120), (10, 362880)]
    for a, expected in test_values:
        result = gamma(a)
        assert abs(result - expected) / expected < 1e-4

def test_gamma_properties():
    """测试伽马函数的基本性质"""
    # 测试正值性
    assert gamma(2.5) > 0
    
    # 测试递增性（在x>1的区间）
    x1, x2 = 2.0, 3.0
    assert gamma(x2) > gamma(x1)
    
    # 测试计算稳定性
    results = [gamma(a) for a in np.linspace(1.5, 4.5, 10)]
    assert all(np.isfinite(results))

if __name__ == '__main__':
    pytest.main([__file__])