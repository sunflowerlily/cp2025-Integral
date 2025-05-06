import numpy as np
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生的实现
from template import integrand, gauss_quadrature, cv

def test_integrand():
    """测试被积函数的实现"""
    # 测试单个值
    assert abs(integrand(1.0) - 0.45431431) < 1e-7
    
    # 测试数组输入
    x = np.array([1.0, 2.0, 3.0])
    expected = np.array([0.45431431, 0.31455917, 0.21914472])
    assert np.allclose(integrand(x), expected, rtol=1e-7)
    
    # 测试边界情况
    assert np.isfinite(integrand(0.1))  # 接近0的情况
    assert np.isfinite(integrand(10.0))  # 较大值的情况

def test_gauss_quadrature():
    """测试高斯积分的实现"""
    # 测试简单函数的积分
    def f(x): return x**2
    result = gauss_quadrature(f, 0, 1, 10)
    assert abs(result - 1/3) < 1e-10
    
    # 测试指数函数的积分
    def g(x): return np.exp(-x)
    result = gauss_quadrature(g, 0, 1, 20)
    assert abs(result - (1 - np.exp(-1))) < 1e-10
    
    # 测试积分区间变换
    result = gauss_quadrature(f, -1, 1, 10)
    assert abs(result - 2/3) < 1e-10

def test_cv():
    """测试热容计算的实现"""
    # 测试低温极限
    cv_low = cv(5.0)
    assert cv_low > 0 and cv_low < 1e-19  # 低温下热容应该很小
    
    # 测试室温附近的值
    cv_room = cv(300.0)
    expected_room = 2.49e-17  # 根据理论计算的大致值
    assert abs(cv_room - expected_room) / expected_room < 0.1
    
    # 测试高温极限
    cv_high = cv(500.0)
    expected_high = 2.49e-17  # 高温极限应该接近杜隆-珀替定律的预测
    assert abs(cv_high - expected_high) / expected_high < 0.1
    
    # 测试温度序列
    temperatures = np.array([10.0, 100.0, 200.0, 400.0])
    results = np.array([cv(T) for T in temperatures])
    assert np.all(np.diff(results) >= 0)  # 热容应该随温度单调增加
    assert np.all(np.isfinite(results))  # 所有结果应该是有限值

def test_physical_constraints():
    """测试物理约束条件"""
    # 热容应该始终为正
    temperatures = np.linspace(5, 500, 20)
    for T in temperatures:
        assert cv(T) > 0
    
    # 热容应该随温度增加而增加（在低温区域）
    T1, T2 = 10.0, 50.0
    assert cv(T1) < cv(T2)
    
    # 高温下热容应该趋于定值（杜隆-珀替定律）
    cv_400 = cv(400.0)
    cv_500 = cv(500.0)
    assert abs(cv_400 - cv_500) / cv_400 < 0.1

if __name__ == '__main__':
    pytest.main([__file__])