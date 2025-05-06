import numpy as np
import matplotlib.pyplot as plt

def original_integrand(x, a):
    """伽马函数的原始被积函数：x^(a-1) * e^(-x)
    
    这个函数在x=a-1处达到最大值，这个性质对于后续的变量代换很重要。
    为了避免数值计算中的上溢和下溢，对不同的x值范围采用不同的计算策略。
    
    参数：
    x : float 或 numpy.ndarray
        积分变量
    a : float
        伽马函数的参数
    
    返回：
    float 或 numpy.ndarray：被积函数的值
    """
    if isinstance(x, np.ndarray):
        # 处理数组输入
        result = np.zeros_like(x, dtype=float)
        # 对于小的x值，使用对数计算避免下溢
        small_x = x < 0.1
        if np.any(small_x):
            log_result = (a-1) * np.log(x[small_x]) - x[small_x]
            result[small_x] = np.exp(log_result)
        # 对于正常范围的x值，直接计算
        normal_x = ~small_x
        if np.any(normal_x):
            result[normal_x] = x[normal_x]**(a-1) * np.exp(-x[normal_x])
        return result
    else:
        # 处理标量输入
        if x < 0.1:
            log_result = (a-1) * np.log(x) - x
            return np.exp(log_result)
        return x**(a-1) * np.exp(-x)

def plot_integrand():
    """绘制不同a值的被积函数
    
    生成一个专业的图表，展示a=2,3,4时被积函数的形状。
    包括适当的标签、图例和网格线。
    """
    x = np.linspace(0.01, 5, 500)
    a_values = [2, 3, 4]
    
    plt.figure(figsize=(10, 6))
    for a in a_values:
        y = original_integrand(x, a)
        plt.plot(x, y, label=f'a={a}')
        # 标记最大值点
        x_max = a - 1
        y_max = original_integrand(x_max, a)
        plt.plot(x_max, y_max, 'o')
        plt.annotate(f'max at x={x_max}', 
                     xy=(x_max, y_max),
                     xytext=(x_max+0.2, y_max+0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel('x')
    plt.ylabel('f(x) = x^(a-1) * e^(-x)')
    plt.title('伽马函数的被积函数')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def find_optimal_c(a):
    """计算最优的c值，使得变换后的被积函数峰值位于z=1/2
    
    当x = a-1时被积函数达到最大值，我们希望这个点对应z = 1/2。
    根据变换公式 z = x/(c+x)，可以解出：
    1/2 = (a-1)/(c+a-1)
    因此 c = a-1
    
    参数：
    a : float
        伽马函数的参数
    
    返回：
    float：最优的c值
    """
    return a - 1

def transformed_integrand(z, a, c):
    """变换后的被积函数
    
    使用变换 x = cz/(1-z) 将[0,∞)变换到[0,1]。
    变换后的被积函数需要乘上Jacobian行列式 |dx/dz| = c/(1-z)^2。
    
    参数：
    z : float 或 numpy.ndarray
        新的积分变量，范围是[0,1]
    a : float
        伽马函数的参数
    c : float
        变换参数
    
    返回：
    float 或 numpy.ndarray：变换后的被积函数值
    """
    # 防止在z=1处出现数值问题
    if isinstance(z, np.ndarray):
        z = np.clip(z, 0, 0.9999999)
    else:
        z = min(z, 0.9999999)
    
    # 计算原始变量x
    x = c * z / (1 - z)
    
    # 计算原始被积函数值
    f = original_integrand(x, a)
    
    # 计算Jacobian行列式
    jacobian = c / (1 - z)**2
    
    return f * jacobian

def gauss_quadrature(f, a, b, n, *args):
    """实现高斯-勒让德积分
    
    使用高斯-勒让德求积公式计算定积分。这个方法对于光滑函数特别有效。
    
    参数：
    f : callable
        被积函数
    a, b : float
        积分区间的端点
    n : int
        高斯点的数量
    *args : tuple
        传递给被积函数的额外参数
    
    返回：
    float：积分结果
    """
    # 获取高斯-勒让德求积的节点和权重
    x, w = np.polynomial.legendre.leggauss(n)
    
    # 将[-1,1]区间映射到[a,b]区间
    t = 0.5 * (x + 1) * (b - a) + a
    
    # 计算积分
    return 0.5 * (b - a) * np.sum(w * f(t, *args))

def gamma(a):
    """计算伽马函数值
    
    使用变量代换和高斯积分方法计算伽马函数。
    变换将[0,∞)映射到[0,1]，并使得被积函数的峰值位于z=1/2附近。
    
    参数：
    a : float
        伽马函数的参数
    
    返回：
    float：伽马函数的值
    """
    # 计算最优的变换参数
    c = find_optimal_c(a)
    
    # 使用高斯积分计算（使用足够多的点以确保精度）
    n = 100
    result = gauss_quadrature(transformed_integrand, 0, 1, n, a, c)
    
    return result

def test_gamma():
    """测试伽马函数的计算结果"""
    # 测试Γ(3/2)
    result = gamma(1.5)
    expected = 0.886226925
    print(f"Γ(3/2) = {result:.6f} (expected: {expected:.6f})")
    
    # 测试整数值
    test_values = [3, 6, 10]
    print("\n测试整数值：")
    print("-" * 40)
    print("a\tΓ(a)\t\t(a-1)!")
    print("-" * 40)
    for a in test_values:
        result = gamma(a)
        factorial = np.math.factorial(a-1)
        print(f"{a}\t{result:.6e}\t{factorial}")

def main():
    # 绘制原始被积函数
    plot_integrand()
    
    # 运行测试
    test_gamma()

if __name__ == '__main__':
    main()