import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理常数
G = 6.67430e-11  # 引力常数

class GravityPlate:
    def __init__(self, length, width, density):
        """初始化均匀薄片
        
        参数：
        length: 薄片长度（米）
        width: 薄片宽度（米）
        density: 面密度（kg/m²）
        """
        self.length = length
        self.width = width
        self.density = density
        self.G = G
    
    def gauss_quadrature(self, f, a, b, n):
        """实现高斯-勒让德积分
        
        参数：
        f: 被积函数
        a, b: 积分区间
        n: 高斯点数
        
        返回：
        float: 积分结果
        """
        # 获取高斯点和权重
        x, w = np.polynomial.legendre.leggauss(n)
        
        # 变换到积分区间
        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        
        # 计算积分
        return 0.5 * (b - a) * np.sum(w * f(t))
    
    def gravity_element(self, x, y, z):
        """计算单个面元对场点的引力
        
        参数：
        x, y: 面元的坐标
        z: 场点的高度
        
        返回：
        float: 引力大小
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        return self.G * self.density * z / r**3
    
    def total_gravity(self, z, n_points=20):
        """计算高度z处的总引力
        
        使用二重高斯积分计算总引力。
        
        参数：
        z: 场点的高度
        n_points: 每个维度的高斯点数
        
        返回：
        float: 总引力大小
        """
        # 定义x方向的被积函数
        def fx(x):
            def fy(y):
                return self.gravity_element(x, y, z)
            # 对y积分
            return self.gauss_quadrature(fy, -self.width/2, self.width/2, n_points)
        
        # 对x积分
        return self.gauss_quadrature(fx, -self.length/2, self.length/2, n_points)
    
    def gravity_field(self, x, y, z):
        """计算空间点(x,y,z)处的引力场
        
        参数：
        x, y, z: 场点坐标
        
        返回：
        tuple: (Fx, Fy, Fz) 引力的三个分量
        """
        def integrand_x(x_prime):
            def integrand_y(y_prime):
                dx = x - x_prime
                dy = y - y_prime
                r = np.sqrt(dx**2 + dy**2 + z**2)
                return self.G * self.density / r**3 * np.array([dx, dy, z])
            return self.gauss_quadrature(integrand_y, -self.width/2, self.width/2, 20)
        
        result = self.gauss_quadrature(integrand_x, -self.length/2, self.length/2, 20)
        return tuple(result)
    
    def plot_gravity_field(self):
        """绘制引力场分布图"""
        # 创建网格点
        x = np.linspace(-2*self.length, 2*self.length, 20)
        y = np.linspace(-2*self.width, 2*self.width, 20)
        z = self.length  # 固定高度
        X, Y = np.meshgrid(x, y)
        
        # 计算每个点的引力场
        Fx = np.zeros_like(X)
        Fy = np.zeros_like(X)
        Fz = np.zeros_like(X)
        F_mag = np.zeros_like(X)
        
        for i in range(len(x)):
            for j in range(len(y)):
                Fx[j,i], Fy[j,i], Fz[j,i] = self.gravity_field(X[j,i], Y[j,i], z)
                F_mag[j,i] = np.sqrt(Fx[j,i]**2 + Fy[j,i]**2 + Fz[j,i]**2)
        
        # 绘制引力场
        plt.figure(figsize=(12, 8))
        plt.streamplot(X, Y, Fx, Fy, color=F_mag, cmap='viridis')
        plt.colorbar(label='引力场强度 (N)')
        
        # 绘制薄片位置
        rect = plt.Rectangle((-self.length/2, -self.width/2),
                           self.length, self.width,
                           fill=False, color='r')
        plt.gca().add_patch(rect)
        
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'高度z={z}m处的引力场分布')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    
    def plot_gravity_height(self):
        """绘制引力随高度的变化图"""
        # 计算不同高度的引力
        z = np.logspace(-2, 2, 100) * self.length
        F = np.array([self.total_gravity(h) for h in z])
        
        # 计算理论极限（远场近似）
        m = self.density * self.length * self.width
        F_point = self.G * m / z**2
        
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.loglog(z, F, 'b-', label='数值计算')
        plt.loglog(z, F_point, 'r--', label='点质量近似')
        
        plt.xlabel('高度 (m)')
        plt.ylabel('引力 (N)')
        plt.title('引力随高度的变化关系')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def analyze_accuracy(self):
        """分析数值计算的精度"""
        # 测试不同高斯点数的结果
        n_points = [4, 8, 16, 32, 64]
        z = self.length  # 选择一个典型高度
        
        results = [self.total_gravity(z, n) for n in n_points]
        errors = np.abs(np.array(results[1:]) - np.array(results[:-1]))
        
        # 绘制收敛性分析
        plt.figure(figsize=(10, 6))
        plt.loglog(n_points[:-1], errors, 'bo-')
        
        plt.xlabel('高斯点数')
        plt.ylabel('相邻阶数结果的差异')
        plt.title('数值计算精度分析')
        plt.grid(True)
        plt.show()
        
        # 输出收敛性数据
        print("\n精度分析：")
        print("-" * 50)
        print("高斯点数\t\t计算结果\t\t相对误差")
        print("-" * 50)
        
        for i, n in enumerate(n_points):
            result = results[i]
            if i > 0:
                rel_error = abs(result - results[i-1])/results[i-1]
                print(f"{n}\t\t\t{result:.6e}\t\t{rel_error:.2e}")
            else:
                print(f"{n}\t\t\t{result:.6e}\t\t---")

def test_plate():
    """测试均匀薄片的引力计算"""
    # 创建一个均匀薄片实例
    plate = GravityPlate(length=1.0, width=1.0, density=1.0)
    
    # 测试不同高度的引力
    test_heights = [0.1, 0.5, 1.0, 2.0]
    
    print("引力测试：")
    print("-" * 50)
    print("高度 (m)\t\t引力 (N)")
    print("-" * 50)
    
    for z in test_heights:
        F = plate.total_gravity(z)
        print(f"{z:.1f}\t\t\t{F:.3e}")

def main():
    # 创建薄片实例
    plate = GravityPlate(length=1.0, width=1.0, density=1.0)
    
    # 绘制引力场分布
    plate.plot_gravity_field()
    
    # 绘制引力-高度关系
    plate.plot_gravity_height()
    
    # 分析计算精度
    plate.analyze_accuracy()
    
    # 运行测试
    test_plate()

if __name__ == '__main__':
    main()