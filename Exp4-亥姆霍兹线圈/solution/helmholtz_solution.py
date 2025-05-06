import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

# 物理常数
mu0 = 4*np.pi*1e-7  # 真空磁导率

class HelmholtzCoil:
    def __init__(self, radius, current, n_turns=100):
        """初始化亥姆霍兹线圈
        
        参数：
        radius: 线圈半径（米）
        current: 电流（安培）
        n_turns: 线圈匝数
        """
        self.radius = radius
        self.current = current
        self.n_turns = n_turns
        
        # 两个线圈的位置，间距等于半径
        self.z1 = -radius/2
        self.z2 = radius/2
    
    def k_parameter(self, r, z, z0):
        """计算椭圆积分的参数k
        
        参数：
        r: 径向距离
        z: 轴向位置
        z0: 线圈中心的z坐标
        
        返回：
        float: 椭圆积分参数k
        """
        Z = z - z0
        gamma = (1 + r/self.radius)**2 + (Z/self.radius)**2
        return np.sqrt(4*r/(self.radius*gamma))
    
    def single_coil_field(self, r, z, z0):
        """计算单个线圈在点(r,z)处产生的磁场
        
        使用完全椭圆积分计算磁场。对于r=0的情况使用极限形式。
        
        参数：
        r: 径向距离
        z: 轴向位置
        z0: 线圈中心的z坐标
        
        返回：
        tuple: (Br, Bz) 磁场的径向和轴向分量
        """
        Z = z - z0
        
        if r < 1e-10:  # 处理轴上点
            # 轴上点的解析解
            Br = 0
            Bz = mu0 * self.current * self.n_turns * self.radius**2 /\
                 (2 * (self.radius**2 + Z**2)**(3/2))
            return Br, Bz
        
        # 计算椭圆积分参数
        k = self.k_parameter(r, z, z0)
        k2 = k*k
        
        # 计算完全椭圆积分
        K = ellipk(k2)
        E = ellipe(k2)
        
        # 计算公共因子
        prefactor = mu0 * self.current * self.n_turns / (2*np.pi)
        gamma = (1 + r/self.radius)**2 + (Z/self.radius)**2
        sqrt_gamma = np.sqrt(gamma)
        
        # 计算径向分量
        Br = prefactor * (Z/(r*sqrt_gamma)) * \
             (K + (self.radius**2 - r**2 - Z**2)/((self.radius - r)**2 + Z**2)*E)
        
        # 计算轴向分量
        Bz = prefactor / sqrt_gamma * \
             (K - (self.radius**2 + r**2 + Z**2)/((self.radius - r)**2 + Z**2)*E)
        
        return Br, Bz
    
    def total_field(self, r, z):
        """计算两个线圈在点(r,z)处产生的总磁场
        
        参数：
        r: 径向距离
        z: 轴向位置
        
        返回：
        tuple: (Br, Bz) 总磁场的径向和轴向分量
        """
        # 计算两个线圈的磁场
        Br1, Bz1 = self.single_coil_field(r, z, self.z1)
        Br2, Bz2 = self.single_coil_field(r, z, self.z2)
        
        # 叠加磁场
        return Br1 + Br2, Bz1 + Bz2
    
    def plot_field_strength(self):
        """绘制磁场强度分布图"""
        # 创建网格点
        r = np.linspace(0, 2*self.radius, 100)
        z = np.linspace(-2*self.radius, 2*self.radius, 100)
        R, Z = np.meshgrid(r, z)
        
        # 计算每个点的磁场
        Br = np.zeros_like(R)
        Bz = np.zeros_like(R)
        B_magnitude = np.zeros_like(R)
        
        for i in range(len(r)):
            for j in range(len(z)):
                Br[j,i], Bz[j,i] = self.total_field(R[j,i], Z[j,i])
                B_magnitude[j,i] = np.sqrt(Br[j,i]**2 + Bz[j,i]**2)
        
        # 绘制磁场强度等值线
        plt.figure(figsize=(12, 8))
        levels = np.linspace(0, B_magnitude.max(), 20)
        plt.contourf(Z/self.radius, R/self.radius, B_magnitude*1e6, levels=levels)
        plt.colorbar(label='磁感应强度 (μT)')
        
        # 绘制线圈位置
        plt.plot([self.z1/self.radius, self.z1/self.radius], [0, 1], 'r-', linewidth=2)
        plt.plot([self.z2/self.radius, self.z2/self.radius], [0, 1], 'r-', linewidth=2)
        
        plt.xlabel('z/R')
        plt.ylabel('r/R')
        plt.title('亥姆霍兹线圈磁场强度分布')
        plt.grid(True)
        plt.show()
    
    def plot_field_lines(self):
        """绘制磁力线分布图"""
        # 创建网格点
        r = np.linspace(0, 2*self.radius, 20)
        z = np.linspace(-2*self.radius, 2*self.radius, 40)
        R, Z = np.meshgrid(r, z)
        
        # 计算每个点的磁场
        Br = np.zeros_like(R)
        Bz = np.zeros_like(R)
        
        for i in range(len(r)):
            for j in range(len(z)):
                Br[j,i], Bz[j,i] = self.total_field(R[j,i], Z[j,i])
        
        # 绘制磁力线
        plt.figure(figsize=(12, 8))
        plt.streamplot(Z/self.radius, R/self.radius, Bz, Br, density=2)
        
        # 绘制线圈位置
        plt.plot([self.z1/self.radius, self.z1/self.radius], [0, 1], 'r-', linewidth=2)
        plt.plot([self.z2/self.radius, self.z2/self.radius], [0, 1], 'r-', linewidth=2)
        
        plt.xlabel('z/R')
        plt.ylabel('r/R')
        plt.title('亥姆霍兹线圈磁力线分布')
        plt.grid(True)
        plt.show()
    
    def analyze_uniformity(self):
        """分析中心区域的磁场均匀性"""
        # 计算中心点的磁场
        B0 = self.total_field(0, 0)[1]  # 只取z分量
        
        # 在中心区域进行扫描
        r_scan = np.linspace(0, 0.2*self.radius, 20)
        z_scan = np.linspace(-0.2*self.radius, 0.2*self.radius, 40)
        R, Z = np.meshgrid(r_scan, z_scan)
        
        # 计算相对偏差
        deviation = np.zeros_like(R)
        for i in range(len(r_scan)):
            for j in range(len(z_scan)):
                Br, Bz = self.total_field(R[j,i], Z[j,i])
                B = np.sqrt(Br**2 + Bz**2)
                deviation[j,i] = abs(B - B0)/B0 * 100  # 百分比偏差
        
        # 绘制偏差分布
        plt.figure(figsize=(12, 8))
        levels = np.linspace(0, deviation.max(), 20)
        plt.contourf(Z/self.radius, R/self.radius, deviation, levels=levels)
        plt.colorbar(label='相对偏差 (%)')
        
        plt.xlabel('z/R')
        plt.ylabel('r/R')
        plt.title('中心区域磁场均匀性分析')
        plt.grid(True)
        plt.show()
        
        # 输出最大偏差
        max_dev = deviation.max()
        print(f"中心区域内的最大磁场偏差：{max_dev:.2f}%")

def test_coil():
    """测试亥姆霍兹线圈的磁场计算"""
    # 创建一个亥姆霍兹线圈实例
    coil = HelmholtzCoil(radius=0.1, current=1.0, n_turns=100)
    
    # 测试轴上点的磁场
    test_points = [
        (0, 0),      # 中心点
        (0, 0.05),   # 轴上点
        (0.05, 0),   # 径向点
        (0.05, 0.05) # 一般点
    ]
    
    print("磁场测试：")
    print("-" * 50)
    print("位置 (r,z)\t\t磁场 (T)\t\t方向 (度)")
    print("-" * 50)
    
    for r, z in test_points:
        Br, Bz = coil.total_field(r, z)
        B = np.sqrt(Br**2 + Bz**2)
        theta = np.arctan2(Br, Bz) * 180/np.pi
        print(f"({r:.2f}, {z:.2f})\t\t{B:.3e}\t\t{theta:.1f}")

def main():
    # 创建线圈实例
    coil = HelmholtzCoil(radius=0.1, current=1.0)
    
    # 绘制磁场强度分布
    coil.plot_field_strength()
    
    # 绘制磁力线分布
    coil.plot_field_lines()
    
    # 分析磁场均匀性
    coil.analyze_uniformity()
    
    # 运行测试
    test_coil()

if __name__ == '__main__':
    main()