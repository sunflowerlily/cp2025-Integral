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
        # 在这里初始化线圈参数
        pass
    
    def k_parameter(self, r, z, z0):
        """计算椭圆积分的参数k
        
        参数：
        r: 径向距离
        z: 轴向位置
        z0: 线圈中心的z坐标
        
        返回：
        float: 椭圆积分参数k
        """
        # 在这里计算k参数
        pass
    
    def single_coil_field(self, r, z, z0):
        """计算单个线圈在点(r,z)处产生的磁场
        
        参数：
        r: 径向距离
        z: 轴向位置
        z0: 线圈中心的z坐标
        
        返回：
        tuple: (Br, Bz) 磁场的径向和轴向分量
        """
        # 在这里计算单个线圈的磁场
        pass
    
    def total_field(self, r, z):
        """计算两个线圈在点(r,z)处产生的总磁场
        
        参数：
        r: 径向距离
        z: 轴向位置
        
        返回：
        tuple: (Br, Bz) 总磁场的径向和轴向分量
        """
        # 在这里计算总磁场
        pass
    
    def plot_field_strength(self):
        """绘制磁场强度分布图"""
        # 在这里实现磁场强度的可视化
        pass
    
    def plot_field_lines(self):
        """绘制磁力线分布图"""
        # 在这里实现磁力线的可视化
        pass
    
    def analyze_uniformity(self):
        """分析中心区域的磁场均匀性"""
        # 在这里分析磁场均匀性
        pass

def test_coil():
    """测试亥姆霍兹线圈的磁场计算"""
    # 创建一个亥姆霍兹线圈实例
    # 半径10cm，电流1A，100匝
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