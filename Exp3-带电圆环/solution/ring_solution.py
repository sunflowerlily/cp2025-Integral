import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理常数
eps0 = 8.85e-12  # 真空介电常数
q = 1e-9  # 电荷量，设为1nC
a = 0.1  # 圆环半径，单位：m

def ring_element(theta):
    """计算圆环上给定参数theta处的点的坐标
    
    使用参数方程表示圆环上的点：
    x = a * cos(theta)
    y = a * sin(theta)
    z = 0
    
    参数：
    theta : float 或 numpy.ndarray
        参数方程中的角度参数
    
    返回：
    tuple：(x, y, z) 坐标
    """
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    z = np.zeros_like(theta)
    return x, y, z

def distance(x, y, z, theta):
    """计算空间点(x,y,z)到圆环上参数为theta的点的距离
    
    参数：
    x, y, z : float
        场点坐标
    theta : float
        圆环上点的参数
    
    返回：
    float：距离值
    """
    x_ring, y_ring, z_ring = ring_element(theta)
    return np.sqrt((x - x_ring)**2 + (y - y_ring)**2 + (z - z_ring)**2)

def potential(x, y, z):
    """计算空间点(x,y,z)处的电势
    
    使用数值积分计算电势：
    V = q/(4πε₀) ∮ (1/r) dl
    
    参数：
    x, y, z : float
        场点坐标
    
    返回：
    float：电势值
    """
    # 积分参数
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    dtheta = 2*np.pi / n_points
    
    # 计算积分
    integrand = 1.0 / distance(x, y, z, theta)
    integral = np.sum(integrand) * dtheta
    
    # 计算电势
    return q/(4*np.pi*eps0) * integral/n_points

def electric_field(x, y, z):
    """计算空间点(x,y,z)处的电场
    
    使用数值微分计算电场：E = -∇V
    
    参数：
    x, y, z : float
        场点坐标
    
    返回：
    tuple：(Ex, Ey, Ez) 电场各分量
    """
    h = 1e-6  # 数值微分步长
    
    # 计算x方向的电场分量
    Ex = -(potential(x+h, y, z) - potential(x-h, y, z))/(2*h)
    
    # 计算y方向的电场分量
    Ey = -(potential(x, y+h, z) - potential(x, y-h, z))/(2*h)
    
    # 计算z方向的电场分量
    Ez = -(potential(x, y, z+h) - potential(x, y, z-h))/(2*h)
    
    return Ex, Ey, Ez

def plot_potential_contour():
    """绘制xz平面上的等势线"""
    # 创建网格点
    x = np.linspace(-3*a, 3*a, 50)
    z = np.linspace(-3*a, 3*a, 50)
    X, Z = np.meshgrid(x, z)
    
    # 计算电势
    V = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(z)):
            V[j,i] = potential(X[j,i], 0, Z[j,i])
    
    # 绘制等势线
    plt.figure(figsize=(10, 8))
    plt.contour(X, Z, V, levels=20)
    plt.colorbar(label='电势 (V)')
    
    # 绘制圆环位置
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(a*np.cos(theta), np.zeros_like(theta), 'r.')
    
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title('带电圆环的等势线(y=0平面)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_field_lines():
    """绘制xz平面上的电场线"""
    # 创建网格点
    x = np.linspace(-3*a, 3*a, 20)
    z = np.linspace(-3*a, 3*a, 20)
    X, Z = np.meshgrid(x, z)
    
    # 计算电场
    Ex = np.zeros_like(X)
    Ez = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(z)):
            Ex[j,i], _, Ez[j,i] = electric_field(X[j,i], 0, Z[j,i])
    
    # 计算场强
    E = np.sqrt(Ex**2 + Ez**2)
    
    # 绘制电场箭头
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Z, Ex/E, Ez/E, E,
               angles='xy', scale_units='xy', scale=40)
    plt.colorbar(label='场强 (V/m)')
    
    # 绘制圆环位置
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(a*np.cos(theta), np.zeros_like(theta), 'r.')
    
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title('带电圆环的电场分布(y=0平面)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_3d_potential():
    """绘制三维等势面"""
    # 创建三维网格点
    x = np.linspace(-2*a, 2*a, 30)
    y = np.linspace(-2*a, 2*a, 30)
    z = np.linspace(-2*a, 2*a, 30)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # 计算电势
    V = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                V[j,i,k] = potential(X[j,i,k], Y[j,i,k], Z[j,i,k])
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制等势面
    levels = np.linspace(V.min(), V.max(), 10)
    ax.contour3D(X, Y, Z, V, levels=levels)
    
    # 绘制圆环
    theta = np.linspace(0, 2*np.pi, 100)
    x_ring = a * np.cos(theta)
    y_ring = a * np.sin(theta)
    z_ring = np.zeros_like(theta)
    ax.plot3D(x_ring, y_ring, z_ring, 'r-')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('带电圆环的三维等势面')
    plt.show()

def test_symmetry():
    """测试计算结果的对称性"""
    # 测试点
    test_points = [
        (0.2, 0, 0),    # x轴上的点
        (0, 0.2, 0),    # y轴上的点
        (0.2, 0, 0.2),  # xz平面上的点
        (0, 0.2, 0.2)   # yz平面上的点
    ]
    
    print("对称性测试：")
    print("-" * 50)
    print("位置\t\t\t电势 (V)\t\t电场 (V/m)")
    print("-" * 50)
    
    for x, y, z in test_points:
        V = potential(x, y, z)
        Ex, Ey, Ez = electric_field(x, y, z)
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        print(f"({x:.1f}, {y:.1f}, {z:.1f})\t{V:.3e}\t\t{E_mag:.3e}")

def main():
    # 绘制等势线
    plot_potential_contour()
    
    # 绘制电场线
    plot_field_lines()
    
    # 绘制三维等势面
    plot_3d_potential()
    
    # 运行对称性测试
    test_symmetry()

if __name__ == '__main__':
    main()