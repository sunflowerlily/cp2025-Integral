import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import quad # 如果需要精确计算单点，可以取消注释

# --- 常量定义 ---
a = 1.0  # 圆环半径 (单位: m)
# q = 1.0  # 可以定义 q 参数，或者直接在 C 中体现
# V(x,y,z) = q/(2*pi) * integral(...)
# C 对应 q/(2*pi)，这里设 q=1
C = 1.0 / (2 * np.pi)

# --- 计算函数 ---

def calculate_potential_on_grid(y_coords, z_coords):
    """
    在 yz 平面 (x=0) 的网格上计算电势 V(0, y, z)。
    使用 numpy 的向量化和 trapz 进行数值积分。

    参数:
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组

    返回:
        V (np.ndarray): 在 (y, z) 网格上的电势值 (z 维度优先)
        y_grid (np.ndarray): 绘图用的二维 y 网格坐标
        z_grid (np.ndarray): 绘图用的二维 z 网格坐标
    """
    print("开始计算电势...")
    # 1. 创建 y, z, phi 网格 (使用 np.mgrid 或 np.meshgrid)
    #    注意维度顺序和 phi 的积分点数
    # z_grid, y_grid, phi_grid = ...

    # 2. 计算场点到圆环上各点的距离 R
    #    圆环方程: x_s = a*cos(phi), y_s = a*sin(phi), z_s = 0
    #    场点: (0, y_grid, z_grid)
    # R = ...

    # 3. 处理 R 可能为零或非常小的情况，避免除零错误
    # R[R < 1e-10] = 1e-10

    # 4. 计算电势微元 dV = C / R
    # dV = ...

    # 5. 对 phi 进行积分 (例如使用 np.trapz)
    #    注意指定积分轴和积分步长 dx (如果 trapz 需要)
    # V = np.trapz(...)

    print("电势计算完成.")
    # 6. 返回计算得到的电势 V 和对应的 y_grid, z_grid (取一个切片)
    # return V, y_grid[:,:,0], z_grid[:,:,0]
    pass # 学生在此处实现代码

def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """
    根据电势 V 计算 yz 平面上的电场 E = -∇V。
    使用 np.gradient 进行数值微分。

    参数:
        V (np.ndarray): 电势网格 (z 维度优先)
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组

    返回:
        Ey (np.ndarray): 电场的 y 分量
        Ez (np.ndarray): 电场的 z 分量
    """
    print("开始计算电场...")
    # 1. 计算 y 和 z 方向的网格间距 dy, dz
    # dz = ...
    # dy = ...

    # 2. 使用 np.gradient 计算电势的负梯度
    #    注意 V 的维度顺序和 gradient 返回值的顺序
    #    E = -∇V
    # grad_z, grad_y = np.gradient(-V, dz, dy)
    # Ez = grad_z
    # Ey = grad_y

    print("电场计算完成.")
    # 3. 返回电场的 y 和 z 分量
    # return Ey, Ez
    pass # 学生在此处实现代码

# --- 可视化函数 ---

def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """
    绘制 yz 平面上的等势线和电场线。

    参数:
        y_coords, z_coords: 定义网格的坐标范围
        V: 电势网格
        Ey, Ez: 电场分量网格
        y_grid, z_grid: 绘图用的二维网格坐标
    """
    print("开始绘图...")
    fig = plt.figure('Potential and Electric Field of Charged Ring (yz plane, x=0)', figsize=(12, 6))

    # 1. 绘制等势线图 (左侧子图)
    plt.subplot(1, 2, 1)
    #    - 使用 plt.contourf 绘制填充等势线图，设置 levels 和 cmap
    # contourf_plot = plt.contourf(...)
    #    - 添加颜色条 plt.colorbar()
    # plt.colorbar(...)
    #    - (可选) 使用 plt.contour 叠加绘制等势线线条
    # plt.contour(...)
    #    - 设置坐标轴标签 (xlabel, ylabel) 和标题 (title)
    # plt.xlabel('y / a')
    # ...
    #    - 设置坐标轴比例一致 plt.gca().set_aspect('equal', adjustable='box')
    #    - 添加网格 plt.grid()

    # 2. 绘制电场线图 (右侧子图)
    plt.subplot(1, 2, 2)
    #    - (可选) 计算电场强度 E_magnitude 用于着色
    # E_magnitude = ...
    #    - 使用 plt.streamplot 绘制电场线，传入 y_grid, z_grid, Ey, Ez
    #      可以设置 color, cmap, linewidth, density, arrowstyle 等参数
    # stream_plot = plt.streamplot(...)
    #    - 设置坐标轴标签和标题
    # plt.xlabel('y / a')
    # ...
    #    - 设置坐标轴比例一致
    #    - 添加网格
    #    - (可选) 标记圆环截面位置 plt.plot([-1, 1], [0, 0], 'ro', ...)
    #    - 添加图例 plt.legend()

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()
    print("绘图完成.")
    pass # 学生在此处实现代码

# --- 主程序 ---
if __name__ == "__main__":
    # 定义计算区域 (yz 平面, x=0)
    # 范围可以以圆环半径 a 为单位
    num_points_y = 40 # y 方向点数
    num_points_z = 40 # z 方向点数
    range_factor = 2  # 计算范围是半径的多少倍
    y_range = np.linspace(-range_factor * a, range_factor * a, num_points_y)
    z_range = np.linspace(-range_factor * a, range_factor * a, num_points_z)

    # 1. 计算电势
    # 调用 calculate_potential_on_grid 函数获取 V, y_grid, z_grid
    # V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)
    V, y_grid, z_grid = None, None, None # 占位符，学生需要取消注释并调用函数

    # 2. 计算电场
    # 调用 calculate_electric_field_on_grid 函数获取 Ey, Ez
    # Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)
    Ey, Ez = None, None # 占位符

    # 3. 可视化
    # 确保 V, Ey, Ez, y_grid, z_grid 都有有效值后再绘图
    if V is not None and Ey is not None:
         plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
    else:
         print("计算未完成，无法绘图。请先实现计算函数。")