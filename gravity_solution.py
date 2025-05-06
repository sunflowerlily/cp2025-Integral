if __name__ == '__main__':
    # 参数设置 (边长10m，质量1e4kg)
    length = 10
    mass = 1e4
    
    # 计算并绘制引力曲线
    plot_force_vs_height(length, mass)
    
    # 打印几个关键点的引力值
    for z in [0.1, 1, 5, 10]:
        F = calculate_force(length, mass, z)
        print(f"高度 z = {z:.1f}m 处的引力 F_z = {F:.3e} N")
    
    # 测试点
    test_z = [0.1, 0.5, 1, 2, 5, 10]
    
    # 比较积分方法
    compare_integration_methods(length, mass, test_z)