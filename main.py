import numpy as np
# [路径修正] 使用新的模块路径
from src.utils.read_config import load_config
from src.kinematics import forward_kinematics

def main():
    """
    主函数，用于运行连续体机器人正运动学仿真。
    """
    # 1. 从config.json加载机器人参数
    try:
        # [路径修正] 使用新的配置文件路径
        robot_parameters = load_config('config/config.json')
        print("成功从config/config.json加载机器人参数:")
        print(robot_parameters)
    except FileNotFoundError:
        print("错误: config.json 未找到。请确保它在正确的目录中。")
        return

    # 2. 定义一个示例配置向量'q'
    # 假设机器人仅在第一个活动段(CMS_Proximal)
    # 在x-z平面(phi=0)内以0.5的曲率弯曲。
    # 其他段是直的。
    q_sample = [
        robot_parameters['Geometry']['PSS_initial_length'],  # L_p (来自配置的被动段长度)
        0,                       # kappa_p (被动段是直的)
        0,                       # phi_p
        0.5,                     # kappa_c1 (在此处弯曲)
        0,                       # phi_c1 (在x-z平面内弯曲)
        0,                       # kappa_c2 (远端段是直的)
        0                        # phi_c2
    ]
    print(f"\n为示例配置q计算正运动学:\n{q_sample}")

    # 3. 计算正运动学
    T_final = forward_kinematics(q_sample, robot_parameters)

    # 4. 打印结果
    np.set_printoptions(precision=4, suppress=True)
    print(f"\n生成的变换矩阵 (T_E_B):\n{T_final}")

if __name__ == '__main__':
    main()
