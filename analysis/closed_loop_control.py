import numpy as np
import matplotlib.pyplot as plt
import time

# 导入项目模块
from src.kinematics import forward_kinematics
from src.outer_solver import calculate_pose_error # 借用误差计算函数
from src.utils.read_config import load_config

class VirtualRobot:
    """一个简化的虚拟机器人模型，只包含运动学，用于测试控制器。"""
    def __init__(self, initial_q, params):
        """使用初始构型和机器人参数初始化。"""
        self.q = np.array(initial_q, dtype=float)
        self.params = params
        self.T_current, _ = forward_kinematics(self.q, self.params)

    def update_q(self, delta_q):
        """接收一个微小的构型变化量，更新机器人状态。"""
        self.q += delta_q
        # 重新计算正运动学，得到新的位姿
        self.T_current, _ = forward_kinematics(self.q, self.params)

    def get_pose_feedback(self, noise_level=0.001):
        """获取机器人当前的位姿，并模拟加入少量传感器噪声。"""
        noisy_pose = self.T_current.copy()
        # 只对位置增加噪声
        noise = (np.random.rand(3) - 0.5) * noise_level
        noisy_pose[:3, 3] += noise
        return noisy_pose

class PController:
    """一个简单的P控制器，用于位姿控制。"""
    def __init__(self, Kp):
        """使用一个6x7的增益矩阵Kp初始化。"""
        self.Kp = Kp

    def calculate_control_output(self, error_vector):
        """
        根据6D位姿误差，计算出一个7D的构型调整量 delta_q。
        这是一个简化的映射，更高级的控制器会使用雅可比矩阵。
        """
        # 直接将6D误差通过增益矩阵，映射为7D的构型调整量
        delta_q = self.Kp @ error_vector
        return delta_q

def main():
    """
    闭环控制仿真主函数。
    """
    print("--- 开始闭环控制仿真 ---")
    start_time = time.time()

    # 1. 加载配置，初始化机器人和控制器
    config = load_config('config/config.json')
    
    # 定义一个起始构型和目标构型
    q_initial = np.array(config['Initial_State']['q0'])
    q_target = np.array([0.145, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    T_target, _ = forward_kinematics(q_target, config)

    # 初始化虚拟机器人和P控制器
    robot = VirtualRobot(q_initial, config)
    
    # 定义P控制器的增益矩阵 Kp (6x7)
    # 这是一个需要被整定（tuning）的关键参数。我们先凭经验给一个初始值。
    # 误差的前3维是位置（米），后3维是姿态（弧度）。
    # 输出的前1维是长度（米），后6维是曲率和角度。
    Kp = np.diag([-0.5, -0.5, -0.5,  # 位置误差 -> 长度/曲率/角度... (负号表示朝误差减小的方向调整)
                   -0.1, -0.1, -0.1, -0.1]) # 姿态误差 -> ...
    # 这是一个简化的对角矩阵，实际的Kp可能是非对角的。我们先只用一个简单的形式。
    # 为了将6D误差映射到7D构型，我们扩展一下Kp
    Kp_expanded = np.zeros((7, 6))
    Kp_expanded[0, 2] = -1.0  # Z轴误差主要影响PSS段长度Lp
    Kp_expanded[1, 1] = -5.0  # Y轴误差主要影响kappa_p
    Kp_expanded[3, 0] = 5.0   # X轴误差主要影响kappa_c1
    Kp_expanded[5, 0] = 5.0   # X轴误差也影响kappa_c2
    Kp_expanded[2, 4] = -0.5  # 姿态误差影响phi...
    Kp_expanded[4, 3] = -0.5
    Kp_expanded[6, 5] = -0.5
    controller = PController(Kp_expanded)

    # 2. 仿真循环
    simulation_steps = 200
    dt = 0.1 # 时间步长
    history_error_norm = []
    history_q = []

    print(f"目标位姿 T_target:\n{np.round(T_target, 3)}")

    for i in range(simulation_steps):
        # a. 感知：获取当前位姿反馈
        T_current = robot.get_pose_feedback()
        
        # b. 决策：计算误差并生成控制指令
        error_vec = calculate_pose_error(T_current, T_target)
        delta_q = controller.calculate_control_output(error_vec) * dt

        # c. 执行：更新机器人状态
        robot.update_q(delta_q)

        # d. 记录数据用于绘图
        error_norm = np.linalg.norm(error_vec[:3]) # 只记录位置误差的范数
        history_error_norm.append(error_norm)
        history_q.append(robot.q.copy())

        if i % 20 == 0:
            print(f"Step {i:3d}: Position Error = {error_norm:.4f} m")

    print(f"Step {simulation_steps}: Position Error = {history_error_norm[-1]:.4f} m")

    # 3. 绘制结果
    print("\n--- 仿真完成，正在绘制结果图 ---")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 误差收敛曲线
    ax1.plot(history_error_norm, 'b-')
    ax1.set_title('Positional Error Convergence over Time')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Position Error (m)')
    ax1.grid(True)

    # 图2: 构型参数变化曲线
    q_array = np.array(history_q)
    labels = ['Lp', 'kp', 'phip', 'kc1', 'phic1', 'kc2', 'phic2']
    for i in range(q_array.shape[1]):
        ax2.plot(q_array[:, i], label=labels[i])
    ax2.set_title('Configuration (q) Evolution over Time')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Parameter Value')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    output_path = "plots/closed_loop_p_control_sim.png"
    plt.savefig(output_path)
    print(f"结果图已保存至: {output_path}")
    plt.close()

    end_time = time.time()
    print(f"\n分析完成，总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
