## **文档修订准则 (Document Revision Standard)**

1.  **核心方案 (Core Design):** 本文档是项目的核心物理模型和数学方案的**唯一权威来源 (Single Source of Truth)**。
2.  **演进式更新 (Evolutionary Updates):** 本文档记录了从早期探索到最终方案的设计演进，通过**注释和保留旧方案**的方式，为未来的迭代提供了宝贵的历史参考。
3.  **代码同步 (Code Synchronization):** 本文档中描述的最终方案，与代码库中 `V-Final` 版本的 `statics.py`, `solver.py`, `outer_solver.py` 等核心模块的实现**严格保持一致**。




# **基于 Cosserat Rod 的连续体机器人耦合建模与分析 (V-Cosserat 最终技术方案)**

---

## **1. 引言 (Introduction)**

### **1.1 背景与意义**

连续体机器人因其高柔顺性、强冗余性，在狭小空间、非结构化环境中展现出独特优势。传统的 **PCC (Piecewise Constant Curvature)** 模型，尽管简洁高效，但在以下方面存在严重不足：

1. **局限的几何描述能力**：难以刻画弯曲段之间的平滑过渡，易出现“全弯或不弯”的病态解；
2. **物理一致性不足**：在能量最小化下，常常得到不符合物理直觉的解，例如弯曲后末端高度不降低；
3. **缆绳路径近似过度**：对缆绳绕行的几何建模存在简化，导致驱动映射与实际差异显著。

为解决上述问题，本方案采用 **Cosserat Rod 理论**，建立分布式、连续化的物理模型，结合数值积分方法，实现高保真度的运动学与静力学建模。此模型能够精确描述柔顺结构在多点驱动下的变形与受力分布，避免 PCC 病态解，并为后续的逆运动学与控制提供坚实基础。

---

## **2. 机器人结构与状态变量定义**

### **2.1 机器人结构**

机器人由三段串联组成：

* **PSS (Passive Structural Section)**: 长度 $L_p = 0.2 m$，柔顺但无驱动；
* **CMS1 (Proximal Cable-driven Main Section)**: 长度 $L_{c1} = 0.05 m$，四线驱动；
* **CMS2 (Distal Cable-driven Main Section)**: 长度 $L_{c2} = 0.05 m$，四线驱动。

### **2.2 Cosserat 状态变量**

Cosserat Rod 在弧长 $s \in [0, L]$ 上的状态由：

$$
X(s) = \{ R(s), p(s), n(s), m(s) \}
$$

组成：

* $R(s) \in SO(3)$: 局部旋转矩阵；
* $p(s) \in \mathbb{R}^3$: 空间位置；
* $n(s)$: 内力；
* $m(s)$: 内力矩。

### **2.3 离散化策略**

为计算可行性，将每段划分为 $N=10$ 个有限单元。每个单元的控制变量为局部曲率向量：

$$
\kappa_i = (\kappa_x, \kappa_y, \kappa_z)^T
$$

其中 $\kappa_z$ 表示扭转，通常可忽略。

---

## **3. 运动学建模 (Kinematics)**

### **3.1 Cosserat 微分方程**

机器人形态由以下 ODE 控制：

$$
\frac{dR}{ds} = R \hat{\kappa}(s), \quad \frac{dp}{ds} = R e_z
$$

其中：

* $\hat{\kappa}$ 为曲率向量的反对称矩阵；
* $e_z = (0,0,1)^T$。

### **3.2 数值积分**

采用 **RK4（四阶 Runge-Kutta）** 积分，从基座到末端迭代得到：

* PSS 段末端位姿 $T_{pss}$
* CMS1 段末端位姿 $T_{cms1}$
* CMS2 段末端位姿 $T_{cms2}$

最终末端位姿：

$$
T_E^B = T_{pss} \cdot T_{cms1} \cdot T_{cms2}
$$

---

## **4. 静力学建模 (Statics)**

### **4.1 势能最小化原则**

平衡构型 $q_{eq}$ 由以下条件给出：

$$
q_{eq} = \arg\min_q U_{total}(q, \Delta l_{motor}), \quad \nabla_q U_{total} = 0
$$

### **4.2 势能构成**

1. **弯曲能 (Distributed Bending Energy)**

$$
U_{bend} = \sum_{i=1}^N \frac{1}{2} EI \|\kappa_i\|^2 \Delta s
$$

2. **重力势能**

$$
U_g = \sum_j m_j g z_{com,j}
$$

3. **缆绳能量 (Cable Stretch Energy)**
   缆绳路径由 Cosserat 解确定：

$$
l_i(q) = \int_0^L \| p_{cable}(s;q) - p_{anchor,i}\| ds
$$

驱动映射：

$$
\Delta l_i(q) = l_{straight,i} - l_i(q)
$$

缆绳能：

$$
U_{cable} = \frac{1}{2} k_c \sum_i [\max(0, \Delta l_{motor,i} - \Delta l_i(q))]^2
$$

### **4.3 总势能**

$$
U_{total} = U_{bend} + U_g + U_{cable}
$$

---

## **5. 约束条件 (Constraints)**

1. **几何约束**：每个单元保持非伸长假设（弧长守恒）。
2. **边界条件**：基座固定，末端自由。
3. **驱动约束**：电机位移在 $[-0.12, 0.12]$ m 范围。
4. **物理约束**：曲率上界 $\kappa_{max} = 10 rad/m$。

---

## **6. 求解器设计 (Solvers)**

### **6.1 内循环 (静力学平衡求解)**

* 优化变量：离散化曲率向量 $\kappa_i$ (约 30 维)。
* 目标：最小化总势能 $U_{total}$。
* 方法：

  * 一阶：L-BFGS-B
  * 二阶修正：高性能 Hessian 近似（Gauss-Newton）

### **6.2 外循环 (逆运动学)**

1. **Phase 1: 全局探索 (PSO)**

   * 输入空间：$\Delta l_{motor}$
   * 粒子数：64，迭代 20-50 代。
2. **Phase 2: Newton-Raphson 精收敛**

   * 利用隐函数定理：

   $$
   J_{task} = J_{kin} (-H^{-1} C)
   $$

   * 更新律：阻尼最小二乘 + 线搜索。

---

## **7. 框架与模块结构 (Framework)**

* `kinematics_cosserat.py` : Cosserat 积分正运动学
* `statics_cosserat.py` : 势能、梯度、Hessian
* `solver_cosserat.py` : 静力学平衡求解
* `outer_solver_cosserat.py` : IK 外循环

---

## **8. 结论与展望 (Conclusion & Future Work)**

### **优势**

* 高保真度：避免 PCC “中空”和“高度不降”问题；
* 物理一致性：缆绳路径由积分自动保证；
* 更平滑：CMS 段能连续过渡弯曲。

### **劣势**

* 计算复杂度高，需要并行或 GPU 加速；
* 内循环优化变量增多（20\~30维）；
* 求解时间相对 PCC 更长。

### **未来工作**

* 优化数值积分与能量求解器，提升实时性；
* 将当前静力学模型拓展为动力学模型；
* 结合实验标定，提高参数物理一致性。
