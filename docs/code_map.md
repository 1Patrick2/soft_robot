# 项目代码地图 (Code Map) V7.0 - Cosserat 模型

本文档旨在清晰地展示项目中各个核心文件之间的依赖关系，帮助开发者理解代码结构。当前版本已全面迁移至基于 **Cosserat 杆理论** 的物理模型。

## 依赖关系网 (V7.0 - Cosserat 模型)

[验证层 (Verification)] ──────────────────────────────────
  └── 🔍 `verification/*.py` (单元/集成验证脚本)
      ├── **职责**: 对`src/cosserat`层各模块进行独立的、高精度的单元测试和集成测试。是保证物理模型、梯度、雅可比等核心组件正确性的最后防线。
      └── **依赖**: `src/cosserat/*`

[分析层 (Analysis)] ──────────────────────────────────────────
  └── 🔬 `analysis/path_tracking.py` (核心应用：路径跟踪)
      ├── **职责**: 实现高层路径跟踪策略，调用逆运动学求解器完成任务。
      └── **依赖**: `src/cosserat/outer_solver.py`

  └── 🔬 `analysis/workspace_analysis.py` (工作空间分析)
      ├── **职责**: 通过并行化的**同伦跟踪 (Homotopy Tracking)** 高效生成高质量的工作空间。
      └── **依赖**: `src/cosserat/solver.py`, `src/cosserat/kinematics.py`

[求解应用层 (Outer Solver)] ──────────────────────────────────
  └── 🚀 `src/cosserat/outer_solver.py` (全局逆解器: 解析雅可比)
      ├── **职责**: 提供单点的、高精度的逆运动学求解服务 `solve_ik_globally`。
      ├── **算法**: 基于**解析任务雅可比 `J_task`** 的牛顿法迭代求解。优化变量为 **`Δl_motor` (8维)**。
      └── **依赖**: `src/cosserat/solver.py` (评估每个`Δl_motor`对应的末端位姿), `src/cosserat/kinematics.py` (计算位姿误差)。

[内循环求解层 (Inner Solver)] ──────────────────────────────────
  └── ⚙️ `src/cosserat/solver.py` (静态平衡求解器)
      ├── **职责**: 核心函数 `solve_static_equilibrium`，输入电机位移`Δl_motor`，求解静态平衡构型`q_eq`。
      ├── **算法**: 使用高性能的 `L-BFGS-B` 算法，求解总势能梯度为零 `∇q(U_total) = 0` 的点。
      └── **依赖**: `src/cosserat/statics.py` (调用总势能 `U_total` 及其梯度 `∇U_total`)。

[物理建模层 (Physics Modeling)] ───────────────────────────────
  └── 🧱 `src/cosserat/statics.py` (核心静力学模型)
      ├── **职责**: 定义系统的**总势能函数 `U_total`** 及其**梯度 `∇U_total`**。它是连接“运动学模型”和“静态平衡”的桥梁。
      ├── **核心**:
      │   - **势能原理**: 基于最小势能原理，通过寻找势能函数的极小值点来确定系统的静态平衡状态。
      │   - **能量构成**: `U_total = U_elastic + U_gravity + U_cable`。所有能量项的计算都依赖于由`kinematics.py`提供的、在特定构型`q`下的机器人几何形状。
      │   - **梯度计算**: 采用“解析+数值”的混合方法计算总梯度，为`solver.py`提供高效且鲁棒的下降方向指引。
      └── **依赖**: `src/cosserat/kinematics.py` (获取计算势能所需的几何信息)。

  └── 🧱 `src/cosserat/kinematics.py` (核心运动学模型)
      ├── **职责**: 提供正向运动学 `forward_kinematics`，通过对 Cosserat 杆方程进行数值积分，得到末端位姿。
      ├── **核心**:
      │   - **控制方程**: `p'(s) = R(s) * v(s)`, `R'(s) = R(s) * hat(u(s))`
      │   - **数值方法**: 采用 **RK4 (四阶龙格-库塔)** 进行高精度积分。
      ├── **现状**: 此模块被认为是**完全正确且已通过验证的**。
      └── **依赖**: `src/utils/*`, `config/config.json`。

[底层 (Low Level)] ──────────────────────────────────────────
  └── 📦 `src/utils/*`
  └── 📄 `config/config.json`

[文档 (Documentation)] ──────────────────────────────────────────
  └── 📄 `docs/core_system_manual.md`
  └── 📄 `docs/code_map.md`
  └── 📄 `docs/readme.md`
  └── 📄 `docs/run.md`
  └── 📄 `docs/test.md`
  └── 📄 `docs/worklog.md`