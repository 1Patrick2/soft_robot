# 项目代码地图 (Code Map) V6.0

本文档旨在清晰地展示项目中各个核心文件之间的依赖关系，帮助开发者理解代码结构，以便在修改文件时，能够清楚地知道可能影响到的其他文件。

## 依赖关系网 (V6.0 - 纯位移控制最终版)

[验证层 (Verification)] ──────────────────────────────────
  └── 🔍 `verification/*.py` (单元/集成验证脚本)
      ├── **职责**: 对`src`层各模块进行独立的、高精度的单元测试和集成测试。是保证物理模型、梯度、雅可比等核心组件正确性的最后防线。
      └── **依赖**: `src/*`

[分析层 (Analysis)] ──────────────────────────────────────────
  └── 🔬 `analysis/path_tracking.py` (核心应用：路径跟踪)
      ├── **职责**: 实现高层路径跟踪策略，调用逆运动学求解器完成任务。
      └── **依赖**: `src/outer_solver.py`

  └── 🔬 `analysis/workspace_analysis.py` (工作空间分析)
      ├── **职责**: 通过并行化的蒙特卡洛正向求解，高效生成工作空间。
      └── **依赖**: `src/solver.py`, `src/kinematics.py`

[求解应用层 (Outer Solver)] ──────────────────────────────────
  └── 🚀 `src/outer_solver.py` (V6.0 - 全局逆解器: PSO+Newton)
      ├── **职责**: 提供单点的、高精度的逆运动学求解服务 `solve_ik_globally`。
      ├── **算法**: 采用“**轻量化PSO全局探索 + 纯位移空间牛顿法精炼**”的“接力赛”策略。优化变量为 **`Δl_motor` (8维)**。
      └── **依赖**: `src/solver.py` (用于评估每个`Δl_motor`对应的末端位姿), `src/kinematics.py` (用于计算位姿误差)。

[内循环求解层 (Inner Solver)] ──────────────────────────────────
  └── ⚙️ `src/solver.py` (V6.0 - 静态平衡求解器)
      ├── **职责**: 核心函数 `solve_static_equilibrium_disp_ctrl`，输入电机位移`Δl_motor`，求解静态平衡构型`q_eq`。
      ├── **算法**: 使用高性能的 `L-BFGS-B` 算法，求解总势能梯度为零 `∇q(U_total) = 0` 的点。
      └── **依赖**: `src/statics.py` (调用总势能 `U_total` 及其梯度 `∇U_total`)。

[物理建模层 (Physics Modeling)] ───────────────────────────────
  └── 🧱 `src/statics.py` (V6.0 - 核心静力学模型)
      ├── **职责**: 定义系统的总势能 `U_total = U_elastic + U_gravity + U_cable` 及其梯度 `∇U_total`。
      ├── **核心**:
      │   - **构型空间**: 所有计算均基于 **6D** 构型 `q`。
      │   - **驱动映射**: 定义了从构型`q`到缆绳**长度变化量** `Δl_robot(q)` 的正确物理映射。
      │   - **平滑模型**: 使用光滑近似函数处理缆绳松弛，确保总势能函数处处可微。
      ├── **现状**: 为保证系统可用性，当前版本临时使用**高精度数值梯度** (`approx_fprime`) 作为后备方案。**核心开发任务是分块修复各项势能的解析梯度**。
      └── **依赖**: `src/kinematics.py` (获取计算势能所需的几何信息)。

  └── 🧱 `src/kinematics.py` (V6.0 - 核心运动学模型)
      ├── **职责**: 提供正向运动学 `T_EB(q)` 及其运动学雅可比 `J_kin`。
      ├── **核心**: 所有计算均基于 **6D** 构型 `q`。此模块被认为是**完全正确且已通过验证的**。
      └── **依赖**: `src/utils/*`, `config/config.json`。

[底层 (Low Level)] ──────────────────────────────────────────
  └── 📦 `src/nondimensionalizer.py`
  └── 📦 `src/utils/*`
  └── 📄 `config/config.json`

[文档 (Documentation)] ──────────────────────────────────────────
  └── 📄 `docs/core_system_manual.md`
  └── 📄 `docs/code_map.md`
  └── 📄 `docs/readme.md`
  └── 📄 `docs/run.md`
  └── 📄 `docs/test.md`
  └── 📄 `docs/worklog.md`
