# --- 调试配置文件 ---
# 此文件用于存放全局调试开关, 以避免模块间的循环导入问题。

# 单段调试模式开关
# True: 系统将只计算单段(CMS_Proximal)的运动学和静力学。
# False: 系统将使用完整的原始三段模型。
DEBUG_SINGLE_SEGMENT = False