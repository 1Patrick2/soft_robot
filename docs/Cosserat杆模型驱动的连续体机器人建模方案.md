## **文档修订准则 (Document Revision Standard)**

1.  **核心方案 (Core Design):** 本文档是项目的核心物理模型和数学方案的**唯一权威来源 (Single Source of Truth)**。
2.  **演进式更新 (Evolutionary Updates):** 本文档记录了从早期探索到最终方案的设计演进，通过**注释和保留旧方案**的方式，为未来的迭代提供了宝贵的历史参考。
3.  **代码同步 (Code Synchronization):** 本文档中描述的最终方案，与代码库中 `V-Final` 版本的 `statics.py`, `solver.py`, `outer_solver.py` 等核心模块的实现**严格保持一致**。




# V-Cosserat：从原理到实现的逐步推导与数值方案

## 目录（快速导航）

1.  概览与基本假设
2.  Cosserat 基本方程（连续形式）
3.  构型与变量离散化（从连续到数值）
4.  正运动学（数值积分 RK4）及灵敏度方程（变分方程）
5.  缆绳路径与长度映射的严格推导
6.  总势能与能量项逐项导出（并给出其对离散参数的梯度）
7.  雅可比 / 海森（Hessian）推导与高效近似（Gauss-Newton）
8.  隐函数定理求解任务雅可比（J\_task）的严格推导
9.  数值方案与算法伪代码（内循环、外循环、warm-start/homotopy）
10. 实现细节、复杂度与参数建议

-----

## 1\. 概览与基本假设

  * **目标**：用 Cosserat Rod 建模三段串联连续体（PSS, CMS1, CMS2），使模型能真实反映沿长方向曲率分布以及缆线路径随构型变化的影响，从而避免 PCC 的覆盖限制与病态分布问题。
  * **假设**（为了实用性）：各段可近似为不可伸长、可弯曲且可以扭转的细杆（inextensible Cosserat rod）；截面刚性足够，截面不随弯曲扭曲改变（刚性截面假设）。
  * **变量**：以弧长 $s$ 为自变量，定义在每个段的区间上；力学变量包括位置 $p(s)$、旋转矩阵 $R(s)$、内力 $n(s)$、内力矩 $m(s)$、曲率向量 $\\kappa(s)$。

## 2\. Cosserat 连续方程（连续形式，推导基础）

### 2.1 运动学：旋转与位置关系

定义局部基 ${d\_1(s), d\_2(s), d\_3(s)}$ 作为杆截面在全局坐标系下的方向，收集为旋转矩阵 $R(s) = [d\_1, d\_2, d\_3] \\in SO(3)$。弧长变量 $s\\in[0,L]$。

微分几何关系：

$$
\frac{dR}{ds} = R \,\widehat{\kappa}(s)
\qquad
\frac{dp}{ds} = R e_3 \quad(\text{inextensible: } \|dp/ds\|=1)
$$其中 $e\_3 = (0,0,1)^\\top$ 表示局部轴（杆沿局部 z 轴方向伸长），$\\widehat{\\kappa}$ 是曲率—扭转向量 $\\kappa=(\\kappa\_x,\\kappa\_y,\\kappa\_z)^\\top$ 的反对称矩阵表示：

$$\\widehat{\\kappa} =
\\begin{bmatrix}
0      & -\\kappa\_z & \\kappa\_y \\
\\kappa\_z & 0       & -\\kappa\_x \\
\-\\kappa\_y & \\kappa\_x & 0
\\end{bmatrix}
$$\> 注：若忽略扭转，可将 $\\kappa\_z\\approx 0$。

### 2.2 力平衡 / 力矩平衡（静态平衡形式）

静力平衡（无体力分布项或有重力）：

$$
\frac{dn}{ds} + f_{ext}(s) = 0
$$$$
\frac{dm}{ds} + \frac{dp}{ds}\times n + l_{ext}(s) = 0
$$其中 $f\_{ext}$ 是分布外力密度（例如重力和链索对节点的分布作用），$l\_{ext}$ 是分布外力矩密度（通常为0）。

### 2.3 本构关系（线性弹性）

假设线弹性（小应变在杆截面上成立），内力矩和曲率的线性关系：

$$m(s) = B(s),(\\kappa(s) - \\kappa\_0(s))
$$其中 $B(s)$ 是截面抗弯扭刚度矩阵（对各方向弯曲 / 扭转给出 $EI\_x,EI\_y,GJ$），$\\kappa\_0$ 为预曲率（若无预曲率则 $\\kappa\_0=0$）。

若允许剪切/伸长：

$$
n(s) = S(s)\, (v(s)-v_0(s))
$$其中 $v(s)=R(s)^\\top p'(s)$ 为剪切/伸长应变；但我们采用不可伸长假设 $v=e\_3$（或只允许小伸长），则通常忽略或将 $n$ 作为约束乘子处理。

## 3\. 离散化：从连续到可计算的自由度

为实现数值求解，将整杆在弧长上离散为 $N$ 个单元（建议每段 6–12 单元，整个机器人 $N\\approx20–40$，工程上常取 $N\\approx20$）。对离散形式定义：

* 单元 $i$ 的长度 $\\Delta s\_i$（均匀时 $\\Delta s = L/N$）。
* 在单元中心或端点定义曲率向量 $\\kappa\_i\\in\\mathbb{R}^3$（工程上常用中心常数近似）。
* 离散变量向量 $\\mathbf{x} = [\\kappa\_1^\\top, \\kappa\_2^\\top, \\dots, \\kappa\_N^\\top]^\\top \\in \\mathbb{R}^{3N}$；若忽略扭转可用 $2N$ 维（只包含 $\\kappa\_x,\\kappa\_y$）。

离散化带来数值公式的替代关系：

* 旋转与位置的数值积分（以 RK4 或指数映射组合）可从已知 $R(s\_{i-1}),p(s\_{i-1})$ 对单元 $i$ 应用：
$$

```
$$R(s\_i) \\approx R(s\_{i-1}) \\exp(\\widehat{\\kappa\_i}\\Delta s)
$$
$$$$
$$p(s\_i) \\approx p(s\_{i-1}) + \\int\_{0}^{\\Delta s} R(s\_{i-1}) \\exp(\\widehat{\\kappa\_i}\\xi) e\_3 , d\\xi
$$
$$积分项可用 RK4 或闭式近似（若 $\\kappa$ 非常小可采用级数展开）。
```

## 4\. 正运动学（数值积分）与灵敏度（变分）方程 — 逐步推导

### 4.1 RK4 积分（单元推进步骤）

给定单元 $i$ 的曲率 $\\kappa\_i$，已知 $R\_{i-1}, p\_{i-1}$，求 $R\_i, p\_i$。
用 RK4 对 ODE：

$$
\dot{R} = R \widehat{\kappa}(s), \quad \dot{p} = R e_3
$$在单元区间 $[0,\\Delta s]$ 做标准 RK4。实现细节略，但关键是我们能通过 $\\kappa\_i$ 计算出 $R\_i, p\_i$——这就是正运动学。

### 4.2 变分（灵敏度）方程（为什么需要）

为了得到能量关于自由度 $\\mathbf{x}$ 的梯度（用于优化器），需要计算：

* $\\partial p(s) / \\partial \\kappa\_j$
* $\\partial R(s) / \\partial \\kappa\_j$

即：若改变单元 $j$ 的曲率，会如何影响沿杆的旋转与位置？这由线性化方程（变分方程）给出。

### 4.3 推导变分方程（连续形式）

令对 $\\kappa$ 的微扰为 $\\delta\\kappa(s)$，对应变数为 $\\delta R(s), \\delta p(s)$。从运动学方程线性化：
起始：

$$\\delta R' = \\delta R ,\\widehat{\\kappa} + R ,\\widehat{\\delta\\kappa}
$$$$
\\delta p' = \\delta R , e\_3
$$将 $\\delta R = R \\widehat{\\eta}$（存在向量 $\\eta(s)$ 使得 $\\delta R = R \\widehat{\\eta}$，这是因为 $T\_{R}SO(3)$ 的切空间表示），代入得：

$$
R \widehat{\eta}' = R \widehat{\eta}\widehat{\kappa} + R \widehat{\delta\kappa}
\Rightarrow
\widehat{\eta}' = \widehat{\eta}\widehat{\kappa} + \widehat{\delta\kappa}
$$转换为向量形式（利用向量恒等式）得到：

$$\\eta' = \\delta\\kappa - \\kappa \\times \\eta
$$同时

$$
\delta p' = R \widehat{\eta} e_3 = R (\eta \times e_3)
$$所以

$$\\delta p(s) = \\int\_0^s R(\\xi) (\\eta(\\xi) \\times e\_3) , d\\xi
$$这是一组线性常系数（随解变化）ODE，可随主积分一起以数值方法解出 —— 这是\*\*灵敏度方程（tangent linear model）\*\*的连续形式。离散后对每个自由度（单元）只需解一次线性微分方程组，代价与变量数成线性比例（并可以向量化并行化）。

## 5\. 缆绳路径与长度映射（严格推导）

这是本模型的核心：给定杆的形状 $p(s),R(s)$，如何精确计算第 $i$ 根缆绳的路径长度 $l\_i(q)$ 及其对自由度的导数。

### 5.1 缆绳在杆上的轨迹参数化

假设缆绳在杆第 $s$ 处穿出截面上的局部点 $\\rho\_i(s)$（例如在截面圆周上的固定角度位置），在全局坐标下其位置为：

$$
p_{cable,i}(s) = p(s) + R(s)\, \rho_i(s)
$$若缆线穿出点沿截面角度固定且截面半径固定，则 $\\rho\_i(s) = r\_i$（常向量），单位为米。常见设置：$r\_i = [r\\cos\\alpha\_i, r\\sin\\alpha\_i, 0]^\\top$。

### 5.2 路径长度（连续公式）

对一根缆线，从 $s=a$（起点）到 $s=b$（终点）：

$$l\_i = \\int\_a^b \\left| \\frac{d}{ds} p\_{cable,i}(s) \\right| ds
\= \\int\_a^b \\left| p'(s) + R'(s) r\_i \\right| ds
$$利用 $p'(s) = R(s)e\_3, ; R'(s) = R(s)\\widehat{\\kappa}(s)$：

$$
p'(s) + R'(s) r_i = R(s) \big( e_3 + \widehat{\kappa}(s) r_i \big)
$$由于 $R(s)$ 为正交矩阵，范数不变：

$$|p'(s) + R'(s) r\_i| = | e\_3 + \\widehat{\\kappa}(s) r\_i |
$$因此得到一个非常简洁的公式：

$$
\boxed{ \; l_i = \int_a^b \sqrt{ \big(e_3 + \widehat{\kappa}(s) r_i\big)^\top \big(e_3 + \widehat{\kappa}(s) r_i\big) } \; ds \; }
$$> 注：这公式体现了缆绳在各截面绕行形成附加的局部线元素长度，由局部曲率与截面偏移 $r\_i$ 决定。

### 5.3 离散化后的单元表达（实用计算）

若单元 $j$ 上 $\\kappa\_j$ 近似为常值，则在该单元上：

$$l\_{i,j} \\approx | e\_3 + \\widehat{\\kappa\_j} r\_i | , \\Delta s\_j
$$整根缆线长度：

$$
l_i \approx \sum_{j \in \text{path}} \| e_3 + \widehat{\kappa_j} r_i \| \, \Delta s_j
$$这里 “path” 表示缆线穿过的单元集合（通常为 CMS 段所在的单元）。

### 5.4 关于直立状态与基准长度

直立（$\\kappa\\equiv 0$）时：
$l\_{i,\\text{straight}} = \\int\_a^b |e\_3| ds = b-a$（等于弧长），因此定义驱动映射：

$$\\Delta l\_i(q) = l\_{i,\\text{straight}} - l\_i(q)
$$这与 PCC 版本的“缩短量”语义一致。

## 6\. 总势能与梯度逐项推导（离散形式、逐步算子）

总势能（离散）：

$$
U(\mathbf{x},\Delta l_{motor}) = U_{bend}(\mathbf{x}) + U_g(\mathbf{x}) + U_{cable}(\mathbf{x},\Delta l_{motor}) + U_{reg}(\mathbf{x})
$$### 6.1 弯曲能（离散）

由第 3 节离散化：

$$U\_{bend} = \\tfrac12 \\sum\_{j=1}^N \\kappa\_j^\\top B\_j \\kappa\_j , \\Delta s\_j
$$因此对单元 $j$ 的梯度（对 $\\kappa\_j$）：

$$
\frac{\partial U_{bend}}{\partial \kappa_j} = B_j \kappa_j \, \Delta s_j
$$这是局部且对角的贡献，方便构成海森块。

### 6.2 重力能（离散）

若杆体离散为质元 $m\_j$（对应单元质心），单元质心位置为 $p\_{com,j}$（可由 $p\_{j-1},p\_j$ 线性近似或精确积分得到），则：

$$U\_g = \\sum\_j m\_j g , (p\_{com,j})\_z
$$梯度：

$$
\frac{\partial U_g}{\partial \kappa_k}
= \sum_j m_j g \, \frac{\partial (p_{com,j})_z}{\partial \kappa_k}
$$其中 $\\partial p\_{com,j}/\\partial \\kappa\_k$ 可由第 4 节变分方程（灵敏度方程）数值求解得到（通过同时积分灵敏度方程来获得对所有 $\\kappa\_k$ 的影响）。

### 6.3 缆绳能（离散）与其梯度（逐步推导）

离散化下，对于第 $i$ 根缆绳：

$$l\_i(\\mathbf{x}) = \\sum\_{j \\in path(i)} \\ell\_{i,j}(\\kappa\_j), \\quad
\\ell\_{i,j}(\\kappa\_j) = | e\_3 + \\widehat{\\kappa\_j} r\_i | \\Delta s\_j
$$定义伸长/拉伸量（注意 sign）：

$$
\text{stretch}_i = \Delta l_{motor,i} - \Delta l_i(\mathbf{x})
= \Delta l_{motor,i} - \big( l_{i,\text{straight}} - l_i(\mathbf{x}) \big)
= \Delta l_{motor,i} - l_{i,\text{straight}} + l_i(\mathbf{x})
$$（在实现中通常直接计算 $\\text{stretch} = \\Delta l\_{motor} - \\Delta l(\\mathbf{x})$）

采用光滑化的零下截断（如你已实现的 `smooth_max_zero`）：

$$s\_i = \\mathrm{smooth\_max\_zero}(\\text{stretch}\_i)
$$缆绳能：

$$
U_{cable} = \tfrac12 k_c \sum_i s_i^2
$$现在推导对单元自由度 $\\kappa\_j$ 的偏导。先对单个缆绳求导：

$$\\frac{\\partial U\_{cable}}{\\partial \\kappa\_j}
\= k\_c \\sum\_i s\_i \\frac{\\partial s\_i}{\\partial \\kappa\_j}
\= k\_c \\sum\_i s\_i , s\_i'(\\text{stretch}\_i) \\cdot \\frac{\\partial \\text{stretch}\_i}{\\partial \\kappa\_j}
$$其中 $s\_i'(\\cdot)$ 是 `smooth_max_zero` 的导数（你已实现）。而

$$
\frac{\partial \text{stretch}_i}{\partial \kappa_j} = \frac{\partial l_i(\mathbf{x})}{\partial \kappa_j}
= \begin{cases}
\sum_{k\in path(i)\cap \{j\}} \frac{\partial \ell_{i,k}}{\partial \kappa_j}, & \text{若单元 } j \in path(i) \\
0, & \text{否则}
\end{cases}
$$而对单元 $k=j$，

$$\\frac{\\partial \\ell\_{i,j}}{\\partial \\kappa\_j}
\= \\frac{ \\big( e\_3 + \\widehat{\\kappa\_j} r\_i \\big)^\\top }{ | e\_3 + \\widehat{\\kappa\_j} r\_i | } \\cdot \\left( \\frac{\\partial \\big( \\widehat{\\kappa\_j} r\_i \\big) }{ \\partial \\kappa\_j } \\right) \\Delta s\_j
$$注意：

$$
\widehat{\kappa} r_i = \kappa \times r_i
$$并且对于分量 $\\kappa = [\\kappa\_x,\\kappa\_y,\\kappa\_z]$，

$$\\frac{\\partial ( \\kappa \\times r\_i )}{\\partial \\kappa\_x} = e\_x \\times r\_i, \\quad
\\frac{\\partial ( \\kappa \\times r\_i )}{\\partial \\kappa\_y} = e\_y \\times r\_i, \\quad
\\frac{\\partial ( \\kappa \\times r\_i )}{\\partial \\kappa\_z} = e\_z \\times r\_i
$$所以在向量形式中可以写成：

$$
\frac{\partial \ell_{i,j}}{\partial \kappa_j}
= \left( \frac{ e_3 + \kappa_j \times r_i }{ \| e_3 + \kappa_j \times r_i \| } \right)^\top \cdot \big[ e_x \times r_i, \; e_y \times r_i, \; e_z \times r_i \big] \, \Delta s_j
$$这给出对 $\\kappa\_j$ 的三列导数向量（或对 $\\kappa\_x,\\kappa\_y,\\kappa\_z$ 分别的偏导）。在忽略扭转的情形下（只有 $\\kappa\_x,\\kappa\_y$），取相应子向量即可。

将这些局部贡献按缆绳汇总并加权 $k\_c s\_i s\_i'$ 就得到 $U\_{cable}$ 对所有 $\\kappa\_j$ 的梯度。

### 6.4 正则项与总梯度

若加入正则项 $U\_{reg} = \\tfrac12 \\lambda |\\mathbf{x}|^2$，其梯度为 $\\lambda \\mathbf{x}$。

最终总梯度：

$$\\nabla\_{\\mathbf{x}} U = \\underbrace{[B\_j \\kappa\_j \\Delta s\_j]*j}*{\\text{弯曲项}} + \\underbrace{ \\left[ \\sum\_{cells} m,g,\\tfrac{\\partial p\_{com}}{\\partial \\kappa\_j} \\right]*j }*{\\text{重力项}} + \\underbrace{ k\_c \\sum\_i s\_i s\_i' ; \\tfrac{\\partial l\_i}{\\partial \\kappa\_j} }\_{\\text{缆绳项}} + \\lambda \\mathbf{x}
$$上式每一项在离散实现中都有明确求法：弯曲项直接；重力项与缆绳项通过灵敏度方程或局部导数公式获得。

$$
##7 \. Hessian（海森）与高效近似（Gauss-Newton）

完整海森 $H = \\nabla\_{\\mathbf{x}}^2 U$ 包含每项二次导数：

  * **弯曲项海森**是块对角：每个单元 $j$ 的贡献为 $B\_j \\Delta s\_j$（常数矩阵），直接可用。
  * **缆绳项二阶导数**较复杂（包含二阶导数的向量商项），但在常见做法里用 Gauss-Newton 近似即可：
    $$
    $$$$H\_{cable} \\approx k\_c , J\_{l}^\\top , W , J\_{l}
    $$
    $$$$其中 $J\_{l}$ 是 缆绳长度对自由度的雅可比矩阵（尺寸 \#cables × \#vars），而 $W$ 是对角权重矩阵：$W\_{ii} = s\_i'^2 + s\_i s\_i'' \\cdot (\\text{stretch}*i)$  —— 在很多实现中，取 $W*{ii}=s\_i'^2$（更稳定），或者在稳定的 smooth 函数下忽略 $s\_i''$ 项，简化为 $H\_{cable} \\approx k\_c J\_{l}^\\top J\_{l}$，即标准的 GN 近似。
  * **重力海森**通常较小（来自 $\\partial^2 p\_{com}/\\partial \\kappa^2$），可以近似忽略或用数值差分/有限近似得到。

所以常用高性能近似：

$$
H \approx H_E + k_c \, J_{l}^\top J_{l} + \lambda I
$$其中 $H\_E$ 是弯曲海森块对角矩阵。

这个近似保留了主导项并显著降低代价 —— 并且在求解牛顿步时能快速产生稳定的更新方向（Gauss-Newton / Levenberg-Marquardt 类型）。


## 8\. 任务雅可比（J\_task）通过隐函数定理的严格推导

目标：对外循环，我们要把末端位姿（或任务量）对电机位移 $\\Delta l\_{motor}$ 的导数（任务雅可比）计算出来，供 Newton-style 更新使用。

### 8.1 设定隐函数

内循环平衡满足：

$$F(\\mathbf{x}, \\Delta l\_{motor}) \\triangleq \\nabla\_{\\mathbf{x}} U(\\mathbf{x},\\Delta l\_{motor}) = 0
$$对给定 $\\Delta l\_{motor}$ 求 $\\mathbf{x}=\\mathbf{x}(\\Delta l\_{motor})$。

外部任务（例如末端位姿）由运动学映射：

$$
y(\mathbf{x}) = \text{task}(\mathbf{x}) \in \mathbb{R}^m
$$我们希望计算：

$$J\_{\\text{task}} = \\frac{\\partial y}{\\partial \\Delta l\_{motor}} = \\frac{\\partial y}{\\partial \\mathbf{x}} \\frac{\\partial \\mathbf{x}}{\\partial \\Delta l\_{motor}} = J\_{kin} ; \\frac{\\partial \\mathbf{x}}{\\partial \\Delta l\_{motor}}
$$其中 $J\_{kin} = \\partial y / \\partial \\mathbf{x}$（可通过灵敏度方程或数值差分获得）。

### 8.2 隐函数微分（隐函数定理）

对 $F(\\mathbf{x}(\\Delta l), \\Delta l) = 0$ 两边对 $\\Delta l$ 求导：

$$
\frac{\partial F}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial \Delta l} + \frac{\partial F}{\partial \Delta l} = 0
$$记 $H \\triangleq \\partial F / \\partial \\mathbf{x} = \\nabla\_{\\mathbf{x}}^2 U$（亦即总海森），并记

$$C \\triangleq \\frac{\\partial F}{\\partial \\Delta l} = \\frac{\\partial}{\\partial \\Delta l} \\nabla\_{\\mathbf{x}} U(\\mathbf{x},\\Delta l)
$$则：

$$
\frac{\partial \mathbf{x}}{\partial \Delta l} = - H^{-1} C
$$因此任务雅可比：

$$\\boxed{ ; J\_{task} = J\_{kin} ; \\left( - H^{-1} C \\right) ; }
$$\#\#\# 8.3 具体计算 C（缆绳对梯度的影响）

由于只有缆绳项直接依赖 $\\Delta l\_{motor}$，并且 $U\_{cable} = \\tfrac12 k\_c \\sum\_i s\_i^2$，其中 $s\_i = \\mathrm{smooth\_max\_zero}(\\text{stretch}\_i)$ 且 $\\text{stretch}*i = \\Delta l*{motor,i} - \\Delta l\_i(\\mathbf{x})$。

先求对 $\\Delta l\_{motor}$ 的一阶偏导：

$$
\frac{\partial U_{cable}}{\partial \Delta l_{motor}} = k_c \, \mathrm{diag}( s_i s_i' )
$$接着对梯度（即对 $\\mathbf{x}$ 的偏导）求偏导（链式）：

$$C = \\frac{\\partial}{\\partial \\Delta l\_{motor}} \\nabla\_{\\mathbf{x}} U
\= \\frac{\\partial}{\\partial \\Delta l\_{motor}} \\left( k\_c \\sum\_i s\_i s\_i' \\frac{\\partial l\_i}{\\partial \\mathbf{x}} \\right)
$$但注意 $\\partial l\_i/\\partial \\mathbf{x}$ 不依赖于 $\\Delta l\_{motor}$，因此只有 $s\_i s\_i'$ 随 $\\Delta l\_{motor}$ 变化有项：

$$
C = - k_c \; J_l^\top \; \mathrm{diag}( s_i' )  \quad \text{(签名注意：取决于定义)}
$$更直观的工程式（常用）：

$$C = - k\_c ; J\_l^\\top ; D\_s
$$其中 $J\_l$ 是缆绳长度对 $\\mathbf{x}$ 的雅可比（\#cables × \#vars），$D\_s$ 是对角矩阵，$D\_{s,ii} = s\_i'$（smooth 的一阶导）。符号细节按你对 $F=\\nabla\_x U$ 的定义会得到相应正负；在实现中直接按照链式法则编程即可。

代入：

$$
\frac{\partial \mathbf{x}}{\partial \Delta l_{motor}} = - H^{-1} C = H^{-1} \, k_c \, J_l^\top \, D_s
$$因此

$$J\_{task} = J\_{kin} , H^{-1} , k\_c , J\_l^\\top , D\_s
$$或与前述符号统一：

$$
J_{task} = J_{kin} \, (-H^{-1} C)
$$这个表达式是外循环中牛顿步所需的核心雅可比。注意实际实现中我们使用 $H$ 的近似（见第 7 节），并充分利用矩阵乘算的稀疏性与结构以加速求解。

## 9\. 数值方案与算法（伪代码）

### 9.1 内循环（静力学平衡求解）

**目标**：给定 $\\Delta l\_{motor}$，求 $\\mathbf{x}^\\star = \\arg\\min U(\\mathbf{x},\\Delta l\_{motor})$。

**伪代码**：

```python
function solve_static_equilibrium(x0, delta_l_motor, params):
x = x0
for iter in 1..max_iter:
# 1) 前向积分：给定 x（各单元kappa），用 RK4 得到 p(s), R(s)
R, p = forward_cosserat(x, params)

# 2) 计算 U, gradient g = ∇_x U  (弯曲项直接, 重力和缆绳用灵敏度方程或局部公式)
g = compute_gradient(x, R, p, delta_l_motor)
if ||g|| < tol: 
break

# 3) 近似 H: H ≈ H_E + k_c * J_l^T J_l + lambda I
H_approx = assemble_H_approx(x, R, p)

# 4) 解线性系统（或用 L-BFGS 直接）： solve H d = -g
d = solve_linear(H_approx, -g)

# 5) line search / damping
alpha = line_search(U, x, d)
x = x + alpha * d

return x
```

**实现选项**：

* 若变量维度较大（\>100），用 L-BFGS-B；若 `H_approx` 性能好，可用直接求解得到 Newton 步（更快收敛）。

### 9.2 外循环（逆运动学：PSO + Newton）

**伪代码**：

```python
function solve_ik(target_pose, config):
# Phase 1: PSO in Δl space
best_delta_l = PSO_optimize(cost=cost_function_with_inner_solver)

# Phase 2: Newton refinement
delta_l = best_delta_l
for iter in 1..max_outer_iter:
# inner solve
x = solve_static_equilibrium(x0_guess, delta_l, params)

# compute task error y(x) - target
err = task_error(x, target_pose)
if norm(err) < tol: 
return success

# compute J_kin
J_kin = compute_task_jacobian_via_sensitivities(x)

# compute H_approx and C
H_approx = assemble_H_approx(...)
C = assemble_C(...)

# compute d x / d delta_l = - H^{-1} C
DxDdl = solve_linear(H_approx, -C)   # solve H * M = -C  (matrix solve)
J_task = J_kin @ DxDdl

# compute outer update using damping: solve (J_task^T J_task + mu I) d = -J_task^T err
d_delta_l = solve_outer_linear(...)

# apply step with line_search
apply_step_with_line_search(delta_l, d_delta_l)

return result
```

### 9.3 Warm-start、Homotopy 策略（工程实用）

* **pretension continuation**：从小预紧开始，逐步增大到目标预紧，每步用前一解热启动；
* **beta (smooth) continuation**：逐步增加 $\\beta$（smooth 的陡峭度），避免 sudden nonconvexity；
* **grid warm-start**：在 workspace 采样时，引入邻点热启动策略（nearest neighbor warm start）。

## 10\. 实现细节、复杂度与参数建议

### 10.1 变量维数与计算代价

* 若每段取 $N\_{seg}$ 单元，整杆变量数 $n\_{var} = 2 \\times (N\_{pss}+N\_{c1}+N\_{c2})$（若只取 $\\kappa\_x,\\kappa\_y$）。
* 常见选择：每段 8–12 单元 → 总变量 \~ 48–72；内循环的 Hessian 近似求逆（或解线性系统）成本 $O(n\_{var}^3)$（直接法）或 $O(n\_{var}^2)$（稀疏/解法器），可采用鲁棒稀疏解算器＋并行化。

### 10.2 数值精度与稳定性

* 若 $\\kappa$ 较大，RK4 积分步长 $\\Delta s$ 需足够小以保持旋转精度；建议 $\\Delta s \\le 0.005m$（工程经验）。
* smooth 函数 `smooth_max_zero` 的 $\\beta$ 对收敛性敏感：从小到大做 continuation（例如 5→10→20→50）。

### 10.3 并行化与加速

* 前向积分与灵敏度方程可在单个样本上串行计算，但 workspace 多点采样时可在样本层面并行（多进程/多机）。
* 对于外循环 PSO，各粒子的内循环独立，天然并行化。

### 10.4 模块建议（代码组织）

* `kinematics_cosserat.py`：前向积分 + 灵敏度方程求解器（提供 `forward(x)` 和 `forward_with_sensitivities(x)` 接口）。
* `cable_mapping.py`：根据 $p(s),R(s)$ 与 $r\_i$ 计算每根缆绳长度与其对 $x$ 的雅可比（局部单元公式）。
* `statics_cosserat.py`：能量、梯度、Hessian 近似（接口：`U_and_grad(x, delta_l)`、`H_approx(x)`）。
* `solver_cosserat.py`：内循环优化（L-BFGS-B 与 GN 结合）。
* `outer_solver_cosserat.py`：PSO + Newton 逆解逻辑。
* `utils/smooth.py`：`smooth_max_zero` 与其一、二阶导实现及 beta-continuation helpers。

## 总结（要点回顾）

* Cosserat 模型从第一性原理出发，能 **严格得出缆绳长度的积分表达**，并由此计算对曲率分布的灵敏度，保证能量与梯度在物理上自洽。
* 关键数学公式：
* 缆绳长度： $l\_i = \\int\_a^b |e\_3 + \\widehat{\\kappa}(s) r\_i|, ds$
* 能量梯度的缆绳贡献通过 $\\partial \\ell\_{i,j} / \\partial \\kappa\_j$ 的显式表达计算；弯曲能梯度直接为 $B\_j \\kappa\_j \\Delta s$。
* 任务雅可比通过隐函数定理得到： $J\_{task} = J\_{kin} (-H^{-1} C)$（并给出 $C$ 的构造方式）。
* 数值实现需要处理灵敏度方程（变分方程）的数值解与矩阵方程的高效求解；工程上常借助 Gauss-Newton 近似来达到速度/准确度平衡。
* 推荐的过渡策略：先实现 Hybrid 形式（把每个 CMS 细分若干单元但仍用 PCC 形式）作为验证，再迁移到完整 Cosserat 实现并用 RK4 + 灵敏度方程做精确梯度，最终把 GN 近似与并行化结合以恢复运行效率。
$$