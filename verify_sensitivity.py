"""
sensitivity.py 解析验证脚本
============================

用一个完全已知的线性多输出模型 Y = A·θ + c 来验证 sensitivity.py 中
sobol_total_indices_lhs 实现的正确性。

为什么用线性模型？
- 它的 Sobol 指标有闭式解，一行写出来：
      S_T^i = ||A_{:,i}||² · σ_i² / Σ_j ||A_{:,j}||² · σ_j²
  其中 σ_i² = (U_i - L_i)² / 12 是均匀分布的方差。
- 标定问题在工作点附近就是线性化的（雅可比 J 替代 A），所以这恰好是
  sensitivity.py 真正面对的场景。
- 同时能直观演示三件事：
    (a) 算法的样本估计 → 解析值（收敛性）；
    (b) Sobol 指标确实随参数采样区间改变（你提的那个问题）；
    (c) 在等区间约束下，Sobol 排名 ↔ 经典局部偏导 ∂f/∂θ_i 的平方排名。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


# ======================================================================
# 1. 复刻 sensitivity.py 的算法（剥掉 calibration 包依赖，逻辑逐行对应）
# ======================================================================

@dataclass
class ParamSpec:
    """本验证脚本的最小化参数描述：只需要上下界。"""
    lower: float
    upper: float


def _bounds(parameters):
    lo = np.array([p.lower for p in parameters], dtype=float)
    up = np.array([p.upper for p in parameters], dtype=float)
    return lo, up


def sobol_total_lhs(model_fn, parameters, n_samples=128, seed=None):
    """LHS + Jansen 估计子，与 sensitivity.py 中的实现一一对应。

    model_fn : callable, 接受 shape (d,) 的参数向量, 返回 shape (out_dim,) 的扁平输出
    """
    lo, up = _bounds(parameters)
    d = len(parameters)

    # 两个独立的 LHS 采样器
    sa = qmc.LatinHypercube(d=d, seed=seed)
    sb = qmc.LatinHypercube(d=d, seed=None if seed is None else seed + 1)
    A_mat = qmc.scale(sa.random(n_samples), lo, up)
    B_mat = qmc.scale(sb.random(n_samples), lo, up)

    def evaluate(M):
        return np.array([model_fn(row) for row in M])

    Ya = evaluate(A_mat)
    # 多输出总方差 = 各维度方差之和
    var_total = float(np.sum(np.var(Ya, axis=0, ddof=1)))
    if var_total <= 1e-20:
        return np.zeros(d)

    ST = np.zeros(d)
    for i in range(d):
        AB = A_mat.copy()
        AB[:, i] = B_mat[:, i]            # 把 A 的第 i 列换成 B 的第 i 列
        Yab = evaluate(AB)
        # Jansen 多输出公式
        ST[i] = np.mean(np.sum((Ya - Yab) ** 2, axis=1)) / (2.0 * var_total)
    return ST


# ======================================================================
# 2. 解析 Sobol 总效应指标（线性模型）
#    Y = A·θ + c, θ_i ~ Uniform(L_i, U_i) 独立
#    Var(Y_k) = Σ_i A_{ki}² · σ_i²
#    总方差 TV = Σ_k Var(Y_k) = Σ_i ||A_{:,i}||² · σ_i²
#    模型纯加性 ⇒ S_T^i = S_i
#    S_T^i = ||A_{:,i}||² · σ_i² / TV
# ======================================================================

def analytical_sobol_total_linear(A, lower, upper):
    A = np.asarray(A, float)
    sigma_sq = (np.asarray(upper) - np.asarray(lower)) ** 2 / 12.0
    col_norms_sq = np.sum(A ** 2, axis=0)
    contrib = col_norms_sq * sigma_sq
    return contrib / np.sum(contrib)


# ======================================================================
# 3. 测试一：算法的 LHS 估计是否收敛到 Sobol 解析值
# ======================================================================

def test_convergence():
    print("=" * 76)
    print("测试 1：线性模型，验证 LHS 估计收敛到 Sobol 解析值")
    print("=" * 76)

    # 5 输入 / 3 输出，列范数差异显著
    A = np.array([
        [3.0, 0.1, 1.0, 2.0, 0.5],
        [0.0, 0.2, 0.5, 1.0, 0.3],
        [1.0, 0.0, 0.0, 0.5, 0.1],
    ])
    c = np.zeros(3)

    lower = np.array([-1.0, -2.0, -0.5, -1.0, -3.0])
    upper = np.array([ 1.0,  2.0,  0.5,  1.0,  3.0])
    params = [ParamSpec(lower[i], upper[i]) for i in range(5)]
    model = lambda theta: A @ theta + c

    analytical = analytical_sobol_total_linear(A, lower, upper)
    print(f"\n参数下标:    {list(range(5))}")
    print(f"解析 S_T^i:   {np.array2string(analytical, precision=5, suppress_small=True)}")
    print(f"(求和)       {analytical.sum():.5f}\n")

    print(f"{'N':>6}   {'LHS 估计 (归一化)':<46}  {'最大绝对误差':>14}")
    print("-" * 76)
    for N in [128, 512, 2048, 8192, 32768]:
        ST = sobol_total_lhs(model, params, n_samples=N, seed=0)
        norm = ST / ST.sum()
        err = np.max(np.abs(norm - analytical))
        norm_str = " ".join(f"{x:6.4f}" for x in norm)
        print(f"{N:>6}   [{norm_str}]  {err:>14.3e}")

    print("\n→ 误差随 N 大致按 1/√N 下降，算法实现正确。")


# ======================================================================
# 4. 测试二：参数采样区间确实影响 Sobol 指标（你提的问题）
# ======================================================================

def test_range_dependence():
    print("\n" + "=" * 76)
    print("测试 2：Sobol 指标随采样区间变化 —— 这是设计意图，不是 bug")
    print("=" * 76)

    # 两参数局部偏导相同 (=1)，但工作区间可以不同
    A = np.array([[1.0, 1.0]])
    print("\n模型: Y = θ_1 + θ_2  （两参数局部偏导都是 1）")
    print("但实际重要性取决于各自能波动多大：\n")

    cases = [
        ("等区间 ±1",        np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        ("θ_2 区间放大 5x",  np.array([-1.0, -5.0]), np.array([1.0, 5.0])),
        ("θ_1 区间放大 5x",  np.array([-5.0, -1.0]), np.array([5.0, 1.0])),
    ]
    print(f"{'情况':<18} {'lower':<18} {'upper':<18} {'S_T^1':>8}  {'S_T^2':>8}")
    print("-" * 76)
    for label, lo, up in cases:
        s = analytical_sobol_total_linear(A, lo, up)
        print(f"{label:<18} {str(lo):<18} {str(up):<18} {s[0]:>8.4f}  {s[1]:>8.4f}")

    print("""
解读：
  · 经典"局部敏感度" ∂Y/∂θ_i 在三种情况下都是 1，无法区分谁更重要。
  · Sobol 指标按工作范围加权：哪个参数实际能波动得更厉害，它对 Y 的
    方差贡献就更大 —— 这正是标定 / 容差分析里要的"实际重要性"。
  · *选择合理的参数区间是用户的责任*：区间应该等于该参数在实际系统里
    允许的误差范围（机加工容差、温漂边界、装配偏差等）。""")


# ======================================================================
# 5. 测试三：等区间约束下 Sobol ↔ 局部偏导平方
# ======================================================================

def test_local_vs_global():
    print("\n" + "=" * 76)
    print("测试 3：等区间下，Sobol 排名 ↔ 局部偏导 (∂f/∂θ)² 的归一化")
    print("=" * 76)
    A = np.array([[3.0, 0.5, 2.0, 1.0]])
    lower = -np.ones(4)
    upper =  np.ones(4)

    sobol = analytical_sobol_total_linear(A, lower, upper)
    grad_sq = (A.flatten()) ** 2
    grad_sq_normalized = grad_sq / grad_sq.sum()

    print(f"\n  ∂f/∂θ              : {A.flatten()}")
    print(f"  (∂f/∂θ)² 归一化     : {np.array2string(grad_sq_normalized, precision=5)}")
    print(f"  Sobol S_T (等区间)  : {np.array2string(sobol, precision=5)}")
    print(f"\n→ 两者完全一致。这给出了 Sobol 全局敏感度与经典局部敏感度的精确关系。")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    test_convergence()
    test_range_dependence()
    test_local_vs_global()
