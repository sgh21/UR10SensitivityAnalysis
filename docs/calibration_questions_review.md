# 标定流程问题分析与修改建议

本文基于当前代码实现进行分析，主要参考文件包括：

- `calibration/pipeline.py`
- `calibration/redundancy.py`
- `calibration/stepwise_lm.py`
- `calibration/sensitivity.py`
- `calibration/parameters.py`
- `calibration/robot_model.py`

论文信息：Xiaoyu Lu 等，*A Novel Multisource Error Calibration Method Based on Sensitivity Analysis*，IEEE Transactions on Industrial Electronics，2025，DOI: `10.1109/TIE.2025.3585018`。

说明：下面的“29 个独立参数”结果对应当前实测数据 `dataset/real_world200.pkl`，并使用当前代码在 `sensitivity_samples=2, seed=123` 时得到的结果。冗余关系由输出雅可比列空间分析得到，不是唯一物理分解；如果敏感性样本数、随机种子、位姿数据、容差或独立参数选择策略变化，冗余参数与独立参数的对应组合也会变化。

## 1. 29 个独立参数之外，其余参数和谁冗余

当前代码的冗余分析逻辑在 `calibration/redundancy.py`：

1. 在零误差向量处对所有参数做有限差分，得到输出雅可比 `J`。
2. 构造 `T = J.T @ J`。
3. 对 `T` 做 SVD，按 `redundancy_tolerance` 判断秩 `rank`。
4. 若 `rank < 54`，只允许 `rank` 个参数参与后续辨识。
5. 当论文式组合枚举不可行时，当前代码按敏感性排序优先选择能增加雅可比秩的参数，剩余参数视为冗余。

当前实测运行结果：

```text
rank = 29
nullity = 25
independent_count = 29
redundant_count = 25
```

当前 29 个独立参数为：

```text
delta_alpha_1, delta_alpha_2, delta_alpha_3, delta_alpha_4, delta_alpha_5,
delta_a_2, delta_a_3, delta_a_4, delta_a_5, delta_a_6,
delta_d_2, delta_d_5,
delta_Btx, delta_Bty, delta_Btz, delta_Buy, delta_Buz,
delta_Ttx, delta_Tty, delta_Ttz,
delta_rrd_1, delta_rrd_2, delta_rrd_3, delta_rrd_4, delta_rrd_5,
delta_backlash_1, delta_backlash_2, delta_backlash_3, delta_backlash_4
```

其余 25 个冗余参数的近似关系如下。`best` 是与某一个独立参数列的最高相关项；`combo` 是用独立参数列空间重构该冗余参数时，贡献最大的组合项；`rel_error` 是归一化列空间重构残差，越接近 0 表示越可由当前独立参数组合解释。

| 冗余参数 | 主要冗余对象或组合 | rel_error | 解释 |
|---|---:|---:|---|
| `delta_alpha_6` | `delta_d_5:+1.000` | `3.08e-08` | 位置观测下几乎与 `delta_d_5` 等效。 |
| `delta_a_1` | `delta_Bty:+0.999, delta_Btx:-0.036, delta_Btz:+0.011` | `9.27e-10` | 第 1 连杆 x 向长度误差在当前基坐标姿态下主要表现为基座平移组合。 |
| `delta_d_1` | `delta_Btz:+1.000, delta_Btx:+0.019, delta_Bty:-0.011` | `8.54e-10` | 第 1 关节 d 偏差主要表现为基座 z 向平移组合。 |
| `delta_d_3` | `delta_d_2:+1.000` | `1.80e-09` | 当前位姿/模型下与 `delta_d_2` 近似共线。 |
| `delta_d_4` | `delta_d_2:+1.000` | `1.91e-09` | 当前位姿/模型下与 `delta_d_2` 近似共线。 |
| `delta_d_6` | `delta_Ttz:+1.000` | `2.93e-09` | 末端附近 d 偏差与工具坐标 z 平移等效。 |
| `delta_theta_1` | `delta_Buz:+1.015, delta_Buy:-0.026, delta_alpha_1:-0.012` | `8.55e-09` | 关节 1 零偏与基座姿态误差组合强相关。 |
| `delta_theta_2` | `delta_backlash_2:-1.000` | `1.88e-08` | 当前方向数据下关节 2 零偏与回差项不可分。 |
| `delta_theta_3` | `delta_backlash_3:+1.000` | `1.60e-08` | 当前方向数据下关节 3 零偏与回差项不可分。 |
| `delta_theta_4` | `delta_backlash_4:-1.000` | `8.06e-08` | 当前方向数据下关节 4 零偏与回差项不可分。 |
| `delta_theta_5` | `delta_a_6:-1.000` | `2.81e-08` | 位置观测下与末端附近长度项高度相关。 |
| `delta_theta_6` | 不可观测 | - | 末端第 6 轴旋转在当前只看 TCP 位置时不改变位置。 |
| `delta_Bux` | `delta_alpha_1:+1.000` | `3.34e-09` | 基座 x 姿态误差与第 1 个 MD-H 扭转角误差等效。 |
| `delta_Tux` | 不可观测 | - | 只看位置且当前工具偏置形式下，该工具姿态项无位置影响。 |
| `delta_Tuy` | 不可观测 | - | 同上。 |
| `delta_Tuz` | 不可观测 | - | 同上。 |
| `delta_rrd_6` | 不可观测 | - | 第 6 轴角度变化在当前只看位置时不可观测。 |
| `delta_backlash_5` | `delta_a_6:+1.000` | `6.64e-08` | 当前位姿/位置观测下与末端附近长度项高度相关。 |
| `delta_backlash_6` | 不可观测 | - | 第 6 轴回差对当前 TCP 位置无影响。 |
| `delta_flex_1` | 不可观测 | - | 当前实测数据 `payload=0`，柔顺项 `tau * delta_flex` 为 0。 |
| `delta_flex_2` | 不可观测 | - | 同上。 |
| `delta_flex_3` | 不可观测 | - | 同上。 |
| `delta_flex_4` | 不可观测 | - | 同上。 |
| `delta_flex_5` | 不可观测 | - | 同上。 |
| `delta_flex_6` | 不可观测 | - | 同上。 |

关键结论：

- 冗余不是“某参数一定物理上等于另一个参数”，而是“在当前数据、当前观测量、当前名义模型和当前容差下，这些参数对输出位置的影响列线性相关或不可观测”。
- 只使用末端位置，不使用姿态，会天然导致第 6 轴相关角度项、工具姿态项更容易不可观测。
- `payload=0` 时，所有柔顺参数必然不可辨识，因为 `tau = 0`，模型里的 `tau * delta_flex` 恒为 0。

## 2. 五个辨识阶段如何划分，以及后阶段是否协同优化

阶段划分由 `calibration/stepwise_lm.py` 中的 `make_paper_batches(...)` 决定。

当前逻辑：

1. 先取冗余分析后的独立参数集合。
2. 在独立参数集合内，按 Sobol 总敏感度归一化分数 `normalized_scores` 从高到低排序。
3. 第 1 阶段取高敏参数集合 `H`：
   - 从最高敏参数开始累加；
   - 当累计敏感度达到 `high_cumulative_score * total_independent_score` 时停止；
   - 当前默认 `high_cumulative_score = 0.80`。
4. 剩余独立参数按 `parameter.group` 分组：
   - `kinematic`
   - `frame`
   - `reduction_ratio`
   - `backlash`
   - `flexibility`
5. 各组按组内敏感度总和从大到小排序，依次作为后续阶段。

当前五个阶段为：

```text
Stage 1:
delta_a_2, delta_d_5, delta_a_6, delta_d_2, delta_a_3,
delta_Buy, delta_Btx, delta_Ttx

Stage 2:
delta_Btz, delta_Tty, delta_Bty, delta_Ttz, delta_Buz

Stage 3:
delta_alpha_2, delta_a_4, delta_alpha_1, delta_alpha_3,
delta_alpha_4, delta_a_5, delta_alpha_5

Stage 4:
delta_backlash_3, delta_backlash_2, delta_backlash_4, delta_backlash_1

Stage 5:
delta_rrd_2, delta_rrd_3, delta_rrd_5, delta_rrd_4, delta_rrd_1
```

对应阶段误差：

| 阶段 | 新增参数数 | 累计参数数 | 欧氏 RMSE mm |
|---:|---:|---:|---:|
| 1 | 8 | 8 | 1.755 |
| 2 | 5 | 13 | 0.946 |
| 3 | 7 | 20 | 0.216 |
| 4 | 4 | 24 | 0.209 |
| 5 | 5 | 29 | 0.163 |

后一个阶段是否加入之前的参数协同优化？

是。`identify_stepwise_lm(...)` 中的核心逻辑是：

```python
active = _unique(active + list(batch))
x0 = full[active].copy()
least_squares(... active ...)
full[active] = result.x
```

因此每个新阶段都会把新参数加入 `active`，然后对“之前所有 active 参数 + 当前新增参数”一起重新优化。它不是只优化新增参数。

## 3. 为什么仿真和真实数据的独立参数个数不一致

这件事不一定不合理。当前代码中的独立参数个数本质上等于当前数据集输出雅可比的数值秩，而雅可比秩依赖以下因素：

1. 位姿样本分布  
   实测数据的关节范围较窄，仿真数据通常按 `joint_limits` 随机采样，激励更充分，因此仿真 rank 可能更高。

2. 负载条件  
   当前实测 `payload=0`，所有柔顺参数不可观测。仿真如果设置 `payload=20`，柔顺参数会被激励，rank 会增加。

3. 到位方向 `directions`  
   回差项是 `direction * delta_backlash`。如果真实数据方向变化不足，`delta_theta` 和 `delta_backlash` 容易相关；仿真中方向随机时两者更容易区分。

4. 只使用位置观测  
   如果只观测 TCP 位置，不观测姿态，第 6 轴旋转、工具姿态等参数天然容易不可观测。

5. 数值容差和参数尺度  
   当前 rank 判定使用 `T = J.T @ J` 的相对奇异值阈值。不同数据的奇异值谱不同，会导致 rank 变化。

6. 当前代码的独立集选择受敏感性排序影响  
   代码先跑敏感性，再把 `sensitivity.ranked_indices` 传给冗余分析作为优先级。因此相同 rank 下，独立参数集合会随 Sobol 采样、随机种子和样本数变化。

快速对比结果：

```text
real payload=0:        rank=29, independent=29
simulation payload=0:  rank=33, independent=33
simulation payload=20: rank=37, independent=37
```

更合理的修改建议：

1. 如果目标是比较仿真和真实算法表现，应使用一致的位姿集合、负载、方向数据和观测类型。
2. 如果真实实验要辨识柔顺参数，必须加入非零负载数据；否则 `delta_flex_*` 应直接从参数集中禁用。
3. 冗余分析应优先使用按参数先验尺度归一化后的雅可比，例如 `J_scaled = J @ diag(parameter_scales)`，避免不同单位参数造成秩判定偏置。
4. 独立参数选择最好固定一个确定性优先级，而不是完全依赖随机 Sobol 结果；可以采用“论文参数顺序 + 组优先级 + QR”或“敏感性均值排序 + 固定 seed + 足够样本数”。
5. rank 阈值建议通过奇异值谱断点诊断，不建议长期固定为 `1e-7`。
6. 对真实数据应设计更丰富的标定位姿，特别是覆盖第 5/6 轴姿态变化、正反向到位和多负载工况。

## 4. 当前实现是否与论文冲突，是否一比一复现

当前实现是“按论文思路搭建的 baseline”，不是严格一比一复现。

与论文一致或基本对齐的部分：

- 构建多源误差参数，包括 MD-H 运动学误差、基座/工具坐标误差、减速比、回差和柔顺参数。
- 使用输出雅可比和 `T = J.T @ J` 做相关/冗余参数分析。
- 使用 LHS/Sobol 思路计算参数敏感性。
- 根据敏感性分阶段进行 LM 参数辨识。
- 每个后续阶段累计已有参数协同优化。

与论文可能不完全一致或需要注意的地方：

1. 冗余分析 fallback 不是论文的严格组合枚举  
   论文判据需要枚举或判断可移除相关参数集合。当前 54 维时组合数太大，代码使用“敏感性优先 + rank-preserving”的工程 fallback。这是合理工程近似，但不是严格一比一。

2. 当前 pipeline 先算敏感性，再用敏感性排序辅助选独立参数  
   如果论文流程是先确定独立参数再做敏感性排序，则当前顺序与论文存在差异。

3. 敏感性函数 `f(Dv)` 被实现为“扰动模型相对名义模型的整轨迹 RMS 位置偏移”  
   这是因为当前数据只有位置测量。如果论文使用完整位姿误差或其他影响函数，则不是一比一。

4. 当前 LM 残差只包含位置 xyz，不包含姿态  
   这会造成第 6 轴、工具姿态等参数不可观测。若论文实验同时使用姿态或更完整的测量，该实现会有差异。

5. 参数先验范围是代码中人为设定的 `scale` 和 `±4*scale`  
   如果论文中各参数容差范围不同，Sobol 排名会发生变化。

6. 当前真实数据 `payload=0`，柔顺参数无法辨识  
   如果论文强调重载/柔顺误差辨识，则当前真实数据实验并未覆盖该部分。

7. 当前名义 MD-H 参数需要复核  
   `config/nominal_config.py` 中存在非常大的 `d` 项，这会显著放大 `delta_alpha_*` 的敏感性，可能不是论文设置。

结论：当前代码可以称为“论文方法的工程化近似实现/基线复现”，不应称为严格一比一复现。若论文复现要求严格，需要补齐论文中的真实参数范围、完整位姿误差定义、冗余组合判据、实验位姿/负载设计和输出指标。

## 5. 如何跳过冗余性分析和敏感性分析，做全参数 LM 原始对比

目标：保留当前 stepwise LM 函数，新增一个模式，让 pipeline 可以选择：

- `sobol_stepwise`：当前方法，冗余分析 + Sobol 敏感性 + 分阶段 LM。
- `full_lm`：跳过冗余分析和敏感性分析，直接 54 参数一次性 LM。

最小修改建议如下。

### 5.1 修改 `CalibrationConfig`

在 `calibration/pipeline.py` 的 `CalibrationConfig` 中增加字段：

```python
optimization_mode: str = "sobol_stepwise"
```

### 5.2 在 `run_calibration_on_dataset(...)` 中分支

把当前的 sensitivity/redundancy/batches 构造部分改成：

```python
if cfg.optimization_mode == "full_lm":
    sensitivity = _empty_sensitivity(len(parameters))
    redundancy = _full_parameter_redundancy(len(parameters))
    batches = [list(range(len(parameters)))]
elif cfg.optimization_mode == "sobol_stepwise":
    sensitivity = sobol_total_indices_lhs(
        model,
        joints,
        parameters,
        payloads=payloads,
        directions=directions,
        n_samples=cfg.sensitivity_samples,
        seed=cfg.seed,
    )
    redundancy = analyze_redundancy(
        model,
        joints,
        zero,
        parameters,
        payloads=payloads,
        directions=directions,
        tolerance=cfg.redundancy_tolerance,
        max_combinations=cfg.redundancy_max_combinations,
        preferred_indices=sensitivity.ranked_indices,
    )
    batches = make_paper_batches(
        parameters=parameters,
        ranked_indices=sensitivity.ranked_indices,
        scores=sensitivity.normalized_scores,
        independent_indices=redundancy.independent_indices,
        high_cumulative_score=cfg.high_cumulative_score,
    )
else:
    raise ValueError(f"Unknown optimization_mode: {cfg.optimization_mode}")
```

### 5.3 增加两个轻量 helper

仍放在 `calibration/pipeline.py` 中即可：

```python
def _empty_sensitivity(count: int) -> SensitivityResult:
    zeros = np.zeros(count, dtype=float)
    return SensitivityResult(
        first_order_indices=zeros.copy(),
        total_indices=zeros.copy(),
        normalized_scores=zeros.copy(),
        ranked_indices=list(range(count)),
        output_variance=0.0,
    )


def _full_parameter_redundancy(count: int) -> RedundancyResult:
    empty_j = np.zeros((0, count), dtype=float)
    empty_t = np.zeros((count, count), dtype=float)
    return RedundancyResult(
        jacobian=empty_j,
        normal_matrix=empty_t,
        nullspace=np.zeros((count, 0), dtype=float),
        correlated_sets=[],
        independent_indices=list(range(count)),
        redundant_indices=[],
        rank=count,
        nullity=0,
        condition_number=0.0,
        singular_values=np.zeros(0, dtype=float),
        normal_singular_values=np.zeros(0, dtype=float),
        used_exhaustive_search=False,
    )
```

### 5.4 CLI 增加模式参数

在 `run_real_world.py` 和 `run_simulation.py` 中增加：

```python
parser.add_argument(
    "--optimization-mode",
    choices=("sobol_stepwise", "full_lm"),
    default="sobol_stepwise",
    help="calibration mode",
)
```

构造 config 时改为：

```python
config=CalibrationConfig(
    sensitivity_samples=args.sensitivity_samples,
    seed=args.seed,
    optimization_mode=args.optimization_mode,
)
```

### 5.5 使用方式

当前方法：

```powershell
python run_real_world.py dataset\real_world200.pkl --optimization-mode sobol_stepwise
```

全参数 LM 对比：

```powershell
python run_real_world.py dataset\real_world200.pkl --optimization-mode full_lm
```

注意：

- `full_lm` 会一次性优化 54 个参数，病态和过拟合风险很高。
- 当前实测数据只看位置且 `payload=0`，全参数 LM 中很多参数实际上不可观测，得到的参数值不应解释为物理真实误差。
- `full_lm` 可以作为误差下降能力的原始对比，但不适合作为参数可解释性的结论。

