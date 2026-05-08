# 多源误差敏感度标定 Baseline 手册

## 目标

本项目复现 Lu 等 2025 论文中的多源误差标定主流程，用作后续改进方法的 baseline。代码把数据生成、数据读取、误差模型、敏感度分析、相关性剔除和分步辨识拆开，后续替换某一环节时不需要重写其它模块。

## 项目结构

```text
SA/
  calibration/
    data_io.py          # pkl 数据读取和保存，兼容真实数据和仿真数据
    evaluation.py       # 详细评估，和主辨识逻辑解耦
    metrics.py          # 位置误差统计
    parameters.py       # 论文 54 维多源误差参数定义
    pipeline.py         # 两个主接口和公共 baseline 流程
    redundancy.py       # 基于雅可比秩的相关参数分析
    robot_model.py      # MD-H 多源误差正运动学模型
    sensitivity.py      # LHS + Sobol 总敏感度分析
    stepwise_lm.py      # 分步 Levenberg-Marquardt 参数辨识
    transforms.py       # 位姿变换工具
  config/
    nominal_config.py   # 名义机器人参数和关节采样范围
  simulation/
    generator.py        # 仿真数据生成和打包
  docs/
    MANUAL.md           # 本手册
  run_simulation.py     # 主接口 1：仿真验证
  run_real_world.py     # 主接口 2：真实数据辨识
  main.py               # 默认运行仿真验证
  requirements.txt
```

## 论文方法对应关系

`parameters.py` 定义 54 个参数：

- `kinematic`：24 个 MD-H 参数误差，包含 `delta_alpha/a/d/theta_1..6`
- `frame`：12 个基座和工具坐标系误差，包含 `delta_Bt/Bu` 与 `delta_Tt/Tu`
- `reduction_ratio`：6 个减速比误差 `delta_rrd_1..6`
- `backlash`：6 个关节回差 `delta_backlash_1..6`
- `flexibility`：6 个关节柔顺系数 `delta_flex_1..6`

`robot_model.py` 实现统一误差模型：

```text
q_eff = q + delta_theta + q * delta_rrd
        + direction * delta_backlash
        + tau * delta_flex
tau = Jv.T @ [0, 0, -m*g]
```

`redundancy.py` 对输出雅可比做秩分析，剔除相关参数。实现与论文 II-B 对齐：先构造 `T = J.T @ J`，对 `T` 做 SVD，取最后 `n-r` 个右奇异向量作为零空间。候选相关参数集合必须同时满足 `rank(V_tilde)=n-r` 和移除对应列后的 `rank(T_tilde)=r`。当组合数量可控时执行严格穷举；当 54 维全参数导致组合爆炸时，使用同一零空间满秩判据的主元选择路径得到一个可移除集合。

`sensitivity.py` 与论文 II-C 对齐：使用拉丁超立方采样构造 `A/B` 矩阵，逐列构造 `A_B^(i)`，同时计算 Eq. (21) 的一阶敏感度 `r_i` 和 Eq. (22) 的总敏感度 `s_i`。当前数据只包含位置测量，因此影响函数 `f(Dv)` 定义为整条采样轨迹上“扰动模型相对名义模型”的 RMS 位置偏移，排序使用总敏感度。

`stepwise_lm.py` 按论文 Algorithm 1 组织辨识顺序：先取累计敏感度达到阈值 `g` 的高敏参数集合 `H`，再把剩余参数按误差类型的敏感度和排序，逐步加入 LM 优化。

## 数据格式

真实数据 pkl 至少需要包含：

```python
{
    "joints": ndarray[N, 6],              # 或 joint_configs / q
    "measured_positions": ndarray[N, 3],  # 或 laser_points / positions / points
}
```

可选字段：

```python
{
    "payloads": ndarray[N] 或 float,       # 负载质量 kg
    "directions": ndarray[N, 6],          # 各关节到位方向，取 -1 或 1
}
```

仿真数据会额外保存：

```python
{
    "true_error_vector": ndarray[54],
    "true_error_parameters": dict,
    "parameter_names": list[str],
    "parameter_groups": list[str],
    "nominal_positions": ndarray[N, 3],
    "true_positions": ndarray[N, 3],
    "nominal_robot": dict,
    "metadata": dict,
}
```

## 使用方法

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行仿真验证：

```powershell
python run_simulation.py --output outputs/synthetic_dataset.pkl --samples 120 --payload 20 --noise 0.00002 --sensitivity-samples 128
```

运行真实数据辨识：

```powershell
python run_real_world.py D:\path\to\real_world.pkl --sensitivity-samples 128
```

在 Python 中调用两个主接口：

```python
from calibration.pipeline import run_simulation_validation, run_real_identification

sim_result = run_simulation_validation("outputs/synthetic_dataset.pkl")
real_result = run_real_identification(r"D:\path\to\real_world.pkl")
```

## 主要函数说明

- `generate_synthetic_dataset(...)`：生成名义机器人、采样真实误差、计算真实测量点，并把真值一起打包到 pkl。
- `load_dataset(path)`：把真实或仿真 pkl 统一成 baseline 需要的 schema。
- `run_simulation_validation(...)`：仿真主入口，先生成数据，再调用同一套辨识 pipeline。
- `run_real_identification(path, config)`：真实数据主入口，只读取数据并辨识。
- `run_calibration_on_dataset(dataset, config)`：公共 pipeline，保证仿真和真实数据的辨识完全解耦。
- `build_evaluation_report(result, top_k)`：详细评估入口，输出位置误差、逐轴误差、辨识过程信息；仿真数据额外输出参数真值对比。
- `analyze_redundancy(...)`：构造输出雅可比并返回独立参数、相关参数、秩、条件数和奇异值。
- `sobol_total_indices_lhs(...)`：输出一阶敏感度、总敏感度、归一化总敏感度和参数排序。
- `make_paper_batches(...)`：按论文的高敏集合和误差类型排序生成分步辨识批次。
- `identify_stepwise_lm(...)`：对每个批次运行累计式 LM 参数辨识。

## 后续改进建议

后续改方法时，优先替换单个模块：

- 改敏感度方法：替换 `sensitivity.py`，保持输出 `SensitivityResult`
- 改相关参数剔除：替换 `redundancy.py`，保持 `independent_indices`
- 改优化器：替换 `stepwise_lm.py`，保持 `IdentificationResult`
- 换机器人：只改 `config/nominal_config.py`，必要时扩展 `robot_model.py`

## 详细评估输出

命令行默认输出 `evaluation` 字段，评估代码集中在 `calibration/evaluation.py`，没有混入模型、敏感度和 LM 主逻辑。

真实数据和仿真数据都会输出：

- `evaluation.data`：样本数量、关节范围、负载范围、是否含仿真真值。
- `evaluation.position.norm_before / norm_after`：标定前后欧氏位置误差的 mean、rmse、median、p95、max、std。
- `evaluation.position.improvement_percent`：上述指标的改善百分比。
- `evaluation.position.axis_before / axis_after`：x/y/z 三个方向的 bias、mean_abs、rmse、max_abs、std。
- `evaluation.position.axis_rmse_improvement_percent`：各轴 RMSE 改善百分比。
- `evaluation.position.worst_samples_after`：标定后误差最大的若干样本及其 xyz 残差。
- `evaluation.identification`：参数秩、条件数、相关参数、分步批次和每步 LM 收敛信息。
- `evaluation.identification.used_exhaustive_redundancy_search`：冗余参数搜索是否执行了论文组合穷举；如果为 `false`，说明组合数量过大，使用零空间主元选择得到一个满足论文满秩条件的相关参数集合。
- `top_sensitivity`：每个高敏参数包含 `first_order_index`、`total_index` 和归一化排序分数 `score`。

如果是仿真数据，并且 pkl 中存在 `true_error_vector`，还会输出：

- `evaluation.parameter_truth.subsets`：全部参数、实际参与辨识参数、独立参数、未辨识参数的 MAE、RMSE、最大绝对误差和按先验尺度归一化后的误差。
- `evaluation.parameter_truth.by_group`：按 `kinematic`、`frame`、`reduction_ratio`、`backlash`、`flexibility` 分组统计真值和估计值差异。
- `evaluation.parameter_truth.per_parameter`：54 个参数逐一列出 true、estimated、error、abs_error、normalized_abs_error。
- `evaluation.parameter_truth.largest_absolute_errors`：绝对误差最大的若干参数，数量由 `--top-k` 控制。

示例：

```powershell
python run_simulation.py --samples 120 --payload 20 --top-k 15
python run_real_world.py D:\path\to\real_world.pkl --top-k 15
```
