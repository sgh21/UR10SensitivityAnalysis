# 多源误差敏感度标定 Baseline 手册

## 目标

本项目复现 Lu 等 2025 论文中的多源误差标定主流程，用作后续改进方法的 baseline。代码把数据生成、数据读取、误差模型、敏感度分析、相关性剔除和分步辨识拆开，后续替换某一环节时不需要重写其它模块。

## 项目结构

```text
SA/
  calibration/
    data_io.py          # pkl 数据读取和保存，兼容真实数据和仿真数据
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

`redundancy.py` 对输出雅可比做秩分析，剔除相关参数。论文用 `J.T @ J` 的 SVD 描述相关性；代码保留 SVD 奇异值和条件数，同时用带列主元 QR 选择独立列，避免对 54 维参数做组合枚举。

`sensitivity.py` 使用拉丁超立方采样构造 A/B 矩阵，并按 Sobol 总敏感度思想逐列替换，得到每个参数的总敏感度。

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
- `analyze_redundancy(...)`：构造输出雅可比并返回独立参数、相关参数、秩、条件数和奇异值。
- `sobol_total_indices_lhs(...)`：输出总敏感度、归一化敏感度和参数排序。
- `make_paper_batches(...)`：按论文的高敏集合和误差类型排序生成分步辨识批次。
- `identify_stepwise_lm(...)`：对每个批次运行累计式 LM 参数辨识。

## 后续改进建议

后续改方法时，优先替换单个模块：

- 改敏感度方法：替换 `sensitivity.py`，保持输出 `SensitivityResult`
- 改相关参数剔除：替换 `redundancy.py`，保持 `independent_indices`
- 改优化器：替换 `stepwise_lm.py`，保持 `IdentificationResult`
- 换机器人：只改 `config/nominal_config.py`，必要时扩展 `robot_model.py`
