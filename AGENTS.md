# AGENTS.md

必须先读本文件，再进行任何仓库扫描、源码阅读、运行命令或代码修改。

本文件是给后续 AI 编码会话的项目手册。先读这里，再按需打开源码；不要一上来全仓扫盘。

## 30 秒概览

这是一个复现/工程化 Lu 等 2025 多源误差敏感度标定方法的 Python baseline。目标是在真实或仿真位姿数据上识别 6 轴机器人 54 维误差参数，并输出标定前后位置误差、敏感度排序、冗余参数、分阶段 LM 识别过程和图表。

核心流水线在 `calibration/pipeline.py`：

1. 读取或生成数据集：`joints[N,6]`、`measured_positions[N,3]`，可选 `payloads`、`directions`。
2. 构建 54 维误差参数：`calibration/parameters.py`。
3. 用 `MultiSourceRobotModel` 做名义/带误差正运动学：`calibration/robot_model.py`。
4. `sobol_stepwise` 模式：LHS/Sobol 敏感度排序 -> Jacobian/SVD 冗余分析 -> 按论文风格分批 -> 累计式最小二乘。
5. `full_lm` 模式：跳过敏感度和冗余分析，一次性优化全部 54 个参数，主要用于对比，不适合解释物理参数。
6. 生成 summary、文本报告、可选图表。

## 常用命令

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

仿真验证：

```powershell
python run_simulation.py --output outputs/synthetic_dataset.pkl --samples 120 --payload 20 --noise 0.00002 --sensitivity-samples 128
```

真实数据识别：

```powershell
python run_real_world.py dataset\real_world200.pkl --sensitivity-samples 128
```

跳过 Sobol/冗余，直接全参数 LM：

```powershell
python run_real_world.py dataset\real_world200.pkl --optimization-mode full_lm
```

快速入口：`python main.py` 等价于运行仿真入口 `run_simulation.py`。

依赖很少：`numpy`、`scipy`、`matplotlib`。项目没有测试框架配置，现有验证脚本是 `verify_sensitivity.py` 和 `scripts/verify_dh_to_mdh.py`。

## 目录职责

- `calibration/`：标定核心代码。
- `calibration/data_io.py`：加载/保存 pkl 数据，统一真实和仿真 schema。
- `calibration/parameters.py`：定义论文风格 54 维多源误差向量、尺度、边界、拆包。
- `calibration/robot_model.py`：MDH 正运动学和多源误差模型。
- `calibration/sensitivity.py`：Latin Hypercube + Sobol 一阶/总效应敏感度估计。
- `calibration/redundancy.py`：输出 Jacobian、`T = J.T @ J`、SVD rank/nullspace、冗余参数选择。
- `calibration/stepwise_lm.py`：按敏感度和参数组生成批次，并累计式调用 `scipy.optimize.least_squares`。
- `calibration/pipeline.py`：真实/仿真两个主接口和公共标定流程。
- `calibration/evaluation.py`：详细评估报告，包含位置误差、阶段误差、仿真真值对比。
- `calibration/reporting.py`：把 summary 格式化成 CLI 文本表格。
- `calibration/visualization.py`：生成 PNG 图表和 plots manifest。
- `calibration/transforms.py`：4x4 变换、MDH link transform、位置提取。
- `config/nominal_config.py`：名义机器人 MDH、基座/工具变换、关节采样范围。
- `simulation/generator.py`：生成带真值误差向量的仿真 pkl 数据。
- `run_simulation.py`：生成仿真数据并标定。
- `run_real_world.py`：读取实测 pkl 并标定。
- `docs/`：手册和问题分析。当前终端中部分中文显示为乱码，源码和本文件更可靠。
- `outputs/`：生成物，已被 `.gitignore` 忽略。

## 数据格式

`load_dataset(path)` 接受 dict 或 list pkl。

最小 dict schema：

```python
{
    "joints": ndarray[N, 6],              # 也兼容 joint_configs / q
    "measured_positions": ndarray[N, 3],  # 也兼容 laser_points / positions / points
}
```

可选字段：

```python
{
    "payloads": ndarray[N] 或 float,      # kg；缺省为 0
    "directions": ndarray[N, 6],          # 每关节到位方向，符号化为 -1/1
}
```

仿真数据额外包含 `true_error_vector`、`true_error_parameters`、`parameter_names`、`parameter_groups`、`nominal_positions`、`true_positions`、`nominal_robot`、`metadata`。

注意：当前模型只使用 TCP 位置 `xyz`，没有姿态残差。因此第 6 轴相关角度项、工具姿态项等可能天然不可观测。

## 54 维参数顺序

`build_error_parameters()` 的顺序是稳定契约，很多结果只存 index：

1. `delta_alpha_1..6`，kinematic，rad，scale `7e-4`
2. `delta_a_1..6`，kinematic，m，scale `14e-4`
3. `delta_d_1..6`，kinematic，m，scale `14e-4`
4. `delta_theta_1..6`，kinematic，rad，scale `7e-4`
5. `delta_Btx/y/z`，frame，m，scale `21e-4`
6. `delta_Bux/y/z`，frame，rad，scale `7e-4`
7. `delta_Ttx/y/z`，frame，m，scale `7e-4`
8. `delta_Tux/y/z`，frame，rad，scale `3.5e-4`
9. `delta_rrd_1..6`，reduction_ratio，ratio，scale `5e-5`
10. `delta_backlash_1..6`，backlash，rad，scale `3e-4`
11. `delta_flex_1..6`，flexibility，rad/Nm，scale `5e-7`

每个参数边界是 `[-4*scale, +4*scale]`，用于 Sobol/LHS 采样；`parameter_scales()` 同时用于有限差分和优化尺度。

## 正运动学和误差模型

`MultiSourceRobotModel.transform()` 是核心物理模型：

```text
q_eff = q
      + nominal.theta_offset
      + delta_theta
      + q * delta_rrd
      + direction * delta_backlash
      + tau * delta_flex
```

随后计算：

```text
T = T_base_error
    @ Π_i MDH_i(alpha_i + delta_alpha_i,
                a_i + delta_a_i,
                q_eff_i,
                d_i + delta_d_i)
    @ T_tool_error
```

其中 `modified_dh_transform(alpha, a, theta, d)` 使用本项目约定：

```text
Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)
```

`payload=0` 时 `joint_load_torque()` 返回零，所有 `delta_flex_*` 对输出无影响，不能被识别。`directions` 缺省时按 `sign(q)` 推断，0 视为 +1；如果实测数据方向信息不可靠，`delta_theta_*` 和 `delta_backlash_*` 容易相关。

## 算法链路

### `sobol_stepwise`

默认 `CalibrationConfig.optimization_mode == "sobol_stepwise"`。

1. `sobol_total_indices_lhs()`：
   - 用两个 LHS 矩阵 `A/B` 在参数边界内采样。
   - 对每个样本计算扰动模型相对零误差模型的整条轨迹 RMS 位置偏移。
   - 逐列 swap 得到一阶指数和总效应指数。
   - 用归一化总效应 `normalized_scores` 降序生成 `ranked_indices`。
2. `analyze_redundancy()`：
   - 在当前误差向量处有限差分 `model.batch_positions(...).reshape(-1)` 得到输出 Jacobian。
   - 构造 `normal_matrix = J.T @ J`，对其 SVD。
   - `rank = count(s > tolerance * s0)`，`nullity = n_params - rank`。
   - 组合数可控时穷举论文式 removable set；否则先尝试 nullspace pivot，再用“敏感度优先 + rank-preserving QR”选择一组 independent indices。
3. `make_paper_batches()`：
   - 只保留 `independent_indices`。
   - 第一批是累计敏感度达到 `high_cumulative_score` 默认 0.80 的高敏参数。
   - 其余参数按 `parameter.group` 分组，并按组内敏感度总和降序追加。
4. `identify_stepwise_lm()`：
   - 每阶段是累计优化：新 batch 加入 `active` 后，之前 active 参数会一起重新优化。
   - 如果 residual 个数足够，用 `least_squares(method="lm")`；否则用 `trf`。
   - residual 是 `(predicted_positions - measured_positions).reshape(-1)`，单位米。

### `full_lm`

`pipeline.py` 中 `_empty_sensitivity()` 和 `_full_parameter_redundancy()` 构造占位结果，`batches = [range(54)]`。这会直接一次优化全部参数，不做可观测性筛选。真实数据只有位置观测、且 `payload=0` 时，结果可能严重病态，只能作为误差下降能力对比。

## 输出对象

`run_calibration_on_dataset()` 返回 `CalibrationResult`：

- `parameters`：54 个 `ErrorParameter`
- `redundancy`：`RedundancyResult`
- `sensitivity`：`SensitivityResult`
- `batches`：每阶段参数 index
- `identification`：最终向量和每阶段快照
- `nominal_metrics` / `calibrated_metrics`：位置误差统计，单位米
- `nominal_positions` / `calibrated_positions`
- `dataset`

`summarize_result()` 生成可 JSON 化 dict。CLI 默认用 `format_result_report()` 打印表格；带 `--json` 会直接输出 JSON。图表通过 `generate_evaluation_plots()` 写到 `outputs/figures`。

## 关键配置

`CalibrationConfig` 默认值：

```python
redundancy_tolerance = 1e-7
redundancy_max_combinations = 200_000
sensitivity_samples = 128
high_cumulative_score = 0.80
max_nfev_per_stage = 120
seed = 123
optimization_mode = "sobol_stepwise"
```

`config/nominal_config.py` 单位是米和弧度。`robot_model._validate_nominal_robot()` 会检查明显的长度单位错误；如果替换机器人参数，优先只改 `NOMINAL_ROBOT`，不要改算法模块。

## 已知工程约定和坑

- `outputs/`、`__pycache__/`、`*.pyc*` 已被忽略，不要把生成图、pkl、缓存当源码处理。
- `README.md` 和 `docs/*.md` 在当前环境可能显示乱码；需要信息时优先读源码和本文件。
- `rg` 在当前 Windows 环境曾出现“拒绝访问”，可用 PowerShell `Get-ChildItem -Recurse -File` 替代。
- 当前没有正式单元测试。改核心算法后至少跑一个小样本仿真命令和相关验证脚本。
- `full_lm` 不应替代默认方法，尤其不要把不可观测参数的数值解释成真实物理误差。
- 若实测数据 `payloads` 全为 0，柔顺参数必然不可观测。
- 若只用位置不含姿态，第 6 轴旋转、工具姿态等参数可能不可观测或强相关。
- `redundancy.independent_indices` 会受姿态样本、payload、directions、tolerance、Sobol seed/sample count 影响；不要假设某一组 index 永久固定。
- `make_paper_batches()` 接收的是 index，不是参数名；改参数顺序会影响所有旧输出。
- `visualization.py` 使用 matplotlib Agg 后端，适合无界面环境。

## 修改建议

- 改数据兼容性：优先改 `calibration/data_io.py`，保持输出 schema 不变。
- 改机器人：优先改 `config/nominal_config.py`；只有模型物理项变化时才改 `robot_model.py`。
- 改敏感度：替换 `sensitivity.py`，保持 `SensitivityResult` 字段。
- 改冗余筛选：替换 `redundancy.py`，保持 `RedundancyResult.independent_indices` 语义。
- 改优化策略：替换或扩展 `stepwise_lm.py`，保持 `IdentificationResult.vector/stages`。
- 改 CLI 输出：优先改 `reporting.py` 或 `visualization.py`，避免把展示逻辑塞回 `pipeline.py`。

## 建议验证路径

小改动后：

```powershell
python run_simulation.py --samples 20 --sensitivity-samples 8 --no-plots
```

涉及图表：

```powershell
python run_simulation.py --samples 20 --sensitivity-samples 8 --plots-dir outputs/figures --plot-prefix smoke
```

涉及真实数据：

```powershell
python run_real_world.py dataset\real_world200.pkl --sensitivity-samples 8 --no-plots
```

涉及 DH/MDH 约定：

```powershell
python scripts\verify_dh_to_mdh.py --samples 100
```

涉及 Sobol 实现：

```powershell
python verify_sensitivity.py
```
