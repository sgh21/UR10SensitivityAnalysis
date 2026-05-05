# Multisource Error Calibration Baseline

这是 Lu 等 2025 多源误差敏感度标定方法的 baseline 项目。完整手册见 [docs/MANUAL.md](docs/MANUAL.md)。

两个主入口：

```powershell
python run_simulation.py --output outputs/synthetic_dataset.pkl
python run_real_world.py D:\path\to\real_world.pkl
```
