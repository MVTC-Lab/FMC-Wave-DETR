# FMC-Wave-DETR 核心机制消融

七份 YAML 均完整保留原模型的 backbone、neck、decoder、1000 queries 和 loss 配置，可直接替换原训练入口的模型配置路径。

| 配置 | 唯一变化 |
|---|---|
| FMC-core-full.yaml | Full 参照 |
| FMC-core-spatial-mask.yaml | `mask_source: spatial` |
| FMC-core-global-topk.yaml | `query_mode: global_topk` |
| FMC-core-frequency-global-topk.yaml | `query_mode: frequency_global_topk` |
| FMC-core-fixed-quota.yaml | `query_mode: fixed_quota`，500/300/200 |
| FMC-core-no-calibration.yaml | `semantic_mode: none` |
| FMC-core-acsa-no-frequency.yaml | `acsa_freq_enabled: false` |

统一配置位于 `freq_guidance` 下：

```yaml
freq_guidance:
  mask_source: wavelet
  semantic_mode: residual_logit
  query_mode: scene_conditioned_dynamic
  fixed_quotas: [500, 300, 200]
  acsa_freq_enabled: true
```

新字段 `semantic_mode` / `query_mode` 优先于旧字段 `semantic_calibration.mode` / `query.mode`。提供的七份 YAML 已使两种写法一致。原 Full YAML 无需修改；缺少新增字段时仍按原 Full 配置运行。显式关闭旧 `enabled` 开关会进入历史兼容路径，不属于这七组核心消融；不要用它模拟 No Calibration 或 ACSA w/o Frequency。

Spatial 使用 HSG 中 DWT 前的 `x_c`，经无参数 PixelUnshuffle 转换为与 `enhanced_full` 相同的 `[B,4C,H/2,W/2]`，共用 `mask_generator`。Haar、refinement、重建、coordinate attention 和残差门保持原样。

Global 仅使用分类置信度。Frequency Global 使用 Full 的频率 boost 和局部峰值抑制后全局选择。Fixed 使用相同 priority，每层选择固定数量；不足部分按未选有效候选的全局 priority 补齐。总有效候选不足 1000 时明确报错，避免无效或重复 Query。Frequency Global 和 Fixed 缺少 mask 时也明确报错，避免实验静默退化。Full 原有动态算法、课程和 fallback 保留。

`semantic_mode: none` 只跳过跨层校准，保留 alpha 参数；raw mask 继续进入 ACSA、selector 和 mask loss。`acsa_freq_enabled: false` 只关闭两条频率调制路径，保留真实 DCNv2、offset/modulation 预测和全部融合结构。

初始化会输出实际配置。训练 CSV 增加 `sc_fixed_quota_fill_count`、`sc_global_fallback_count`、`sc_fixed_fallback_count`、`sc_capacity_relax_count`。计数表示当前 batch 的补齐 Query 数或发生对应事件的图像数，不是整个 epoch 的累计计数；原训练框架保持原来的聚合方式。其余 selected、mask 均值、alpha、evidence、capacity 指标继续使用现有框架。字符串模式保存在 `head.last_query_mode`，避免混入数值 loss 日志。

验证如需 Coverage@K，设置：

```yaml
freq_guidance:
  diagnostics:
    enabled: true
    store_last_batch: true
```

读取 `head.last_query_debug['selected_indices']` 和 `['selected_priority']`。只在验证副本中对 `selected_priority` 排序，然后取前 K 个对应索引；不要重排模型传入 decoder 的索引。默认不保存整批 CPU 调试数据。

复现检查（在工程根目录）：

```bash
.venv-ablation/bin/python -m pytest -q tests/test_core_ablation.py
.venv-ablation/bin/python tools/validate_core_ablation.py --checkpoint ultralytics/data/SODA-A-Official_FMC-L_800-SBFG-B40-FP32/weights/best.pt --imgsz 800
.venv-ablation/bin/python tools/smoke_core_ablation_training.py
```

比较脚本使用 `artifacts/core_ablation/baseline/` 下保存的修改前源码。`--checkpoint` 可换成真正的 Full SBFGv2 权重，`--image` 可指定验证图片，`--device cuda` 可在训练服务器复验。未提供 checkpoint 时使用固定种子初始化。当前验证用的旧权重只缺少两个 alpha，均统一初始化为零；其他缺失/多余参数会报错。

环境仅新增独立 `.venv-ablation`，没有修改原 `.venv`。完整测试依赖版本在 `artifacts/core_ablation/test_environment.txt`；`pytorch-wavelets` 依赖 `pkg_resources`，该测试环境使用 `setuptools<81`。

训练协议目标仍为 AI-TOD-R、800×800、40 epochs、4×A40、global batch=16、AdamW、lr=1e-4、weight decay=1e-5、Step LR 24/32/38、warmup 500 updates、clip=35、FP32、HFlip=0.5、不加载预训练权重。**这些模型 YAML 不负责实现训练协议。** 当前提供的通用 trainer 使用 clip=10，未找到相应专用训练入口；应通过已有的 AI-TOD-R 训练入口统一加载这七份模型配置。此次未修改 trainer 的优化器、scheduler、warmup、裁剪或预处理，也未执行正式训练。
