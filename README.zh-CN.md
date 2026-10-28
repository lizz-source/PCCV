# 代码与计算复现说明

本仓库对应论文《Efficient and Verifiable Carrier-Level Shapley Attribution for Pathway Sensitivity in Microbial Composition Models》。英文入口和完整运行命令见[README](README.md)。

## 最短验证流程

使用Python 3.12，按`requirements-cpu.txt`安装CPU依赖。随后依次运行：

```text
python scripts/check_release.py
python -m unittest discover -s tests -v
python run_pipeline.py --mode verify
```

验证模式加载7个真实预测器和40个受控预测器，复算测试指标与3个完整真实归因案例，并检查精确分组的概率和梯度。它不重新训练真实模型。执行输出进入新的`runs/verify_...`目录，原始参考文件保持不变；最后生成`run_summary.json`。单独的同种子重复训练与独立训练种子验证不是一回事。

`--mode reproduce`重跑完整分析，并包含受控预测器拟合；`--mode retrain`还从零重训真实预测器和同种子重复模型。这些模式耗时更长。程序不会据当前机器计时保证其他设备的时长或位级结果。

## 图件与案例

`publication/figures/`含九张正文图和一张补充图，均提供600 dpi PNG及可编辑SVG。`publication/figure_number_map.json`对应论文图号。绘图需要Times New Roman；运行`python publication/publication_figures.py`后，再运行`python publication/verify_display_values.py`。

案例CSV保留CM.29_WGS、ko02020的全部78个载体，包括9个负贡献。案例来自冻结计划的首个样本和首条通路，图中保留置换采样参照。不要将它解释为PCCV在真实数据上的新增胜出证据或已验证的微生物因果机制。

图4B的统计汇总已纠正：字面分类`null`必须保留为分类，排除十个全零机制后，在40个非全零机制中计算空玩家质量比例。比例分配的中位数为0.5381，旧图约0.3793不是该40机制汇总。底层实验记录没有改变。

## 文件与公开上传

上传本目录中清单列出的源代码、数据、模型、结果、文档及隐藏的`.github`、`.gitignore`和`.gitattributes`。不要上传`.venv/`、`vendor/`、`runs/`、日志和缓存。仓库版本、作者与软件引用信息在`CITATION.cff`中；尚未创建的GitHub地址和论文DOI未被编造。

软件采用MIT；第三方数据和依赖保留原条款，详见`NOTICE.md`及`DATA_PROVENANCE.md`。本文从冻结矩阵开始复现，历史原始测序到注释的完整处理链尚未恢复。默认GitHub验证工作流需上传后由仓库所有者手动启动，不能把本地验证报告当作已通过的远程CI。

`.gitattributes`关闭行尾转换，以保持数据与清单的字节哈希。确需修改文件时，记录变更并运行`python scripts/build_manifest.py`生成新清单。
