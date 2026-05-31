# ACRNN Code, Docs, Figures, and Results Package

这是 ACRNN 项目的整理包，已经按“代码、说明文档、最终图、结果摘要”分类放好，适合直接发给队友、老师，或者直接作为 GitHub 仓库内容使用。

## 目录说明

- `code/`: ACRNN 相关代码、绘图脚本、运行脚本，以及模型代码目录 `ACRNN/`。
- `docs/`: 面向读者的项目说明、实验索引、运行说明和项目状态。
- `final_figures/`: 最终海报图和对应汇总文件。
- `results/`: 各实验任务的最终图和结果摘要，按任务分别分类。
- `DATA.md`: 数据集放置说明。DEAP 原始数据不随仓库分发。
- `requirements.txt`: 主要 Python 依赖。

## 适合直接上传的内容

1. `code/`
2. `docs/`
3. `final_figures/`
4. `results/`
5. `DATA.md`
6. `requirements.txt`

`results/strict_200ep_full/` 中保留的是严格复现配置：
`train_keep_prob=0.5`、`epochs=200`、32 个被试、valence + arousal。
旧的 `keep_prob=0.8` 试跑结果和 smoke test 结果没有放入该汇总。

如果你要把它并入现有 GitHub 仓库，可以把这个目录下的四个分类文件夹直接放到仓库根目录，或者按需要再拆分到更合适的位置。
