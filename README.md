# 美国国债收益率与 IG / HY OAS：状态切换实证

研究截止 **2026-11-03**。分析收益率变化与利差变化的同期关系，不是水平之间的因果检验。

## 结果

本次真实 FRED 快照覆盖 2023-11-06 至 2026-11-02；共同有效日度观测 746 条，完整周度变化 155 条。

| 指标 | IG | HY |
|---|---:|---:|
| 周度相关系数 | -0.178 | -0.207 |
| 固定 beta（OAS bp / 国债 bp） | -0.058 | -0.289 |
| beta 的 HAC 95% 区间 | [-0.126, 0.010] | [-0.650, 0.072] |

四个候选模型中，BIC 均偏好仅方差切换的模型。完整切换模型的部分状态持续期很短或有效样本很少，不能据此认定稳定的宏观 regime。没有模拟或补造长历史结果。

## 文件

- [完整中文报告](report.html)：下载后用浏览器打开，四张图表已内嵌，无需额外图片文件。
- [Jupyter Notebook](regime_analysis.ipynb)：完整实现，公开版清空运行输出；结果见报告。
- [分析脚本](analysis.py)：可独立运行。
- [统计结果](results/)：模型比较、状态统计、留出检验、优化日志、快照哈希及验证记录。

## 运行

建议 Python 3.12，在虚拟环境中安装依赖：

```sh
python -m pip install -r requirements.txt
python analysis.py --data data --out rerun --download
```

或打开 Notebook，顺序运行全部单元。若 data/ 尚不存在，Notebook 会尝试联网下载。若已有完整的自有数据快照：

```sh
python analysis.py --data data --out rerun
```

三个输入文件为 `data/DGS10.csv`、`data/BAMLC0A0CM.csv`、`data/BAMLH0A0HYM2.csv`。第一列为日期，第二列名称为对应 series ID，输入数值单位为百分数而非 bp，缺失值留空或写 `.`。

## 数据边界

FRED 从 2026 年 4 月起只保留这两条 ICE OAS 的滚动三年历史，因此无法以本样本检验 2008、2020 或 2022。后续下载的窗口与本次冻结快照可能不同，即使观察截止日期相同也未必能精确复现报告。严格复现需要对应快照，哈希见 results/manifest.json。

这个公开仓库不分发原始 ICE 指数数据或含原始数据的压缩包。数据可按来源条款自行获取；本地完整交付包包含研究时使用的快照。报告中的汇总结果仍保留。

## 方法与验证

先对齐共同有效交易日，再取每周最后共同观测，差分乘 100 转为 bp；不前向填充。主回归以 DGS10 变化解释 IG/HY OAS 变化，分别比较 OLS、仅方差切换、截距/方差切换及完整两状态 Markov 回归。提供 26/52 周滚动 beta 和 HAC 区间。

按时间前 70% 训练、后 30% 留出；测试时参数冻结，逐期更新过滤概率。测试使用当期已实现的国债变化，是条件预测评估而不是提前一周交易回测。平滑状态相关性仅为事后描述。

原分析全部代码单元已在 IPython 进程内顺序执行，数据对齐、bp 换算、AIC/BIC、8 组留出结果、概率范围及图表输出均通过检查。本机原生 Jupyter 内核因预装包兼容问题无法启动，未声称通过原生内核验证。

## 来源

- [FRED DGS10](https://fred.stlouisfed.org/series/DGS10)
- [FRED IG OAS](https://fred.stlouisfed.org/series/BAMLC0A0CM)
- [FRED HY OAS](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)
- [Duffee (1998)](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00089)
- [Collin-Dufresne, Goldstein & Martin (2001)](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00402/abstract)
- [statsmodels MarkovRegression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html)
