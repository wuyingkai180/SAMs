# SAMs 溶剂效应三组数据分析设计

## 目标

在不覆盖历史结果的前提下，沿用并改进现有 OPA 单分子溶剂化分析流程，对 `data/group1`、`data/group2` 和 `data/group3` 的新模拟结果进行可复现分析，并结合经核验的高引用文献解释溶剂对 SAM 形成前驱过程的可能影响。

本项目中的轨迹描述一分子 octadecylphosphonic acid（OPA）在周期性体相溶剂中的行为，没有固体基底或其他 OPA 分子。因此结论限定为预吸附输运、局部溶剂化和受力环境，不能直接声称测得 SAM 生长速率、覆盖度、取向、堆积密度或缺陷率。

## 输入与分组

- `data/group1`：acetone/n-heptane 混合物，acetone 体积分数为 1%、5%、10%、20%、30%、40% 和 50%。用于浓度—响应分析。
- `data/group2`：13 种纯溶剂，包括 acetone、acetonitrile、CPME、cyclohexane、DMC、ethanol、isopropanol、methanol、n-heptane、prol、p-xylene、THF 和 toluene。
- `data/group3`：组成比较集合。核心同基底比较为 acetone、isopropanol、THF 和 toluene 作为低浓度共溶剂、n-heptane 作为主体溶剂；THF/toluene 作为不同主体溶剂条件单列；纯 ethyl acetate 和 propylene carbonate 作为纯溶剂参照单列。
- `data/trash`：全部排除，不读取、不复制到有效结果。

当前完整性核验显示，27 个有效体系均包含 101 帧、0–10 ps 的 OPA motion/force CSV、完整 ASE trajectory、最终 XYZ，以及位移和受力基础图。

## 分析方案

### 1. 数据审计与标准化

建立机器可读 manifest，记录每个体系的逻辑名称、实际文件前缀、分组、组分、目标浓度、帧数、时间范围和文件完整性。名称不一致的 `acetone_5_n-heptane_95/acetone_n-heptane_*` 通过显式映射处理，不重命名原始文件。

核查时间轴、轨迹帧数、原子数和盒长的一致性。检查 OPA COM 是否受周期边界跳跃影响；若存在跳跃，采用 minimum-image/unwrapping 后的位置计算动力学指标，并同时保留原始 CSV 值供追溯。

### 2. 与历史结果可比的基础指标

对所有体系计算：

- OPA 最终位移、平均位移、最大位移与位移轨迹；
- OPA 总受力模的均值、标准差、最大值及力—位移相关；
- time-averaged MSD（TAMSD）、异常扩散指数 alpha、拟合 R²和描述性扩散代理；
- OPA phosphonate 头基与末端烷基碳周围的 RDF、4 Å 配位数和头/尾配位比。

历史的单时间原点 `D_eff` 和 KMeans 仅作为兼容性输出；核心解释优先使用多时间原点 TAMSD、直接可观测量和化学分组。

### 3. 针对混合溶剂的增强指标

对混合体系将 RDF 和配位数按溶剂 species 分解，分别计算共溶剂和主体溶剂在 OPA 头基、尾端附近的局部分布。使用局部组分相对于体相组分的富集因子判断优先溶剂化：富集因子大于 1 表示该 species 在指定局部区域富集，小于 1 表示贫化。

species 识别基于轨迹固定原子顺序、局部结构文件与体系原子计数交叉验证；验证不通过的体系不得生成 species-resolved 结论。

### 4. 三组比较逻辑

- Group 1：按 acetone 浓度排序，绘制各指标的浓度—响应曲线；同时报告 Pearson 和 Spearman 相关、线性趋势以及非单调性。由于每个浓度只有一条轨迹，不做经典组间显著性检验。
- Group 2：按 protic、aprotic polar 和 nonpolar 等化学类别展示，同时保留逐溶剂排名。类别标签只用于解释，不作为模拟产生的变量。
- Group 3：主要比较以 n-heptane 为主体的四种低浓度共溶剂；THF/toluene 单独展示；ethyl acetate 和 propylene carbonate 只作为纯溶剂参照。不同基底和纯溶剂点不混入同条件相关或排名。
- 跨组：检查同名体系是否为重复文件。完全相同的数据只计算一次，在各组视图中引用，不把复制文件作为独立重复。

### 5. 稳健性与不确定性

每条轨迹进行前后半程和连续时间块敏感性检查，报告关键指标随分析窗口的变化。对 alpha 报告拟合区间和 R²；短轨迹导致的精度限制必须进入最终报告。

相关性同时提供 Pearson 与 Spearman；小样本相关只作探索性描述。异常点采用 leave-one-out 敏感性检查，不因其影响趋势就自动删除。

### 6. 文献研究

检索主题包括 phosphonic-acid/alkanethiol SAM 的溶剂效应、吸附与自组装动力学、溶剂极性和氢键作用、MD 中的 RDF/配位数/优先溶剂化，以及短轨迹扩散分析。

优先使用同行评议论文和权威出版社页面，通过 DOI、PubMed、Crossref、OpenAlex 或期刊官网核验题录。所谓“高引用”需要记录可获得的引用计数来源和查询日期；引用数只用于筛选影响力，方法是否适用于当前单 OPA 体系需独立判断。

### 7. 输出

所有新产物写入 `results/new_solvent_analysis_2026-08-24/`：

- `manifest.csv`：输入审计和分组映射；
- `group1_concentration/`：浓度序列汇总、图表与报告；
- `group2_pure_solvents/`：纯溶剂比较汇总、图表与报告；
- `group3_composition/`：组成比较和分层参照结果；
- `combined/`：跨组汇总表、方法说明、文献矩阵和中文简报；
- `analysis_run.json`：脚本版本、参数、运行时间和环境信息。

图形使用 Python 现有技术栈生成，提供适合检查的 PNG，以及在可行时提供 PDF/SVG。最终中文简报区分“数据直接支持”“结合文献的解释”和“需要重复模拟验证”三类陈述。

## 错误处理与保护

- 原始 `data`、历史 `results` 和现有脚本均不被覆盖。
- 缺失文件、列名变化、帧数不匹配、species 映射失败或数值非有限时，该体系在对应分析中标为失败并记录原因，不静默跳过。
- 新分析写入独立目录；重复运行使用临时输出并在完整成功后替换目标文件，避免产生半成品报告。
- 工作区已有未提交改动；本任务只修改明确列入实施计划的分析文件和新增输出。

## 验证标准

- 27 个有效体系全部出现在 manifest，`trash` 中体系为零；
- 基础统计可从原始 CSV 独立复算并与输出一致；
- RDF、配位数和 species 富集计算通过合成小体系或已知计数测试；
- 同一原始数据的跨组引用不会被计作独立重复；
- 所有图表标签、浓度、单位和体系名称与 manifest 一致；
- 最终报告明确说明单轨迹、10 ps、101 帧和无基底的限制；
- 每篇用于方法或解释的论文均有可验证题录和直接链接。

