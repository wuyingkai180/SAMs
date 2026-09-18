"""Align bilingual results and captions with Figure 2(a-c) and Figure 3(a-e)."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ZH=r'''## 结果与讨论

### 纯溶剂环境中的 OPA 短时运动（Figure 2）

14 种纯溶剂中的质心位移曲线揭示了 OPA 在 10 ps 内不同的偏离幅度和时间演化特征（Figure 2a）。正庚烷中的最大位移为 7.127 Å，出现在 8.8 ps；至 10 ps 时仍保持 6.063 Å，为纯溶剂中的最高最终位移。DMC、CPME 和甲苯的最终位移依次为 5.044、4.894 和 4.828 Å，亦表现出较大的初始位置偏离。相比之下，异丙醇中的位移在 1.7 ps 达到 2.590 Å 后回落，最终仅为 0.647 Å。DMC 和 CPME 则分别在 9.3 和 9.7 ps 达到最大位移，说明其最大偏离出现在观察时段后期。图例按最终位移由高到低排列，对应曲线在 10 ps 处的纵向位置。

这些曲线表示 OPA 质心相对于固定初始位置的距离，因而曲线下降意味着分子重新接近初始位置，并不表示累计运动路程减小。纯丙酮中的位移在 4.2 ps 达到 4.351 Å，最终降至 3.015 Å；这与正庚烷中较大的最终偏离形成区别。Figure 2b 展示了纯正庚烷中 OPA 及周围溶剂的模拟快照，为动力学比较提供结构背景。单帧形貌不能定量确定溶剂化强度、分子间相互作用寿命或吸附能力。

TAMSD 参数进一步区分了运动幅度与时间标度（Figure 2c）。为比较不同指数的体系，采用共同参考滞后时间 1 ps 下的拟合 TAMSD，即 $A_{1\mathrm{ps}}=K_\alpha(1\,\mathrm{ps})^\alpha$。正庚烷和丙酮的 $A_{1\mathrm{ps}}$ 分别为 1.722 和 1.694 Å²，位居纯溶剂中的第一和第二，其指数分别为 0.906 和 0.876。两者因此兼具较大的参考时间运动幅度和略低于线性的 TAMSD 增长，但 Figure 2a 中明显不同的最终位移说明，多个时间起点平均得到的 TAMSD 并不等同于相对于单一初始位置的偏离。

CPME 和 DMC 的指数分别为 1.105 和 1.031，是纯溶剂中仅有的两个高于 1 的体系，而其拟合幅度分别为 1.573 和 1.310 Å²，低于正庚烷和丙酮。甲苯、乙醇和乙腈与正庚烷、丙酮同处于 $0.8\leq\alpha<1.0$ 区间；环己烷、甲醇和 THF 位于 $0.6\leq\alpha<0.8$ 区间。乙酸乙酯、异丙醇、碳酸丙烯酯和对二甲苯的指数均低于 0.6，其中碳酸丙烯酯的拟合幅度最小，为 0.490 Å²。不同参数给出的排序并不完全一致：较大的指数不必对应较大的拟合幅度或最终位移。图中的水平分区用于比较有效指数，不是经过验证的扩散类型分类阈值。

### 混合溶剂组成对 OPA 运动的影响（Figure 3）

Figure 3a 展示了 5% 丙酮/95% 正庚烷体系中 OPA 周围的混合溶剂环境。与 Figure 2b 的纯正庚烷模型相对照，该快照直观呈现了组成变化，但不能单凭局部排列判断丙酮是否优先富集于 OPA 周围。Figure 3b 比较了这一混合物、纯正庚烷和纯丙酮中的完整三维质心轨迹。各轨迹以自身初始位置为原点，黑色圆点与彩色方形分别标记起点和终点。三图的坐标范围分别为 ±4.5、±7.5 和 ±4 Å，每个小图内部的三个方向等比例，因此跨图比较运动幅度时应依据刻度，而非路径在画面中的视觉大小。

丙酮浓度系列表明，组成变化引起非单调的运动响应（Figure 3c）。七种浓度中，5% 丙酮体系的拟合幅度最大，$A_{1\mathrm{ps}}=1.984$ Å²，其指数为 0.861。1% 体系的指数与其接近，为 0.860，但拟合幅度较低，为 1.580 Å²。丙酮比例增加至 10% 和 20% 后，拟合幅度依次降至 1.650 和 1.366 Å²，而指数分别升至 0.890 和 0.966。因此，20% 体系的 TAMSD 增长最接近线性，但并不具有最大的参考时间运动幅度。进一步增加至 30% 时，指数降至 0.450；在 40% 和 50% 时分别回升至 0.744 和 0.807。相应拟合幅度为 1.127、1.031 和 1.417 Å²，表明增加丙酮含量并未持续提高 OPA 的短时运动幅度。

完整位移曲线提供了与拟合参数互补的证据（Figure 3d）。该面板包含浓度系列和等比例组分系列中的 11 个不同混合体系，共享的 5% 丙酮/95% 正庚烷轨迹仅绘制一次。5% 丙酮体系在 8.8 ps 达到 5.362 Å 的最大位移，10 ps 时仍保持 4.357 Å，两者均为这 11 个混合体系中的最高值。10% 丙酮体系的最终位移为 3.794 Å，居第二位。20% 丙酮、5% 甲苯和 5% 异丙醇在正庚烷中的最终位移分别为 3.071、3.065 和 3.037 Å，数值接近，单条轨迹不足以确定三者内在移动性的可靠排序。5% THF/95% 正庚烷则在 2.3 ps 达到 2.173 Å 后回落至 1.084 Å，呈现较早偏离、随后重新接近初始位置的过程。

在名义体积比均为 5:95 的五种混合物中，组分改变同时影响拟合幅度和指数（Figure 3e）。5% 丙酮/95% 正庚烷的拟合幅度最高；5% 异丙醇/95% 正庚烷次之，其 $A_{1\mathrm{ps}}=1.578$ Å²、$\alpha=0.821$。5% 甲苯/95% 正庚烷具有本组最高的指数 0.919，但拟合幅度仅为 1.136 Å²。5% THF/95% 正庚烷的指数最低，为 0.516，拟合幅度为 0.957 Å²。将其主要溶剂由正庚烷换为甲苯后，5% THF/95% 甲苯的指数升至 0.697，拟合幅度却降至 0.879 Å²。这些差异说明，共溶剂和主要溶剂的选择可对运动幅度与时间依赖性产生不同影响。

结合 Figures 2 和 3，在正庚烷中加入 5% 丙酮后，1 ps 下的拟合 TAMSD 从 1.722 增至 1.984 Å²，但最大位移从 7.127 降至 5.362 Å，最终位移从 6.063 降至 4.357 Å。因此，该混合物的优势体现在参考滞后时间下的拟合幅度，以及所考察混合体系内部的最大和最终偏离，而非所有运动指标均超过纯正庚烷。这一结果与实验中正庚烷、丙酮及 5% 丙酮混合物表现较好的趋势相容，为其提供了短时分子平移层面的解释线索。

这些结论限定于每个体系一条 10 ps 轨迹及 0.2–3.3 ps 的拟合窗口。TAMSD 参数可能依赖有限采样和拟合区间 [1]；指数区间及较大的 $K_\alpha$ 均不能单独证明稳定的异常扩散或更高的宏观扩散系数。将这些短时运动差异进一步解释为 SAM 成膜能力的变化，仍需独立轨迹、更长采样，以及包含显式基底和多个 OPA 分子的模拟，并通过吸附、覆盖率、取向或缺陷等指标验证。

## 图注

**Figure 2 | 纯溶剂环境中的 OPA 短时运动。** **a**，14 种纯溶剂中 OPA 质心相对于初始位置的距离随模拟时间的变化，图例按 10 ps 最终位移由高到低排列。**b**，纯正庚烷中 OPA 及周围溶剂的模拟快照。**c**，纯溶剂的 TAMSD 前因子 $K_\alpha$ 与有效指数 $\alpha$，由 0.2–3.3 ps 内的幂律拟合得到。水平虚线位于 $\alpha=0.6$、0.8 和 1.0；浅蓝和浅橙底色分别标示 0.6–0.8 和 0.8–1.0 区间，仅用于视觉分组。横向箭头表示拟合幅度增加，不表示已测得的宏观扩散系数增大。各体系对应一条含 101 个采样点的轨迹；未展示独立重复的不确定性。

**Figure 3 | 混合溶剂组成对 OPA 短时运动的影响。** **a**，5% 丙酮/95% 正庚烷中 OPA 周围溶剂的模拟快照。**b**，5% 丙酮/95% 正庚烷、纯正庚烷及纯丙酮中的三维质心轨迹；各轨迹相对于自身初始位置绘制，黑色圆点为 0 ps 起点，彩色方形为 10 ps 终点。三个小图的坐标范围分别为 ±4.5、±7.5 和 ±4 Å，每图内部 XYZ 方向等比例。**c**，名义丙酮体积分数为 1%、5%、10%、20%、30%、40% 和 50% 的丙酮/正庚烷体系的 TAMSD 参数。**d**，11 个不同混合体系的位移曲线及按最终位移降序排列的图例。**e**，五种名义体积比为 5:95 的混合物的 TAMSD 参数。**c,e** 的拟合窗口、分区底色和水平虚线与 Figure 2c 一致，横向箭头表示拟合幅度增加。5% 丙酮体系在 **c,e** 中共享同一组数据，并在 **d** 中仅计一次。百分数表示名义体积分数。$K_\alpha$ 的单位为 Å² ps$^{-\alpha}$；跨体系幅度比较使用共同参考滞后时间 1 ps 下的拟合 TAMSD。

'''

EN=r'''## Results and discussion

### Short-time OPA motion in pure solvents (Figure 2)

The centre-of-mass displacement traces in 14 pure solvents reveal differences in the magnitude and temporal evolution of OPA excursions over 10 ps (Figure 2a). In n-heptane, the displacement reached 7.127 Å at 8.8 ps and remained at 6.063 Å at 10 ps, the largest final displacement among the pure solvents. DMC, CPME and toluene also retained substantial displacements from the initial position, ending at 5.044, 4.894 and 4.828 Å, respectively. By contrast, isopropanol reached an early maximum of 2.590 Å at 1.7 ps before returning to 0.647 Å at 10 ps. DMC and CPME reached their maxima at 9.3 and 9.7 ps, respectively, placing their largest excursions near the end of the observation period. The legend is ordered by decreasing final displacement and therefore follows the vertical order of the curve endpoints.

These traces measure distance from a fixed initial position: a decrease indicates that OPA moved closer to that position, rather than a reduction in cumulative distance travelled. In acetone, the displacement reached 4.351 Å at 4.2 ps and decreased to 3.015 Å at 10 ps, contrasting with the larger final displacement in n-heptane. The simulation snapshot in Figure 2b shows OPA surrounded by pure n-heptane and provides structural context for the dynamical comparison. A single configuration does not quantify solvation strength, interaction lifetimes or adsorption propensity.

The TAMSD parameters further distinguish displacement amplitude from temporal scaling (Figure 2c). To compare systems with different exponents, amplitudes are evaluated as the fitted TAMSD at a common reference lag of 1 ps, $A_{1\mathrm{ps}}=K_\alpha(1\,\mathrm{ps})^\alpha$. The values for n-heptane and acetone were 1.722 and 1.694 Å², ranking first and second among the pure solvents, with exponents of 0.906 and 0.876, respectively. Both solvents thus combined relatively large reference-lag amplitudes with slightly sublinear TAMSD growth. Their different final displacements in Figure 2a nevertheless show that TAMSD, which averages over multiple time origins, is distinct from displacement relative to a single starting position.

CPME and DMC were the only pure solvents with exponents above unity, at 1.105 and 1.031, whereas their fitted amplitudes of 1.573 and 1.310 Å² were lower than those of n-heptane and acetone. Toluene, ethanol and acetonitrile occupied the $0.8\leq\alpha<1.0$ interval together with n-heptane and acetone; cyclohexane, methanol and THF fell within $0.6\leq\alpha<0.8$. Ethyl acetate, isopropanol, propylene carbonate and p-xylene had exponents below 0.6, with propylene carbonate exhibiting the smallest fitted amplitude, 0.490 Å². A larger exponent therefore did not necessarily correspond to a larger amplitude or final displacement. The horizontal bands provide a visual comparison of effective exponents and are not validated thresholds for classifying diffusion mechanisms.

### Composition-dependent OPA motion in mixed solvents (Figure 3)

Figure 3a shows the solvent environment around OPA in 5% acetone/95% n-heptane. Comparison with the pure n-heptane model in Figure 2b illustrates the change in composition, but the local arrangement in a single snapshot does not establish preferential enrichment of acetone around OPA. Figure 3b compares the complete three-dimensional centre-of-mass trajectories in this mixture, pure n-heptane and pure acetone. Each trajectory is referenced to its own initial position, with black circles and coloured squares marking the start and end. The respective axis limits are ±4.5, ±7.5 and ±4 Å, with equal XYZ scales within each panel; comparisons of displacement magnitude must therefore use the axis ticks rather than the apparent size of the paths.

The acetone concentration series exhibited a non-monotonic dynamical response (Figure 3c). Among the seven compositions, the 5% mixture had the largest fitted amplitude, $A_{1\mathrm{ps}}=1.984$ Å², with $\alpha=0.861$. The 1% mixture had a similar exponent, 0.860, but a smaller amplitude of 1.580 Å². Increasing the acetone fraction to 10% and 20% reduced the amplitude to 1.650 and 1.366 Å² while raising the exponent to 0.890 and 0.966, respectively. The 20% mixture thus showed the most nearly linear TAMSD growth, rather than the largest reference-lag amplitude. At 30% acetone, the exponent decreased to 0.450 before recovering to 0.744 and 0.807 at 40% and 50%. The corresponding amplitudes were 1.127, 1.031 and 1.417 Å². Increasing acetone content consequently did not produce a sustained increase in short-time displacement amplitude.

The complete displacement traces provide complementary evidence (Figure 3d). This panel contains 11 distinct mixtures from the concentration and fixed-ratio composition series, with the shared 5% acetone/95% n-heptane trajectory plotted once. The 5% acetone mixture reached a maximum displacement of 5.362 Å at 8.8 ps and retained 4.357 Å at 10 ps, the highest maximum and final displacements among these mixtures. The 10% acetone mixture ranked second in final displacement, at 3.794 Å. The final values for 20% acetone, 5% toluene and 5% isopropanol in n-heptane were close, at 3.071, 3.065 and 3.037 Å, respectively; single trajectories do not resolve a reliable ranking of their underlying mobilities. In contrast, 5% THF/95% n-heptane reached 2.173 Å at 2.3 ps and ended at 1.084 Å, indicating an early excursion followed by a return towards the starting position.

Among the five mixtures at a nominal volume ratio of 5:95, changing the components affected both fitted amplitude and exponent (Figure 3e). The largest amplitude occurred in 5% acetone/95% n-heptane, followed by 5% isopropanol/95% n-heptane, with $A_{1\mathrm{ps}}=1.578$ Å² and $\alpha=0.821$. The 5% toluene/95% n-heptane mixture had the highest exponent, 0.919, but a smaller amplitude of 1.136 Å². The lowest exponent, 0.516, occurred in 5% THF/95% n-heptane, whose amplitude was 0.957 Å². Replacing n-heptane with toluene as the major solvent raised the exponent of the 5% THF mixture to 0.697 but reduced its amplitude to 0.879 Å². Cosolvent and major-solvent identity therefore influenced displacement amplitude and temporal scaling in different ways.

Together, Figures 2 and 3 show that adding 5% acetone to n-heptane increased the fitted TAMSD at 1 ps from 1.722 to 1.984 Å², while reducing the maximum displacement from 7.127 to 5.362 Å and the final displacement from 6.063 to 4.357 Å. The mixture therefore stands out in its reference-lag amplitude and, within the tested mixtures, its maximum and final excursions; it does not exceed pure n-heptane in every measure of motion. These observations are compatible with the favourable experimental performance of n-heptane, acetone and the 5% acetone mixture, providing a short-time translational perspective on those outcomes.

The conclusions are restricted to one 10 ps trajectory per system and the 0.2–3.3 ps fitting window. Finite sampling and the fitting interval can influence TAMSD parameter estimates [1]; neither an exponent interval nor a larger $K_\alpha$ alone establishes persistent anomalous diffusion or a higher macroscopic diffusion coefficient. Linking these short-time differences to SAM formation requires independent trajectories, longer sampling and simulations containing an explicit substrate and multiple OPA molecules, assessed through adsorption, coverage, orientation or defect formation.

## Figure legends

**Figure 2 | Short-time OPA motion in pure solvents.** **a**, Distance of the OPA centre of mass from its initial position in 14 pure solvents over 10 ps. The legend is ordered by decreasing final displacement. **b**, Simulation snapshot of OPA and surrounding n-heptane molecules. **c**, TAMSD prefactor, $K_\alpha$, versus effective exponent, $\alpha$, obtained from power-law fits over 0.2–3.3 ps. Horizontal dashed lines mark $\alpha=0.6$, 0.8 and 1.0; pale blue and orange bands indicate the 0.6–0.8 and 0.8–1.0 intervals for visual grouping only. The horizontal arrow denotes increasing fitted amplitude, rather than an established increase in macroscopic diffusivity. Each system is represented by one trajectory containing 101 sampled points; uncertainty from independent replicates is not shown.

**Figure 3 | Mixed-solvent composition modulates short-time OPA motion.** **a**, Simulation snapshot of OPA in 5% acetone/95% n-heptane. **b**, Three-dimensional centre-of-mass trajectories in 5% acetone/95% n-heptane, pure n-heptane and pure acetone, each referenced to its own starting position. Black circles mark the start at 0 ps and coloured squares mark the end at 10 ps. Axis limits are ±4.5, ±7.5 and ±4 Å, respectively, with equal XYZ scales within each panel. **c**, TAMSD parameters for acetone/n-heptane mixtures containing nominal acetone volume fractions of 1%, 5%, 10%, 20%, 30%, 40% and 50%. **d**, Displacement traces for 11 distinct mixtures, with the legend ordered by decreasing final displacement. **e**, TAMSD parameters for five mixtures at a nominal volume ratio of 5:95. The fitting window, shaded bands and dashed lines in **c,e** follow Figure 2c; horizontal arrows denote increasing fitted amplitude. The 5% acetone mixture shares the same data in **c,e** and is counted once in **d**. Percentages denote nominal volume fractions. $K_\alpha$ has units of Å² ps$^{-\alpha}$; amplitudes are compared across systems using fitted TAMSD at a common reference lag of 1 ps.

'''

for lang,section,ref,new in [('zh','## 结果与讨论','## 参考文献',ZH),
                             ('en','## Results and discussion','## Reference',EN)]:
    p=ROOT/f'manuscript_analysis_{lang}.md'
    text=p.read_text(encoding='utf-8')
    start=text.index(section)
    end=text.index(ref,start)
    prefix=text[:start].replace('Figure X','Figures 2 and 3' if lang=='en' else 'Figures 2 和 3')
    updated=prefix+new+text[end:]
    assert 'Figure X' not in updated
    assert updated.count('$$')%2==0
    p.write_text(updated,encoding='utf-8')
    print('Updated:',p.name)
