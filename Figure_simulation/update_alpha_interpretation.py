"""Update Figure 2 interpretation against the supplied Kepten paper."""
from pathlib import Path
root=Path(__file__).resolve().parent
zh=r'''Figure 2c 的纵轴为 TAMSD 有效标度指数 $\alpha$，表征均方位移随滞后时间增长的规律，而非 TAMSD 本身的大小。按照 Kepten 等人采用的幂律分析框架，$\alpha=1$ 对应正常扩散的线性 MSD 标度，$\alpha<1$ 和 $\alpha>1$ 分别对应亚线性和超线性增长 [1]。该文以 $\alpha=0.3$、0.7、1.3 和 1.7 分别作为强亚扩散、弱亚扩散、弱超扩散和强超扩散的代表情形，但未将 0.6 和 0.8 定义为通用分类阈值。因此，本图在这一物理框架下进一步划分 $\alpha<0.6$、$0.6\leq\alpha<0.8$、$0.8\leq\alpha<1.0$ 和 $\alpha\geq1.0$ 四个展示区间：前三者依次表示由较明显的亚线性增长向接近正常扩散线性标度的变化；最后一区间包含线性边界及超线性增长。这里的指数描述所选拟合窗口内的有效标度，不据此单独确定长期输运机制。

在这一分区中，乙酸乙酯、异丙醇、碳酸丙烯酯和对二甲苯位于 $\alpha<0.6$ 区域；环己烷、甲醇和 THF 位于浅蓝色的 $0.6\leq\alpha<0.8$ 区域。正庚烷、丙酮、甲苯、乙醇和乙腈位于浅橙色的 $0.8\leq\alpha<1.0$ 区域，其 TAMSD 增长较接近线性。CPME 和 DMC 的指数分别为 1.105 和 1.031，在当前窗口内呈超线性增长。由此，纵向比较反映时间标度的差异，并不意味着指数越大，溶剂对 OPA 运动或成膜的促进效果就必然越好。

横轴 $K_\alpha$ 则提供运动幅度信息。对于相同的 $\alpha$，增大 $K_\alpha$ 会使拟合 TAMSD 曲线在各滞后时间整体升高；对于指数不同的体系，采用共同参考滞后时间 1 ps 下的拟合 TAMSD，$A_{1\mathrm{ps}}=K_\alpha(1\,\mathrm{ps})^\alpha$，进行量纲一致的比较。以 ps 为时间单位时，$A_{1\mathrm{ps}}$ 与所列 $K_\alpha$ 的数值相同，因此图中向右移动表示参考滞后时间下更大的均方位移。碳酸丙烯酯的 $A_{1\mathrm{ps}}$ 最小，为 0.490 Å²；正庚烷和丙酮分别达到 1.722 和 1.694 Å²，位居纯溶剂中的第一和第二。CPME 和 DMC 虽具有更高的指数，其拟合幅度分别为 1.573 和 1.310 Å²，均低于正庚烷和丙酮。

综合两条坐标轴，正庚烷和丙酮同时处于接近线性标度的 $0.8\leq\alpha<1.0$ 区间（$\alpha$ 分别为 0.906 和 0.876），并具有本组最大的两个 TAMSD 前因子。这种“较大的运动幅度与接近线性的时间增长相结合”的特征，使两者在本研究的纯溶剂比较中表现出较有利的 OPA 短时平移动力学，与实验中两种溶剂表现较好的趋势一致。其中，正庚烷还具有最高的最大位移和最终位移，而丙酮的优势主要体现在较大的参考时间拟合幅度及接近线性的标度。Figure 2 因而从分子运动层面支持两种溶剂的有利表现，但该指数区间本身并非 SAM 成膜效果的普适最优区间。

'''
en=r'''The vertical coordinate in Figure 2c is the effective TAMSD scaling exponent, $\alpha$, which describes growth with lag time rather than the magnitude of TAMSD itself. In the power-law framework used by Kepten et al., $\alpha=1$ corresponds to the linear MSD scaling of normal diffusion, whereas $\alpha<1$ and $\alpha>1$ describe sublinear and superlinear growth, respectively [1]. That study used $\alpha=0.3$, 0.7, 1.3 and 1.7 as representative cases of strong subdiffusion, weak subdiffusion, weak superdiffusion and strong superdiffusion; it did not establish 0.6 and 0.8 as universal classification thresholds. Within this physical framework, the present figure uses four display intervals: $\alpha<0.6$, $0.6\leq\alpha<0.8$, $0.8\leq\alpha<1.0$ and $\alpha\geq1.0$. The first three describe progressively less sublinear growth approaching the linear scaling of normal diffusion, while the last includes the linear boundary and superlinear growth. These fitted exponents characterize the selected lag-time window rather than independently establishing long-time transport mechanisms.

Ethyl acetate, isopropanol, propylene carbonate and p-xylene occupied the $\alpha<0.6$ region, while cyclohexane, methanol and THF fell within the pale blue $0.6\leq\alpha<0.8$ band. The pale orange $0.8\leq\alpha<1.0$ band contained n-heptane, acetone, toluene, ethanol and acetonitrile, whose TAMSD growth was closer to linear. CPME and DMC had exponents of 1.105 and 1.031, respectively, indicating superlinear growth within the fitted window. Vertical comparisons therefore distinguish temporal scaling; a higher exponent does not necessarily indicate better performance in promoting OPA motion or film formation.

The horizontal coordinate, $K_\alpha$, provides complementary information on displacement amplitude. At a fixed $\alpha$, increasing $K_\alpha$ raises the fitted TAMSD at every lag. When exponents differ, amplitudes are compared at a common reference lag of 1 ps using $A_{1\mathrm{ps}}=K_\alpha(1\,\mathrm{ps})^\alpha$, which has consistent units across systems. With time expressed in ps, $A_{1\mathrm{ps}}$ is numerically equal to the reported $K_\alpha$; movement to the right thus indicates a larger fitted mean-squared displacement at the reference lag. Propylene carbonate had the smallest amplitude, 0.490 Å², whereas n-heptane and acetone ranked first and second at 1.722 and 1.694 Å². Although CPME and DMC had higher exponents, their amplitudes of 1.573 and 1.310 Å² were lower than those of n-heptane and acetone.

Considering both coordinates, n-heptane and acetone combined exponents in the near-linear $0.8\leq\alpha<1.0$ interval, at 0.906 and 0.876, with the two largest TAMSD prefactors among the pure solvents. This combination of large displacement amplitude and nearly linear temporal growth identifies favourable short-time OPA translational dynamics in both solvents, consistent with their favourable experimental performance. N-heptane additionally exhibited the largest maximum and final displacements, whereas the strengths of acetone were most evident in its large reference-lag amplitude and near-linear scaling. Figure 2 thus provides a molecular-motion basis consistent with the favourable performance of both solvents, without establishing this exponent interval as a universal optimum for SAM formation.

'''
for lang,start,new in [('zh','TAMSD 参数进一步区分了运动幅度与时间标度',zh),('en','The TAMSD parameters further distinguish displacement amplitude',en)]:
 p=root/f'manuscript_analysis_{lang}.md'
 s=p.read_text(encoding='utf-8')
 a=s.index(start);b=s.index('### ',a)
 s=s[:a]+new+s[b:]
 p.write_text(s,encoding='utf-8')
 print('Updated',p.name)
