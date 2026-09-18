# Solvent composition modulates the short-time motion of OPA

## 1. Data processing and calculation methods

### 1.1 Dataset organization and sampling

The data were divided into three groups. Group 1 comprised acetone/n-heptane mixtures with nominal acetone volume fractions of 1%, 5%, 10%, 20%, 30%, 40% and 50%, allowing comparison of the same solvent components at different concentrations. Group 2 comprised 14 pure solvents, allowing comparison of OPA motion in different solvent environments. Group 3 comprised five mixtures with different components at the same nominal volume ratio of 5:95: acetone/n-heptane, isopropanol/n-heptane, tetrahydrofuran/n-heptane, toluene/n-heptane and tetrahydrofuran/toluene. This group was used to compare component effects at a fixed volume ratio. Nominal volume fractions are distinct from molecular number fractions.

The systems were solution models containing a single octadecylphosphonic acid (OPA) molecule, and the simulation outputs were labelled as 300 K. Each condition was represented by one dynamical record spanning 0–10 ps and containing 101 sampled points at intervals of 0.1 ps. This sampling interval refers to the time between successive output records and is distinct from the molecular dynamics integration time step. The 5% acetone/95% n-heptane data were displayed in both Groups 1 and 3, giving 26 group entries but 25 distinct systems; the shared entry was not counted as an independent replicate. Hereafter, $N=101$ denotes the number of sampled points, and $t_i=i\Delta t$ denotes the sampling time, where $i=0,\ldots,N-1$ and $\Delta t=0.1$ ps.

### 1.2 Centre-of-mass motion and mean force

The OPA centre of mass was calculated by weighting the atomic positions by their masses:

$$
\mathbf R(t_i)
=\frac{\displaystyle\sum_{a\in\mathrm{OPA}}m_a\mathbf r_a(t_i)}
{\displaystyle\sum_{a\in\mathrm{OPA}}m_a},
$$

Here, $m_a$ and $\mathbf r_a(t_i)$ are the mass and position of atom $a$ in OPA, respectively. Successive centre-of-mass displacements were unwrapped using the periodic simulation cell and the minimum-image convention to obtain a continuous trajectory. In the following analysis, $\mathbf R(t_i)$ refers to the centre-of-mass position after this processing. The distance from the initial position and its maximum were defined as

$$
d(t_i)=\left\|\mathbf R(t_i)-\mathbf R(t_0)\right\|,
\qquad
d_{\max}=\max_{0\leq i\leq N-1}d(t_i).
$$

Here, $\|\cdot\|$ denotes the Euclidean norm of a three-dimensional vector. The distance $d(t_i)$ measures displacement from the initial position rather than cumulative distance travelled. The final displacement, $d(t_{N-1})$, was evaluated separately from the maximum displacement, $d_{\max}$. Displacements are reported in Å.

The molecular net force on OPA was obtained by summing its atomic force vectors at each sampled time, after which the magnitude of the net force was averaged over time:

$$
\mathbf F_{\mathrm{OPA}}(t_i)
=\sum_{a\in\mathrm{OPA}}\mathbf F_a(t_i),
$$

$$
\left\langle\left\|\mathbf F_{\mathrm{OPA}}\right\|\right\rangle
=\frac{1}{N}\sum_{i=0}^{N-1}
\left\|\sum_{a\in\mathrm{OPA}}\mathbf F_a(t_i)\right\|.
$$

The mean force is reported in eV Å$^{-1}$. It is the time-averaged magnitude of the molecular net force, which differs from both the magnitude of the mean force vector and the sum of the individual atomic force magnitudes. Because this statistic does not retain force direction or temporal correlations, it cannot be interpreted directly as a sustained driving force for translation, a friction coefficient or a solvent binding energy.

### 1.3 TAMSD calculation and power-law fitting

Following the single-trajectory framework of Kepten et al. [1, equations (1)–(2)], the time-averaged mean-squared displacement (TAMSD) was calculated from the three-dimensional centre-of-mass coordinates:

$$
\overline{\delta^2(\tau)}
=\frac{1}{N-m}\sum_{i=0}^{N-m-1}
\left\|\mathbf R(t_{i+m})-\mathbf R(t_i)\right\|^2,
\qquad \tau=m\Delta t.
$$

Here, $m$ is the integer lag and $\tau$ is the lag time; $N-m$ displacement pairs contribute to the average at each lag. TAMSD is expressed in Å². Unlike the distance from a fixed initial position, TAMSD averages squared displacements over multiple time origins to describe how molecular motion varies with lag time.

The maximum lag was set to

$$
m_{\max}=\left\lfloor\frac{N}{3}\right\rfloor=33,
$$

corresponding to a maximum lag time of 3.3 ps. After excluding the first lag point at 0.1 ps, the remaining 32 points over 0.2–3.3 ps were fitted by unweighted linear regression in logarithmic space:

$$
\overline{\delta^2(\tau)}=K_\alpha\tau^\alpha,
$$

$$
\ln\overline{\delta^2(\tau)}
=\ln K_\alpha+\alpha\ln\tau.
$$

The fits used numerical lag times expressed in ps and TAMSD values expressed in Å². The slope gives the effective scaling exponent, $\alpha$, and exponentiation of the natural-log intercept, $b$, gives $K_\alpha=\exp(b)$. The coefficient of determination, $R^2$, was evaluated in logarithmic space. This interval is the common fitting window used in Figures 2 and 3; it has not been established by the cited study as optimal for every system considered here. Values of $\alpha<1$ and $\alpha>1$ describe sublinear and superlinear TAMSD growth within the selected window, respectively. A single fit does not establish a stable subdiffusive or superdiffusive mechanism.

### 1.4 Physical meaning of the prefactor and comparison at a common time scale

The prefactor $K_\alpha$ has units of Å² ps$^{-\alpha}$ and is not a conventional diffusion coefficient. Because $\alpha$ varies between systems, the time dimension of the prefactor also varies. Comparisons across systems therefore used the fitted TAMSD at a common reference lag of $\tau_0=1$ ps:

$$
A_{1\mathrm{ps}}
=K_\alpha\tau_0^\alpha
=K_\alpha(1\,\mathrm{ps})^\alpha.
$$

When time is expressed in ps, $A_{1\mathrm{ps}}$ is numerically equal to the reported $K_\alpha$, but has the same units of Å² across all systems. This quantity is the value predicted by the power-law fit at the reference time and should be distinguished from the raw, unfitted TAMSD at the same lag.

## Results and discussion

### Short-time OPA motion in pure solvents (Figure 2)

The centre-of-mass displacement traces in 14 pure solvents reveal differences in the magnitude and temporal evolution of OPA excursions over 10 ps (Figure 2a). In n-heptane, the displacement reached 7.127 Å at 8.8 ps and remained at 6.063 Å at 10 ps, the largest final displacement among the pure solvents. DMC, CPME and toluene also retained substantial displacements from the initial position, ending at 5.044, 4.894 and 4.828 Å, respectively. By contrast, isopropanol reached an early maximum of 2.590 Å at 1.7 ps before returning to 0.647 Å at 10 ps. DMC and CPME reached their maxima at 9.3 and 9.7 ps, respectively, placing their largest excursions near the end of the observation period. The legend is ordered by decreasing final displacement and therefore follows the vertical order of the curve endpoints.

These traces measure distance from a fixed initial position: a decrease indicates that OPA moved closer to that position, rather than a reduction in cumulative distance travelled. In acetone, the displacement reached 4.351 Å at 4.2 ps and decreased to 3.015 Å at 10 ps, contrasting with the larger final displacement in n-heptane. The simulation snapshot in Figure 2b shows OPA surrounded by pure n-heptane and provides structural context for the dynamical comparison. A single configuration does not quantify solvation strength, interaction lifetimes or adsorption propensity.

The vertical coordinate in Figure 2c is the effective TAMSD scaling exponent, $\alpha$, which describes growth with lag time rather than the magnitude of TAMSD itself. In the power-law framework used by Kepten et al., $\alpha=1$ corresponds to the linear MSD scaling of normal diffusion, whereas $\alpha<1$ and $\alpha>1$ describe sublinear and superlinear growth, respectively [1]. Within this physical framework, the present figure uses four display intervals: $\alpha<0.6$, $0.6\leq\alpha<0.8$, $0.8\leq\alpha<1.0$ and $\alpha\geq1.0$. The first three describe progressively less sublinear growth approaching the linear scaling of normal diffusion, while the last includes the linear boundary and superlinear growth. These fitted exponents characterize the selected lag-time window rather than independently establishing long-time transport mechanisms.

Ethyl acetate, isopropanol, propylene carbonate and p-xylene occupied the $\alpha<0.6$ region, while cyclohexane, methanol and THF fell within the pale blue $0.6\leq\alpha<0.8$ band. The pale orange $0.8\leq\alpha<1.0$ band contained n-heptane, acetone, toluene, ethanol and acetonitrile, whose TAMSD growth was closer to linear. CPME and DMC had exponents of 1.105 and 1.031, respectively, indicating superlinear growth within the fitted window. Vertical comparisons therefore distinguish temporal scaling; a higher exponent does not necessarily indicate better performance in promoting OPA motion or film formation.

The horizontal coordinate, $K_\alpha$, provides complementary information on displacement amplitude. At a fixed $\alpha$, increasing $K_\alpha$ raises the fitted TAMSD at every lag. When exponents differ, amplitudes are compared at a common reference lag of 1 ps using $A_{1\mathrm{ps}}=K_\alpha(1\,\mathrm{ps})^\alpha$, which has consistent units across systems. With time expressed in ps, $A_{1\mathrm{ps}}$ is numerically equal to the reported $K_\alpha$; movement to the right thus indicates a larger fitted mean-squared displacement at the reference lag. Propylene carbonate had the smallest amplitude, 0.490 Å², whereas n-heptane and acetone ranked first and second at 1.722 and 1.694 Å². Although CPME and DMC had higher exponents, their amplitudes of 1.573 and 1.310 Å² were lower than those of n-heptane and acetone.

Considering both coordinates, n-heptane and acetone combined exponents in the near-linear $0.8\leq\alpha<1.0$ interval, at 0.906 and 0.876, with the two largest TAMSD prefactors among the pure solvents. This combination of large displacement amplitude and nearly linear temporal growth identifies favourable short-time OPA translational dynamics in both solvents, consistent with their favourable experimental performance.

### Composition-dependent OPA motion in mixed solvents (Figure 3)

Figure 3a shows the solvent environment around OPA in 5% acetone/95% n-heptane. Comparison with the pure n-heptane model in Figure 2b illustrates the change in composition, but the local arrangement in a single snapshot does not establish preferential enrichment of acetone around OPA. Figure 3b compares the complete three-dimensional centre-of-mass trajectories in this mixture, pure n-heptane and pure acetone. Each trajectory is referenced to its own initial position, with black circles and coloured squares marking the start and end.

The acetone concentration series exhibited a non-monotonic dynamical response (Figure 3c). Among the seven compositions, the 5% mixture had the largest fitted amplitude, $A_{1\mathrm{ps}}=1.984$ Å², with $\alpha=0.861$. The 1% mixture had a similar exponent, 0.860, but a smaller amplitude of 1.580 Å². Increasing the acetone fraction to 10% and 20% reduced the amplitude to 1.650 and 1.366 Å² while raising the exponent to 0.890 and 0.966, respectively. The 20% mixture thus showed the most nearly linear TAMSD growth, rather than the largest reference-lag amplitude. At 30% acetone, the exponent decreased to 0.450 before recovering to 0.744 and 0.807 at 40% and 50%. The corresponding amplitudes were 1.127, 1.031 and 1.417 Å². Increasing acetone content consequently did not produce a sustained increase in short-time displacement amplitude.

The complete displacement traces provide complementary evidence (Figure 3d). This panel contains 11 distinct mixtures from the concentration and fixed-ratio composition series, with the shared 5% acetone/95% n-heptane trajectory plotted once. The 5% acetone mixture reached a maximum displacement of 5.362 Å at 8.8 ps and retained 4.357 Å at 10 ps, the highest maximum and final displacements among these mixtures. The 10% acetone mixture ranked second in final displacement, at 3.794 Å. The final values for 20% acetone, 5% toluene and 5% isopropanol in n-heptane were close, at 3.071, 3.065 and 3.037 Å, respectively; single trajectories do not resolve a reliable ranking of their underlying mobilities. In contrast, 5% THF/95% n-heptane reached 2.173 Å at 2.3 ps and ended at 1.084 Å, indicating an early excursion followed by a return towards the starting position.

Among the five mixtures at a nominal volume ratio of 5:95, changing the components affected both fitted amplitude and exponent (Figure 3e). The largest amplitude occurred in 5% acetone/95% n-heptane, followed by 5% isopropanol/95% n-heptane, with $A_{1\mathrm{ps}}=1.578$ Å² and $\alpha=0.821$. The 5% toluene/95% n-heptane mixture had the highest exponent, 0.919, but a smaller amplitude of 1.136 Å². The lowest exponent, 0.516, occurred in 5% THF/95% n-heptane, whose amplitude was 0.957 Å². Replacing n-heptane with toluene as the major solvent raised the exponent of the 5% THF mixture to 0.697 but reduced its amplitude to 0.879 Å². Cosolvent and major-solvent identity therefore influenced displacement amplitude and temporal scaling in different ways.

Together, Figures 2 and 3 show that adding 5% acetone to n-heptane increased the fitted TAMSD at 1 ps from 1.722 to 1.984 Å², while reducing the maximum displacement from 7.127 to 5.362 Å and the final displacement from 6.063 to 4.357 Å. The mixture therefore stands out in its reference-lag amplitude and, within the tested mixtures, its maximum and final excursions; it does not exceed pure n-heptane in every measure of motion. These observations are compatible with the favourable experimental performance of the mixture of 95% n-heptaneand and 5% acetone, providing a short-time translational perspective on those outcomes.

## Figure legends

**Figure 2 | Short-time OPA motion in pure solvents.** **a**, Distance of the OPA centre of mass from its initial position in 14 pure solvents over 10 ps. The legend is ordered by decreasing final displacement. **b**, Simulation snapshot of OPA and surrounding n-heptane molecules. **c**, TAMSD prefactor, $K_\alpha$, versus effective exponent, $\alpha$, obtained from power-law fits over 0.2–3.3 ps. Horizontal dashed lines mark $\alpha=0.6$, 0.8 and 1.0; pale blue and orange bands indicate the 0.6–0.8 and 0.8–1.0 intervals for visual grouping only. The horizontal arrow denotes increasing fitted amplitude, rather than an established increase in macroscopic diffusivity. Each system is represented by one trajectory containing 101 sampled points; uncertainty from independent replicates is not shown.

**Figure 3 | Mixed-solvent composition modulates short-time OPA motion.** **a**, Simulation snapshot of OPA in 5% acetone/95% n-heptane. **b**, Three-dimensional centre-of-mass trajectories in 5% acetone/95% n-heptane, pure n-heptane and pure acetone, each referenced to its own starting position. Black circles mark the start at 0 ps and coloured squares mark the end at 10 ps. Axis limits are ±4.5, ±7.5 and ±4 Å, respectively, with equal XYZ scales within each panel. **c**, TAMSD parameters for acetone/n-heptane mixtures containing nominal acetone volume fractions of 1%, 5%, 10%, 20%, 30%, 40% and 50%. **d**, Displacement traces for 11 distinct mixtures, with the legend ordered by decreasing final displacement. **e**, TAMSD parameters for five mixtures at a nominal volume ratio of 5:95. The fitting window, shaded bands and dashed lines in **c,e** follow Figure 2c; horizontal arrows denote increasing fitted amplitude. The 5% acetone mixture shares the same data in **c,e** and is counted once in **d**. Percentages denote nominal volume fractions. $K_\alpha$ has units of Å² ps$^{-\alpha}$; amplitudes are compared across systems using fitted TAMSD at a common reference lag of 1 ps.

## Reference

1. Kepten, E., Weron, A., Sikora, G., Burnecki, K. & Garini, Y. Guidelines for the fitting of anomalous diffusion mean square displacement graphs from single particle tracking experiments. *PLOS ONE* **10**, e0117722 (2015). [doi:10.1371/journal.pone.0117722](https://doi.org/10.1371/journal.pone.0117722).
