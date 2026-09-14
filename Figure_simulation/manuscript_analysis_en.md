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

The fits used numerical lag times expressed in ps and TAMSD values expressed in Å². The slope gives the effective scaling exponent, $\alpha$, and exponentiation of the natural-log intercept, $b$, gives $K_\alpha=\exp(b)$. The coefficient of determination, $R^2$, was evaluated in logarithmic space. This interval is the common fitting window used in Figure X; it has not been established by the cited study as optimal for every system considered here. Values of $\alpha<1$ and $\alpha>1$ describe sublinear and superlinear TAMSD growth within the selected window, respectively. A single fit does not establish a stable subdiffusive or superdiffusive mechanism.

### 1.4 Physical meaning of the prefactor and comparison at a common time scale

The prefactor $K_\alpha$ has units of Å² ps$^{-\alpha}$ and is not a conventional diffusion coefficient. Because $\alpha$ varies between systems, the time dimension of the prefactor also varies. Comparisons across systems therefore used the fitted TAMSD at a common reference lag of $\tau_0=1$ ps:

$$
A_{1\mathrm{ps}}
=K_\alpha\tau_0^\alpha
=K_\alpha(1\,\mathrm{ps})^\alpha.
$$

When time is expressed in ps, $A_{1\mathrm{ps}}$ is numerically equal to the reported $K_\alpha$, but has the same units of Å² across all systems. This quantity is the value predicted by the power-law fit at the reference time and should be distinguished from the raw, unfitted TAMSD at the same lag.

## Results and discussion

The displacement trajectories, time-averaged mean-squared displacement (TAMSD) parameters and mean molecular forces reveal distinct solvent-dependent responses of octadecylphosphonic acid (OPA) over 10 ps (Figure X). The TAMSD was described by $\overline{\delta^2(\tau)}=K_\alpha\tau^\alpha$, where $K_\alpha$ characterizes the fitted amplitude and $\alpha$ describes its growth with lag time. These parameters distinguish the magnitude of molecular motion from its temporal scaling. To compare amplitudes across systems with different exponents, we use the fitted TAMSD at a reference lag of 1 ps, $A_{1\mathrm{ps}}=K_\alpha(1\mathrm{ps})^\alpha$. Its numerical value equals that of the reported $K_\alpha$ when time is expressed in ps, but its units are uniformly Å². The maximum displacement, $d_{\max}$, provides a complementary measure of the largest excursion from the initial position.

### Acetone concentration separates amplitude from scaling (Figure Xa)

In the acetone/n-heptane series, increasing acetone content produced a non-monotonic response (Figure Xa). The 5% acetone mixture had the largest fitted amplitude, $A_{1\mathrm{ps}}=1.984$ Å², and maximum displacement, $d_{\max}=5.362$ Å, among the seven compositions. Its exponent, $\alpha=0.861$, was similar to that of the 1% mixture, $\alpha=0.860$, despite their different maximum displacements of 5.362 and 2.860 Å. Increasing acetone content to 20% raised the exponent to 0.966, the value closest to unity in this series, while reducing the fitted amplitude to 1.366 Å² and the maximum displacement to 3.517 Å. Thus, the composition with the most nearly linear TAMSD growth differed from that with the greatest displacement amplitude.

A further increase from 20% to 30% acetone reduced $\alpha$ from 0.966 to 0.450 and $d_{\max}$ from 3.517 to 2.396 Å. The mean net-force magnitude remained nearly unchanged, at 0.506 and 0.502 eV Å$^{-1}$, respectively. At 40% and 50% acetone, the exponent recovered to 0.744 and 0.807, whereas the maximum displacements remained modest, at 2.562 and 2.264 Å. The 50% mixture illustrates the distinction between motion at different time scales: its fitted amplitude recovered to 1.417 Å², but its maximum displacement was the smallest in the concentration series. These observations show that neither increasing acetone content nor increasing the mean force consistently promotes larger molecular excursions. Within the acetone/n-heptane concentration series examined here, the 5% acetone mixture exhibited the largest short-time displacement amplitude, consistent with the favourable experimental performance of this composition.

### Pure solvents span distinct TAMSD responses (Figure Xb)

The pure-solvent comparison similarly separated displacement amplitude from temporal scaling (Figure Xb). Among the 14 solvents, n-heptane had the largest fitted amplitude, 1.722 Å², and the largest maximum displacement, 7.127 Å, with $\alpha=0.906$. Cyclopentyl methyl ether (CPME) and dimethyl carbonate (DMC) had the highest exponents, 1.105 and 1.031, but smaller maximum displacements of 5.349 and 5.307 Å. Their fitted amplitudes were 1.573 and 1.310 Å², respectively. The exponents above unity describe superlinear TAMSD growth within the fitted interval; they do not, by themselves, establish persistent superdiffusion. Acetone combined an amplitude close to that of n-heptane, 1.694 Å², with a similar exponent of 0.876, yet its maximum displacement was only 4.351 Å. This difference emphasizes that a TAMSD fit over intermediate lag times and the largest excursion over an entire trajectory need not yield the same solvent ranking.

At the other end of the pure-solvent comparison, propylene carbonate had the smallest fitted amplitude, 0.490 Å², and maximum displacement, 1.713 Å, together with a low exponent of 0.425. The low exponents for p-xylene and isopropanol, 0.407 and 0.464, also indicated sublinear growth. Toluene, by contrast, had an exponent of 0.892 and a maximum displacement of 4.828 Å, compared with 2.089 Å in p-xylene. Ethanol and isopropanol likewise differed in their exponents, 0.860 and 0.464, despite having similarly high mean forces of 0.717 and 0.707 eV Å$^{-1}$. Among the pure solvents examined, n-heptane and acetone ranked first and second in the TAMSD prefactor, $K_\alpha$, corresponding to fitted TAMSD values of 1.722 and 1.694 Å² at a reference lag of 1 ps. Their exponents of 0.906 and 0.876 indicate that both solvents combine relatively large short-time displacement amplitudes with slightly sublinear TAMSD growth (Figure Xb).

### Mixture composition changes the amplitude–exponent relationship (Figure Xc)

In the mixed-solvent comparison, 5% acetone/95% n-heptane again had the largest fitted amplitude and maximum displacement (Figure Xc). The 5% isopropanol/95% n-heptane mixture followed with $A_{1\mathrm{ps}}=1.578$ Å², $\alpha=0.821$ and $d_{\max}=3.651$ Å. Toluene/n-heptane had the highest exponent in this group, 0.919, but a smaller fitted amplitude of 1.136 Å² and a maximum displacement of 3.445 Å. Tetrahydrofuran (THF)/n-heptane exhibited the lowest exponent, 0.516, and the smallest maximum displacement, 2.173 Å, despite having the highest mean force in the mixture comparison, 0.535 eV Å$^{-1}$. Replacing n-heptane with toluene in the THF mixture increased $\alpha$ to 0.697 and $d_{\max}$ to 2.448 Å, while decreasing the fitted amplitude from 0.957 to 0.879 Å² and the mean force to 0.406 eV Å$^{-1}$. The base solvent thus altered the scaling, amplitude and force measures in different ways.

Comparisons with the pure solvents further show that cosolvent effects depend on the observable. Relative to pure n-heptane, the 5% acetone mixture increased the fitted TAMSD at 1 ps from 1.722 to 1.984 Å², while decreasing the maximum displacement from 7.127 to 5.362 Å. Acetone addition therefore enhanced the fitted amplitude at this reference lag without increasing the largest excursion over 10 ps. Conversely, changing from pure isopropanol to its 5% mixture in n-heptane increased $\alpha$ from 0.464 to 0.821 and $d_{\max}$ from 2.590 to 3.651 Å, while reducing the mean force from 0.707 to 0.452 eV Å$^{-1}$. The behaviour of a pure solvent consequently cannot be transferred directly to its role in a mixture.

### Complete trajectories resolve excursions and returns (Figure Xd)

The displacement traces reinforce the distinction between a large excursion and sustained separation from the initial position (Figure Xd). In n-heptane, the displacement reached 7.127 Å but ended at 6.063 Å. The corresponding values were 5.362 and 4.357 Å for the 5% acetone mixture, and 2.590 and 0.647 Å for pure isopropanol. These decreases demonstrate that the plotted trajectories include returns towards the initial position; they are not cumulative distances travelled. Together with the force comparison, they show why neither the maximum displacement nor the average force alone provides a complete description of mobility. In particular, averaging the force magnitude removes the directional and temporal information needed to determine whether successive forces reinforce or oppose molecular translation.

The time at which the maximum displacement occurred further distinguishes these trajectories. Isopropanol reached its maximum of 2.590 Å at 1.7 ps, and THF/n-heptane reached 2.173 Å at 2.3 ps, ending at 0.647 and 1.084 Å, respectively. In contrast, n-heptane and the 5% acetone mixture reached their maxima at 8.8 ps, while CPME and DMC reached theirs at 9.7 and 9.3 ps. The trajectory panel thus distinguishes early excursions followed by substantial returns from trajectories whose largest excursions occur near the end of the observation period. These differences help interpret the low exponents for isopropanol and THF/n-heptane in Figure Xb,c, but neither a return towards the origin nor a late maximum alone identifies a specific transport mechanism. The distance traces also omit the direction of motion and should not be interpreted as three-dimensional paths.

### Mean force varies independently of the TAMSD ranking (Figure Xe)

The mean-force panel spans 0.397–0.717 eV Å$^{-1}$ across the 25 distinct conditions (Figure Xe). Within the acetone concentration series, the mean force rose from 0.397 eV Å$^{-1}$ at 1% acetone to 0.480 at 10%, reached 0.506 at 20%, and changed little at 30% before increasing to 0.542 at 40% and decreasing slightly to 0.532 at 50%. This overall increase contrasts with the non-monotonic trajectory amplitudes and exponents in Figure Xa. In particular, the 5% mixture had the largest fitted amplitude despite a comparatively low mean force of 0.421 eV Å$^{-1}$.

Among pure solvents, ethanol, isopropanol and methanol had mean forces of 0.717, 0.707 and 0.637 eV Å$^{-1}$, respectively, whereas cyclohexane, n-heptane and toluene had lower values of 0.409, 0.434 and 0.437 eV Å$^{-1}$. The mixture comparison showed a similar separation between force and motion: THF/n-heptane had the highest mean force, 0.535 eV Å$^{-1}$, but the lowest exponent, while THF/toluene had the lowest mean force, 0.406 eV Å$^{-1}$, without having the largest fitted amplitude. The circular markers in Figure Xe summarize the magnitude of the net force and cannot resolve force direction, cancellation over time or force–velocity correlations. Consequently, these values do not measure binding strength, friction or the efficiency with which solvent forces produce net translation.

### Maximum displacement provides a complementary measure of excursion (Figure Xf)

The maximum-displacement panel spans 1.713–7.127 Å (Figure Xf). Pure n-heptane had the largest excursion, followed numerically by the 5% acetone mixture, CPME and DMC, with values of 5.362, 5.349 and 5.307 Å. The latter three values are close and do not support a resolved ranking of their underlying mobilities from single trajectories. Toluene reached 4.828 Å, whereas propylene carbonate, p-xylene and THF/n-heptane remained at 1.713, 2.089 and 2.173 Å, respectively. Within the acetone series, the 5% condition had the largest maximum displacement and the 50% condition the smallest, even though the lowest exponent occurred at 30%. The diamond markers therefore report a different aspect of motion from the vertical coordinate in Figure Xa–c.

The shared row order in Figure Xe,f allows the force and displacement of the same condition to be compared directly. Ethanol and isopropanol occupy the high-force end of Figure Xe but have maximum displacements of only 3.717 and 2.590 Å, whereas n-heptane combines a lower mean force with the largest displacement. Low force is not sufficient for a large excursion either: cyclohexane has a mean force of 0.409 eV Å$^{-1}$ but a maximum displacement of 2.855 Å.

Taken together, the TAMSD parameters and complete displacement trajectories identify the 5% acetone mixture as the composition with the most pronounced short-time displacement amplitude within the acetone/n-heptane series (Figure Xa,d,f). This mixture had the highest TAMSD prefactor in the series, corresponding to a fitted TAMSD of 1.984 Å² at 1 ps. Its maximum displacement and final displacement at 10 ps were also the highest in the series, reaching 5.362 and 4.357 Å, respectively. Although the displacement decreased after its peak at 8.8 ps, OPA remained substantially displaced from its initial position at the end of the trajectory. Thus, based on the amplitude at a common reference lag, the maximum excursion and the final net displacement, the 5% acetone mixture showed the most favourable short-time translational behaviour among the compositions examined.

### Interpretation across the six panels

Across the six panels, the simulations are broadly consistent with the favourable experimental performance of n-heptane and acetone (Figure X). Among the pure solvents, n-heptane combined a relatively low mean force of approximately 0.434 eV Å$^{-1}$ with the largest maximum displacement across all conditions, 7.127 Å. Acetone had an intermediate mean force of approximately 0.527 eV Å$^{-1}$ and a maximum displacement of 4.351 Å, placing it towards the higher end of the pure-solvent comparison. The 5% acetone/95% n-heptane mixture combined a low mean force of 0.421 eV Å$^{-1}$ with a large maximum displacement of 5.362 Å, the highest in both the acetone concentration series and the mixture comparison. All three conditions also had relatively large fitted TAMSD amplitudes, indicating comparatively pronounced OPA translation within the simulated time window. These observations provide a short-time molecular-motion perspective consistent with the favourable experimental outcomes, but do not establish lower force or larger displacement as the direct cause of improved film formation.

## Figure legend

**Figure X | Solvent-dependent short-time dynamics and mean molecular force of OPA.** **a–c**, TAMSD prefactor, $K_\alpha$, versus effective scaling exponent, $\alpha$, for the acetone/n-heptane concentration series (**a**, upper left), 14 pure solvents (**b**, upper middle) and five mixed-solvent conditions (**c**, upper right). Parameters were obtained by fitting $\overline{\delta^2(\tau)}=K_\alpha\tau^\alpha$ over 0.2–3.3 ps; the dashed line denotes $\alpha=1$. **d**, Distance of the OPA centre of mass from its initial position over 10 ps (lower left). **e**, Time-averaged magnitude of the molecular net force, shown as circles (left of the two lower-right panels). **f**, Maximum displacement from the initial position, shown as diamonds (rightmost panel). Panels **e** and **f** use the same condition order; each row refers to the same solvent condition across the two panels, and the vertical position is categorical rather than a measured variable. Colours identify solvent conditions according to the shared key. Each distinct condition is represented by one trajectory containing 101 sampled points. The 5% acetone/95% n-heptane trajectory is shared by panels **a** and **c**, represents the same motion in panel **d**, and is counted once in panels **e** and **f**. $K_\alpha$ has units of Å² ps$^{-\alpha}$ and is not a conventional diffusion coefficient. No uncertainty estimates from independent replicates are shown.

## Reference

1. Kepten, E., Weron, A., Sikora, G., Burnecki, K. & Garini, Y. Guidelines for the fitting of anomalous diffusion mean square displacement graphs from single particle tracking experiments. *PLOS ONE* **10**, e0117722 (2015). [doi:10.1371/journal.pone.0117722](https://doi.org/10.1371/journal.pone.0117722).
