# Technical note: the MLE method in inptk

This note gives the math behind `method="mle"`, the default way inptk estimates
ice-nucleating particle (INP) concentrations: the physical model, the
likelihood, the fit and the uncertainty bounds. Sections 1 to 12 are the math.
Section 13 lists the implementation details that do not change the math, and
Section 14 maps each formula to the code in `src/inptk/curve_fit.py` and
`src/inptk/_engine/curve_likelihood.py`.

## 1. Notation

Sections 2, 3, 6 and 7 describe one dilution: the droplets of one dilution in one measurement, run and cycle, with a fixed number of droplets. Section 4 adds an index $s$ when several dilutions are combined.

| Symbol | Meaning |
|---|---|
| $N$ | number of droplets |
| $V$ | volume of one droplet (mL) |
| $d$ | dilution factor |
| $T_0 > T_1 > \dots > T_J$ | observation temperatures, warm to cold |
| $F_j$ | cumulative number of frozen droplets at $T_j$ |
| $n_j = F_j - F_{j-1}$ | droplets that froze in interval $j$ ($F_{-1}=0$) |
| $K(T)$ | cumulative INP concentration of the sample (per mL suspension) |
| $B(T)$ | cumulative water-background concentration (per mL) |
| $\lambda(T)$ | expected number of active INPs in one droplet at $T$ |

## 2. Physical model

We assume singular (time-independent) freezing: each INP becomes active at a fixed temperature, and a droplet freezes once it holds at least one active INP. INPs are Poisson-distributed among droplets, and droplets freeze independently. A droplet holding on average $\lambda$ active INPs stays liquid with probability $e^{-\lambda}$:

```math
P(\text{liquid at } T) = e^{-\lambda(T)}.
```

A droplet contains $V/d$ mL of the original suspension and $V$ mL of water in total, so

```math
\lambda(T) = V\left(\frac{K(T)}{d} + B(T)\right).
```

The background scales with the full droplet volume and does not depend on dilution. For a water blank, $d \to \infty$ and $\lambda = V B$. Without blank correction the $B$ term is dropped.

## 3. Likelihood

Write $\lambda_j = \lambda(T_j)$ and $\lambda_{-1}=0$. Every droplet ends in exactly one of $J+2$ outcomes: it first froze in the interval ending at $T_j$ ($j = 0,\dots,J$), or it was still liquid at the last observation.

**Step 1: outcome probabilities.** A droplet first freezes in interval $j$ if it was liquid at $T_{j-1}$ but not at $T_j$:

```math
p_j = P(\text{liquid at } T_{j-1}) - P(\text{liquid at } T_j) = e^{-\lambda_{j-1}} - e^{-\lambda_j}, \qquad j = 0,\dots,J.
```

It stays liquid to the end with probability

```math
q = e^{-\lambda_J}.
```

The sum telescopes: $\sum_j p_j + q = e^{-\lambda_{-1}} = 1$.

**Step 2: likelihood.** $n_j$ droplets froze in interval $j$ and $N - F_J$ stayed liquid; these counts add up to $N$. Droplets are independent, so the counts are multinomial:

```math
L = \frac{N!}{\prod_j n_j!\,(N - F_J)!}\ \prod_{j=0}^{J} p_j^{\,n_j}\ q^{\,N - F_J}.
```

**Step 3: take the log.** The product becomes a sum. The multinomial coefficient does not depend on $K$ or $B$, so it is a constant:

```math
\log L = \text{const} + \sum_{j=0}^{J} n_j\log p_j + (N - F_J)\log q.
```

**Step 4: simplify each term.** Factor $e^{-\lambda_{j-1}}$ out of $p_j$:

```math
p_j = e^{-\lambda_{j-1}}\left(1 - e^{-(\lambda_j-\lambda_{j-1})}\right) \quad\Longrightarrow\quad \log p_j = -\lambda_{j-1} + \log\left(1-e^{-(\lambda_j-\lambda_{j-1})}\right),
```

```math
\log q = -\lambda_J.
```

**Step 5: substitute.**

```math
\log L = \sum_{j=0}^{J} n_j\left[-\lambda_{j-1} + \log\left(1-e^{-(\lambda_j-\lambda_{j-1})}\right)\right] - (N - F_J)\,\lambda_J + \text{const}.
```

The first part is the chance of staying liquid until the interval starts, and the log term is the chance of then freezing within it.

**Step 6: all dilutions.** With several dilutions (including those from other runs, and the blanks), label each by $s$ and write $L_s$ for its likelihood. The dilutions are independent, so their likelihoods multiply and the logs add. The quantity minimized is the negative log-likelihood:

```math
\log L = \sum_s \log L_s, \qquad \ell = -\log L.
```

Each droplet is counted once, in the interval where it first froze. Treating every image as an independent binomial trial would count the same droplet many times and give intervals that are too narrow.

## 4. A monotone curve as non-negative steps

From here on each dilution has an index $s$: $N_s$, $V_s$, $d_s$, $F_{s,j}$, $n_{s,j}$, $\lambda_{s,j}$, observed at $T_{s,0} > \dots > T_{s,J_s}$. Each run and cycle $r$ has its own background $B_r$, and dilution $s$ belongs to run $r(s)$, so $\lambda_{s,j} = V_s\left(K(T_{s,j})/d_s + B_{r(s)}(T_{s,j})\right)$.

$K$ can only increase on cooling. On a temperature grid $t_1 > t_2 > \dots > t_M$ (by default every observed temperature in the curve) write

```math
K(T) = \sum_{k=1}^{M} \kappa_k\,\mathbf{1}[t_k \ge T], \qquad \kappa_k \ge 0,
```

and likewise $B_r(T) = \sum_k \beta_{r,k}\,\mathbf{1}[t_k \ge T]$ with $\beta_{r,k}\ge 0$. Collect all steps in one vector $x = (\kappa, \beta_1, \beta_2, \dots) \ge 0$. Then every $\lambda_{s,j}$ is linear in $x$:

```math
\lambda_{s,j} = a_{s,j}\cdot x,
```

where $a_{s,j}$ holds $V_s/d_s$ in the $\kappa_k$ entries with $t_k \ge T_{s,j}$ and $V_s$ in the matching $\beta_{r(s),k}$ entries.

Let each freezing event $e$ (a pair $(s,j)$ with $n_{s,j}>0$) have count $n_e$ and step vector $\Delta_e = a_{s,j} - a_{s,j-1}$. Substituting into Section 3 and summing over dilutions, the negative log-likelihood is

```math
\ell(x) = c\cdot x - \sum_e n_e \log\left(1 - e^{-\Delta_e\cdot x}\right),
```

```math
c = \sum_s\left[\sum_j n_{s,j}\,a_{s,j-1} + (N_s - F_{s,J_s})\,a_{s,J_s}\right].
```

The first term is the expected number of INPs the droplets avoided while they stayed liquid. The second term rewards the freezing events. The estimate is

```math
\hat x = \arg\min_{x\ge 0}\ \ell(x), \qquad \hat K(T) = \sum_k \hat\kappa_k\,\mathbf{1}[t_k \ge T].
```

## 5. The minimum is unique in value

Let $g(u) = -\log(1-e^{-u})$ for $u>0$. Then

```math
g'(u) = -\frac{e^{-u}}{1-e^{-u}}, \qquad g''(u) = \frac{e^{-u}}{(1-e^{-u})^2} > 0.
```

$g$ is convex, so $\ell$ (a linear term plus convex functions of linear functions of $x$) is convex, and the constraint $x\ge 0$ is convex. Every local minimum is a global minimum. The gradient is

```math
\nabla \ell(x) = c - \sum_e n_e\,\frac{e^{-\Delta_e\cdot x}}{1-e^{-\Delta_e\cdot x}}\,\Delta_e,
```

and $\ell$ is minimized with a bound-constrained quasi-Newton method (L-BFGS-B).

Two cases are handled separately. If two steps enter every $a_{s,j}$ identically (no observation between their temperatures), the data cannot separate them; they are merged and the sum is placed at the colder temperature. If a column of $c$ is zero, no droplet was seen liquid after that step, so $\ell$ decreases without bound as that step grows; any $K(T)$ that includes it is reported as undefined.

## 6. Observations where nothing froze can be dropped

If a dilution has no new freezing between $T_j$ and $T_m$, the frozen count stays at $F$ and each of those intervals only adds $(N - F)(\lambda_i-\lambda_{i-1})$ to $\ell$. The sum telescopes:

```math
\sum_{i=j+1}^{m}(N-F)(\lambda_i-\lambda_{i-1}) = (N-F)(\lambda_m-\lambda_j).
```

So the interior observations can be removed without changing $\ell$. Only the edges of each increase and the first and last observations are needed.

## 7. Check: one dilution gives the Vali formula

Take one dilution without a blank, on its own observation grid. With $u_j = \lambda_j - \lambda_{j-1} = (V/d)\,\kappa_j \ge 0$, the negative log-likelihood separates:

```math
\ell = \sum_j \left[(N-F_j)\,u_j - n_j\log(1-e^{-u_j})\right].
```

Setting $\partial\ell/\partial u_j = 0$:

```math
N - F_j = n_j\,\frac{e^{-u_j}}{1-e^{-u_j}} \quad\Longrightarrow\quad e^{-u_j} = \frac{N-F_j}{N-F_{j-1}}.
```

Multiplying the steps gives $e^{-\lambda_j} = (N-F_j)/N = 1-f_j$, with frozen fraction $f_j = F_j/N$. Therefore

```math
\hat K(T_j) = \frac{d}{V}\left(-\ln(1-f_j)\right),
```

which is Vali's formula. With a water blank of the same droplet volume, the same argument gives $V B(T_j) = -\ln(1-f_j^{\text{blank}})$ and

```math
\hat K(T_j) = \frac{d}{V}\left[-\ln(1-f_j) + \ln(1-f_j^{\text{blank}})\right],
```

the usual blank subtraction on the $-\ln(1-f)$ scale. This holds while the result increases on cooling. Where the blank would make a step negative, the constraint $\kappa_k\ge 0$ becomes active and the joint fit gives the best monotone compromise.

Numerical check: for $N=60$, $V=0.05$ mL, $d=10$ and $F = 0, 3, 10, 25, 40, 48$, the fitted $\hat K$ matches $\frac{d}{V}(-\ln(1-f))$ to about seven significant figures, with and without a blank.

## 8. Uncertainty: profile likelihood

For a temperature $T$, the target $\theta = K(T) = w\cdot x$ is linear in $x$, where $w$ selects the $\kappa_k$ with $t_k \ge T$. The profile negative log-likelihood is

```math
\ell_p(\theta) = \min_{x\ge 0,\ w\cdot x = \theta}\ \ell(x),
```

and the confidence interval is

```math
\left\{\theta : \ell_p(\theta) - \ell(\hat x) \le \delta\right\}, \qquad \delta = \frac{z^2}{2}.
```

By Wilks' theorem, $2[\ell_p(\theta)-\ell(\hat x)]$ is approximately $\chi^2_1$ at the true $\theta$. With $z = 1.96$, $\delta = 1.9207$, about 95 % coverage. The profile minimizes over all other parameters, including the background curves, so the blank's uncertainty is included once. The intervals are usually asymmetric.

**Computing the endpoints.** Because $\ell$ is convex, the set $C = \{x\ge 0: \ell(x)\le \ell(\hat x)+\delta\}$ is convex, and the endpoints are the minimum and maximum of $w\cdot x$ over $C$. They are found through the Lagrangian form

```math
x(\mu) = \arg\min_{x\ge 0}\ \ell(x) + \mu\, w\cdot x,
```

which only changes $c \to c + \mu w$. $\mu>0$ pushes $K(T)$ down and $\mu<0$ pushes it up. A one-dimensional root search finds the $\mu$ where $\ell(x(\mu)) = \ell(\hat x)+\delta$, and the endpoint is $w\cdot x(\mu)$.

Two edge cases:

- **Lower bound at zero.** If the fit with every step in $w$ fixed at zero is still within $\delta$ of the best, the lower bound is 0.
- **Upper bound through a step with no events.** If no freezing event involves a step $k$, $\ell$ is linear in $\kappa_k$, and the remaining budget goes to it directly: $\Delta\kappa_k = (\text{remaining } \delta)/c_k$.

When $T$ lies between merged temperatures (Section 5), the merge is redone with $w$ as an extra row, so the bound allows the INPs to sit on either side of $T$.

**Check.** For one dilution without a blank, the profile interval for $K(T_j)$ equals the binomial likelihood-ratio interval for $f_j$, mapped through $K = \frac{d}{V}(-\ln(1-f))$, because a likelihood ratio does not depend on the parametrization. For the example in Section 7 the two agree to at least five significant figures.

The bounds are pointwise (each temperature has its own interval) and approximate (Wilks' theorem is asymptotic, and is less accurate when very few or nearly all droplets freeze, or near $K = 0$).

## 9. Combining dilutions

A combined curve has one shared $K(T)$. Each dilution enters with its own $V_s/d_s$, and all log-likelihoods add. Nothing is averaged by hand. A dilution carries the most information where its droplets are partly frozen, and little where they are all liquid or all frozen; this weighting follows from the likelihood. Only one cycle per run enters a curve, because repeated cycles reuse the same droplets.

## 10. Water-blank options

- Each run and cycle has its own background curve $B_r$, fitted jointly with the samples.
- **Blank after first freeze** (optional): $B_r(T) = 0$ for all $T$ warmer than that blank's first freeze; a blank that never freezes gives $B_r \equiv 0$.
- **Blank temperature limits** restrict which blank observations enter the fit, not the samples.

## 11. Units

The fit gives $K$ per mL of original suspension. Other bases multiply by a constant:

```math
K_{\text{air}} = K\,\frac{V_{\text{susp}}}{V_{\text{air}}\, f_{\text{filter}}}\ \ [\text{L}^{-1}], \qquad K_{\text{soil}} = K\,\frac{V_{\text{susp}}}{m_{\text{dry}}}\ \ [\text{g}^{-1}].
```

A profile-likelihood interval is unchanged by rescaling, so the bounds scale by the same factor.

## 12. Assumptions

1. Singular freezing: fixed activation temperature per INP, no cooling-rate dependence.
2. INPs are Poisson-distributed among droplets; droplets freeze independently.
3. The water background is proportional to droplet volume, independent of dilution, and the same for all wells in one run and cycle.
4. Each droplet is counted once, at its first freezing.
5. Uncertainty bounds are approximate and pointwise (Wilks' theorem).

## 13. Implementation details

These choices are made in the code. None of them changes the likelihood above.

**Checking the counts.** $N_s$ must be constant and $F_{s,j}$ must never
decrease. If either fails, inptk stops with an error instead of guessing,
because blank-corrected or merged counts are not independent freezing events.

**One state per temperature.** If the temperature holds or briefly warms,
inptk uses the latest observation at or warmer than each target temperature
(the "latest-warmer" rule). The original rows are kept in the experiment.

**Reporting interval.** Each input contributes only between its own first and
last observed freezing event, and concentrations are reported only inside that
interval. Outside it, the value and its errors are `NaN`. A $K(T)$ that
includes an unbounded step (Section 5) is also `NaN`.

**Output errors.** inptk reports `lower_error` $= \hat\theta - \theta_{\text{low}}$
and `upper_error` $= \theta_{\text{high}} - \hat\theta$. Points where the
estimate is exactly zero are marked `at_zero_boundary`.

**Numerical details of the fit.**

- Each column of the problem is divided by its total weight (`scale` in
  `_reduced`). This only changes units, so the optimizer sees numbers of
  similar size.
- If a trial step makes $\Delta_e\cdot x \le 0$, the log term would be
  infinite. Below $10^{-12}$ the code replaces it by its tangent line, so the
  function and gradient stay finite. A final solution must have
  $\Delta_e\cdot x > 0$ for every event, or the fit is rejected.
- The root search for $\mu$ (Section 8) uses Brent's method. If the optimum
  sits on a flat face, where two solutions give the same penalized value, the
  code interpolates between them until the unpenalized likelihood reaches the
  limit exactly.

**Optional fitting grid (`fit_step_C`, experimental).** With `fit_step_C` set,
the steps sit on a regular grid $t_1 > \dots > t_M$, anchored at the warmest
observation and ending exactly at the coldest one. $K$ is then linear between
grid points instead of a step function:

```math
K(T) = \sum_k \kappa_k\,\phi_k(T), \qquad
\phi_1 = 1, \qquad
\phi_k(T) = \operatorname{clip}\left(\frac{t_{k-1} - T}{t_{k-1} - t_k},\, 0,\, 1\right),\ k \ge 2 .
```

$\phi_k$ is non-decreasing on cooling and $\kappa_k \ge 0$, so $K$ stays
monotone and $\lambda$ stays linear in $x$; Sections 4 to 8 apply unchanged.
The counts are never interpolated: each observation enters at its own measured
temperature. This differs from `temperature_step_C`, which selects which
count observations to use before any estimation.

**Water-blank option names.** "Blank after first freeze" is
`water_blank_after_first_freeze=True` (off by default). "Blank temperature
limits" is `water_blank_temperature_range_C`.

## 14. Code map

| What | Where |
|---|---|
| Per-dilution histories, latest-warmer rule, dropping no-event points | `src/inptk/curve_fit.py` (`_trajectory`, `fit_curve`) |
| $V_s/d_s$ and $V_s$ | `sample_exposure`, `blank_exposure` |
| Step vectors $a_{s,j}$ (`state`), $c$ (`linear`), $\Delta_e$ (`events`), $n_e$ (`counts`), merged and unbounded columns | `CurveLikelihood.__init__`, `_reduced` in `src/inptk/_engine/curve_likelihood.py` |
| $\ell(x)$, gradient, L-BFGS-B fit | `_Likelihood.value_gradient`, `_Likelihood.fit` |
| Profile bounds | `_Likelihood.endpoint`, `CurveLikelihood.estimate` |
| $\delta = z^2/2$ | `fit_curve` (`z**2 / 2`); 95 % value `PROFILE_LIKELIHOOD_DROP_95` in `_engine/math.py` |
| Unit factors | `SampleMetadata.factor_for` |
| Reporting interval, output table | `src/inptk/estimation.py` (`estimate_concentration`) |
