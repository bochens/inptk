# Technical note: the MLE method in inptk

This note explains how inptk estimates ice-nucleating particle (INP)
concentrations with `method="mle"`, which is the default. It gives the physical model,
the likelihood, how the fit is done, and how the uncertainty bounds are found.
Every formula here matches the code in `src/inptk/_engine/curve_likelihood.py`
and `src/inptk/curve_fit.py`. Section 9 checks the results against the classic
Vali formula.

## 1. The question

In a freezing assay, a set of $N$ droplets (or wells) of the same suspension is
cooled. At each observation we count how many have frozen. We want the
cumulative INP concentration $K(T)$: the number of INPs per mL of the original
suspension that are active at temperature $T$ or warmer.

The MLE method answers this with one fit per curve. It uses every freezing
history in the curve at once: all dilutions, and the water blanks. It finds the
single monotone curve $K(T)$ that makes the observed counts most probable.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $s$ | one *droplet set*: one measurement, run and cycle, with a fixed number of droplets |
| $N_s$ | number of droplets in set $s$ |
| $V_s$ | volume of each droplet, mL (inptk reads µL and divides by 1000) |
| $d_s$ | dilution factor of set $s$ (1 = undiluted) |
| $T_{s,0} > T_{s,1} > \dots > T_{s,J_s}$ | the temperatures kept for set $s$, warm to cold |
| $F_{s,j}$ | cumulative number frozen at $T_{s,j}$ |
| $n_{s,j} = F_{s,j} - F_{s,j-1}$ | droplets that froze between $T_{s,j-1}$ and $T_{s,j}$ (with $F_{s,-1} = 0$) |
| $K(T)$ | cumulative INP concentration of the sample, per mL suspension |
| $B_r(T)$ | cumulative background (water blank) concentration for run and cycle $r$, per mL |
| $\lambda_s(T)$ | expected number of active INPs in one droplet of set $s$ at $T$ |

## 3. Physical model

inptk uses the standard singular (time-independent) description:

1. Each INP has a fixed temperature at which it becomes active. A droplet
   freezes as soon as it contains at least one active INP.
2. INPs are randomly scattered through the liquid, so the number in one droplet
   follows a Poisson distribution.
3. Droplets freeze independently of each other.

If a droplet contains on average $\lambda$ INPs that are active at $T$, the
Poisson chance that it holds none is $e^{-\lambda}$. So

```math
P(\text{droplet still liquid at } T) = e^{-\lambda_s(T)} .
```

A droplet of set $s$ holds $V_s/d_s$ mL of the original suspension and $V_s$ mL
of liquid in total. The expected number of active INPs is

```math
\lambda_s(T) = V_s\left(\frac{K(T)}{d_s} + B_{r(s)}(T)\right).
```

Here $B_r$ is the background measured by the water blanks of the same run and
cycle. It scales with the whole droplet volume, and it does not depend on
dilution. A water-blank set has $d_s \to \infty$, so for it
$\lambda_s(T) = V_s B_r(T)$. In the code these two factors are
`sample_exposure` $= V_s/d_s$ and `blank_exposure` $= V_s$. If water-blank
correction is off, or no blank is mapped to the run, the $B_r$ term is absent.

## 4. Likelihood of one droplet set

Each droplet ends up in exactly one of these outcomes:

- it froze between two kept observations $T_{s,j-1}$ and $T_{s,j}$, or
  already by the first observation $T_{s,0}$;
- it was still liquid at the last observation $T_{s,J_s}$.

Write $\lambda_{s,j} = \lambda_s(T_{s,j})$ and $\lambda_{s,-1} = 0$. The
probability of freezing in the interval ending at $T_{s,j}$ is

```math
e^{-\lambda_{s,j-1}} - e^{-\lambda_{s,j}}
= e^{-\lambda_{s,j-1}}\left(1 - e^{-(\lambda_{s,j}-\lambda_{s,j-1})}\right),
```

and the probability of staying liquid to the end is $e^{-\lambda_{s,J_s}}$.
These outcomes are mutually exclusive and their counts add up to $N_s$, so the
counts follow a multinomial distribution. Up to a constant that does not
depend on $K$ or $B$, the log-likelihood is

```math
\log L_s = \sum_{j=0}^{J_s} n_{s,j}\left[-\lambda_{s,j-1} + \log\!\left(1 - e^{-(\lambda_{s,j}-\lambda_{s,j-1})}\right)\right]
 \;-\; (N_s - F_{s,J_s})\,\lambda_{s,J_s}.
```

**Why not one binomial per image?** The image at $-15\,^\circ$C and the image at
$-16\,^\circ$C look at the *same* droplets. A droplet frozen at $-15$ is still
frozen at $-16$. Treating each image as a new, independent binomial trial would
count each droplet many times and give intervals that are far too narrow. The
likelihood above counts each droplet once: by the interval in which it first froze.

Different droplet sets (other dilutions, other runs, blanks) are independent,
so the total log-likelihood is the sum $\log L = \sum_s \log L_s$.

## 5. A monotone curve as a sum of non-negative steps

$K(T)$ can only grow as the temperature drops. inptk enforces this by writing
the curve as a sum of non-negative increments on a temperature grid
$t_1 > t_2 > \dots > t_M$. By default, the grid is every temperature observed in the curve:

```math
K(T) = \sum_{k=1}^{M} \kappa_k\,\mathbf 1[t_k \ge T], \qquad \kappa_k \ge 0,
```

and in the same way $B_r(T) = \sum_k \beta_{r,k}\,\mathbf 1[t_k \ge T]$ with
$\beta_{r,k} \ge 0$. Put all increments into one vector
$x = (\kappa, \beta_{1}, \beta_{2}, \dots) \ge 0$. Then each $\lambda_{s,j}$
is a linear function of $x$:

```math
\lambda_{s,j} = a_{s,j}\cdot x ,
```

where $a_{s,j}$ holds $V_s/d_s$ in the $\kappa$ entries with $t_k \ge T_{s,j}$,
and $V_s$ in the matching $\beta_{r(s)}$ entries. In the code, $a_{s,j}$ is
`state`.

### The negative log-likelihood in matrix form

Substitute $\lambda = a\cdot x$ into Section 4 and collect terms. Each freezing
event $e$ (a $(s, j)$ with $n_{s,j} > 0$) has a step vector
$\Delta_e = a_{s,j} - a_{s,j-1}$ and a count $n_e = n_{s,j}$. The negative log-likelihood is

```math
\ell(x) = c\cdot x \;-\; \sum_{e} n_e \log\!\left(1 - e^{-\Delta_e\cdot x}\right),
```

with

```math
c = \sum_s\left[\sum_{j} n_{s,j}\,a_{s,j-1} + (N_s - F_{s,J_s})\,a_{s,J_s}\right].
```

Term by term, $c\cdot x$ is the expected number of active INPs that the
droplets "had to avoid" while they stayed liquid. The log term rewards the
droplets that froze in each interval. In the code, `linear` is $c$, the rows of
`events` are the $\Delta_e$, and `counts` holds the $n_e$.

The **maximum likelihood estimate** is

```math
\hat x = \arg\min_{x \ge 0} \ell(x), \qquad \hat K(T) = \sum_k \hat\kappa_k \mathbf 1[t_k \ge T].
```

## 6. The fit has one best answer

Let $g(u) = -\log(1 - e^{-u})$ for $u > 0$. Then

```math
g'(u) = -\frac{e^{-u}}{1-e^{-u}}, \qquad g''(u) = \frac{e^{-u}}{(1-e^{-u})^2} > 0 .
```

So $g$ is convex. $\ell$ is a linear term plus a sum of convex functions of
linear functions of $x$, so $\ell$ is convex. The constraint $x \ge 0$ is
convex too. Every local minimum is therefore a global minimum, and the
optimizer cannot get stuck in a wrong valley.

The gradient used by the optimizer is

```math
\nabla \ell(x) = c - \sum_e n_e\,\frac{e^{-\Delta_e\cdot x}}{1 - e^{-\Delta_e\cdot x}}\,\Delta_e .
```

inptk minimizes $\ell$ with L-BFGS-B, with bounds $x \ge 0$
(`_Likelihood.fit`). Two numerical details do not change the answer:

- **Rescaling.** Each column of the problem is divided by its total weight
  (`scale` in `_reduced`). This only changes units, so the optimizer sees
  numbers of similar size.
- **A safety floor.** If a trial step makes $\Delta_e\cdot x \le 0$, the log
  term would be infinite. The code replaces it below $10^{-12}$ by its tangent
  line, so the function and gradient stay finite and consistent. A final
  solution must have $\Delta_e\cdot x > 0$ for every event, or the fit is rejected.

## 7. Which parts of the curve the data can determine

**Columns that look the same.** Two increments $\kappa_k$ and $\kappa_{k'}$
can enter every $a_{s,j}$ in exactly the same way, for example two grid
temperatures with no observation of any set between them. The data cannot
tell them apart. inptk merges them into one column, fits the sum, and reports
the whole sum at the colder of the two temperatures (`_reduced` and the
loop at the end of `CurveLikelihood.__init__`). This means the cumulative curve rises at the last temperature
where the data could still place the INPs. Section 10 explains why this does
not shrink the uncertainty.

**Increments with no cost.** If a column of $c$ is zero, no droplet was
ever observed liquid after that step's temperature. Raising that increment
then costs nothing, and the likelihood keeps improving without bound as it grows. inptk
marks such columns as `unbounded`. Any $K(T)$ that includes them is reported
as `NaN`, not as a large finite number.

**Reporting interval.** Each input contributes only between its own first and
last observed freezing event, and concentrations are reported only inside that
interval. Outside it, the value and its errors are `NaN`.

## 8. Preparing the counts

Before the fit, each droplet set's history is cleaned up in ways that leave
the likelihood unchanged:

1. **A fixed set of droplets.** $N_s$ must be constant and $F_{s,j}$ must never
   decrease. If either fails, inptk stops with an error instead of guessing,
   because blank-corrected or merged counts are not independent freezing events.
2. **One state per temperature.** If the temperature holds or briefly warms,
   inptk uses the latest observation at or warmer than each target temperature
   (the "latest-warmer" rule). The original rows are kept in the experiment.
3. **Dropping observations where nothing froze.** Suppose no droplet of set
   $s$ froze between $T_{s,j}$ and $T_{s,m}$. Each of those intervals then
   adds only $F\,(\lambda_{s,i} - \lambda_{s,i-1})$ to the cost, with the same
   number $F$ frozen. The sum telescopes:

   ```math
   \sum_{i=j+1}^{m} (N_s - F)(\lambda_{s,i}-\lambda_{s,i-1}) = (N_s - F)(\lambda_{s,m} - \lambda_{s,j}).
   ```

   So the interior points can be removed with no change at all to $\ell$.
   inptk keeps both edges of every increase, plus the first and last
   observation (`_trajectory` in `curve_fit.py`).

## 9. Check: one droplet set gives the Vali formula

Take a single droplet set with no water blank, and let the grid be its own
observation temperatures. Write $u_j = \lambda_j - \lambda_{j-1} =
(V/d)\,\kappa_j \ge 0$. The negative log-likelihood separates into one term per
interval:

```math
\ell = \sum_j \Big[(N - F_j)\,u_j - n_j \log(1 - e^{-u_j})\Big]
```

(each droplet still liquid at $T_j$ pays $u_j$; each droplet that froze in
interval $j$ contributes $-\log(1-e^{-u_j})$). Setting the derivative to zero:

```math
(N - F_j) = n_j \frac{e^{-u_j}}{1 - e^{-u_j}}
\;\Longrightarrow\;
e^{-u_j} = \frac{N - F_j}{N - F_{j-1}} .
```

Adding the steps, $e^{-\lambda_j} = (N - F_j)/N = 1 - f_j$, where $f_j = F_j/N$ is the
frozen fraction. So

```math
\hat K(T_j) = \frac{d}{V}\,\bigl(-\ln(1 - f_j)\bigr),
```

which is exactly Vali's formula. The MLE reproduces the classic result when
there is only one droplet set, and it extends that result consistently to many sets.

**With a water blank of the same droplet volume**, the same reasoning gives
$V B(T_j) = -\ln(1 - f^{\text{blank}}_j)$, and

```math
\hat K(T_j) = \frac{d}{V}\Bigl[-\ln(1 - f_j) + \ln(1 - f^{\text{blank}}_j)\Bigr],
```

which is the usual blank subtraction done on the $-\ln(1-f)$ scale. This
holds as long as the result increases with cooling. When the blank would push
an increment below zero, the constraint $\kappa_k \ge 0$ becomes active, and
the joint fit finds the best monotone compromise instead.

**Numerical check.** For $N = 60$, $V = 0.05$ mL, $d = 10$, and
$F = 0, 3, 10, 25, 40, 48$ at $-5, -8, \dots, -20\,^\circ$C, inptk's MLE
matches $\frac{d}{V}(-\ln(1-f))$ to about seven significant figures (for
example 10.2587 and 321.888 per mL). With a blank of
$F^{\text{blank}} = 0, 0, 1, 3, 6, 12$ it matches the subtraction formula above
to the same precision.

## 10. Uncertainty: profile-likelihood bounds

For a temperature $T$, the target is a linear function of the increments:
$\theta = K(T) = w\cdot x$, where $w$ picks the $\kappa_k$ with $t_k \ge T$.

### Definition

The profile negative log-likelihood of $\theta$ is the best fit the model can
reach when $K(T)$ is held at $\theta$:

```math
\ell_p(\theta) = \min_{x \ge 0,\; w\cdot x = \theta} \ell(x).
```

The confidence interval is every $\theta$ whose profile is close to the best fit:

```math
\{\theta : \ell_p(\theta) - \ell(\hat x) \le \delta\}, \qquad \delta = \frac{z^2}{2}.
```

By Wilks' theorem, $2[\ell_p(\theta) - \ell(\hat x)]$ is approximately
$\chi^2_1$ distributed at the true $\theta$. With the default $z = 1.96$,
$\delta = 1.9207$, which gives approximately 95 % coverage. inptk reports
`lower_error` $= \hat\theta - \theta_{\text{low}}$ and `upper_error`
$= \theta_{\text{high}} - \hat\theta$. These errors are usually asymmetric,
which is expected for counting data.

The profile includes every other parameter: the rest of the sample curve
and the full background curves. The blank's uncertainty is part of the
bound, and it is counted once.

### How the endpoints are computed

The set $C = \{x \ge 0 : \ell(x) \le \ell(\hat x) + \delta\}$ is convex,
because $\ell$ is convex. The interval endpoints are the smallest and largest
values of the linear function $w\cdot x$ over $C$. A direct search with that
constraint is awkward, because the gradient of $\ell$ is zero at $\hat x$. inptk
uses the Lagrangian form instead. For a weight $\mu$, it solves

```math
x(\mu) = \arg\min_{x\ge 0}\; \ell(x) + \mu\,w\cdot x ,
```

which only changes the linear term $c \to c + \mu w$, so it is the same kind
of convex fit as before. $\mu > 0$ pushes $K(T)$ down, and $\mu < 0$ pushes it
up. The function $\mu \mapsto \ell(x(\mu))$ is monotone on each side of
zero. inptk finds the $\mu$ where $\ell(x(\mu)) = \ell(\hat x) + \delta$ with a
bracketing root search (Brent's method), and reads off $w\cdot x(\mu)$
(`_Likelihood.endpoint`). If the optimum sits on a flat face, where two solutions
give the same penalized value, it interpolates between them until the
unpenalized likelihood reaches the limit exactly.

Two edge cases have direct answers:

- **Lower bound at zero.** inptk first refits with every increment in $w$
  fixed at zero. If that fit is still within $\delta$ of the best, the lower
  bound is exactly 0.
- **Upper bound through a step with no events.** As $\mu$ becomes more
  negative, the cost of some column may reach zero (the "critical" weight). If no
  freezing event involves that column, the likelihood is linear in it. The
  remaining likelihood budget is then spent on it directly:
  $\Delta\kappa = (\text{remaining}) / c_k$.

**Merged columns are split again for the bounds.** When $T$ falls between
temperatures that were merged in Section 7, the merge is redone with $w$ as an
extra row, so the bound "knows" the INPs could sit on either side of $T$. A
fitted increment of zero therefore does not imply zero uncertainty
between two freezing events.

### Check against the binomial likelihood-ratio interval

For one droplet set without a blank, holding $\lambda_j$ fixed and optimizing
the other steps leaves a binomial likelihood for the $F_j$ frozen out of $N$.
The likelihood ratio does not depend on how the parameter is written, so the profile
interval for $K(T_j)$ is the binomial likelihood-ratio interval for $f_j$,
mapped through $K = \frac{d}{V}(-\ln(1-f))$. For the example in Section 9,
inptk gives lower/upper errors of 7.7077 / 16.348 per mL at $-8\,^\circ$C (3 of
60 frozen) and 89.295 / 115.03 at $-20\,^\circ$C (48 of 60). A separate
binomial likelihood-ratio calculation gives the same numbers to at least
five significant figures.

### What the bounds are, and are not

- They are **pointwise**. Each temperature has its own approximately 95 %
  interval. They are not a band that holds at all temperatures at once.
- Coverage is **approximate**. Wilks' theorem is a large-sample result, and
  it is less accurate when very few droplets freeze or nearly all freeze, and
  near the $K \ge 0$ boundary. Points where the estimate is exactly zero are
  marked `at_zero_boundary`.

## 11. Water-blank options

- **Separate backgrounds.** Each run and selected cycle gets its own background
  curve $B_r$. Water blanks are fitted together with the samples, so they
  inform $B_r$ and, through it, $K$.
- **`water_blank_after_first_freeze=True`.** $B_r(T)$ is fixed at 0 for every
  temperature warmer than that control's first freeze. A control that never
  freezes gets $B_r \equiv 0$. In the model, this sets the $\beta$ entries of
  $a_{s,j}$ to zero above the onset. This is an assumption the user chooses,
  and it is off by default.
- **`water_blank_temperature_range_C`.** This limits which blank observations
  enter the fit. It does not limit the samples.

## 12. Combining several inputs

A combined curve (for example, five dilutions of one sample) has one shared
$K(T)$. Each droplet set enters with its own $V_s/d_s$, and all sets add their
log-likelihoods. Nothing is averaged by hand. A dilution contributes most at
temperatures where its droplets are partly frozen, because there its freezing
counts carry the most information about $K$. A dilution that is fully frozen
or fully liquid at a temperature adds little there. The weighting comes
directly from the likelihood.

Only one cycle per run can enter a curve, because repeated cycles reuse the
same droplets and are not independent.

## 13. Optional fitting grid (`fit_step_C`, experimental)

With `fit_step_C` set, the increments sit on a regular grid
$t_1 > \dots > t_M$, anchored at the warmest observation and ending exactly at
the coldest one. $K$ is linear between grid points instead of a step function:

```math
K(T) = \sum_k \kappa_k\,\phi_k(T), \quad
\phi_1 = 1, \quad
\phi_k(T) = \operatorname{clip}\!\left(\frac{t_{k-1} - T}{t_{k-1} - t_k},\, 0,\, 1\right),\ k \ge 2 .
```

$\phi_k$ is still non-decreasing on cooling and $\kappa_k \ge 0$, so $K$ stays
monotone, and $\lambda$ is still linear in $x$. Sections 4 to 10 apply unchanged. The
observed counts are never interpolated: each observation enters at its own measured
temperature. This differs from `temperature_step_C`, which selects which
count observations to use before any estimation.

## 14. Units

The fit gives $K$ in INP per mL of the original suspension. Other bases
multiply by one constant factor (`SampleMetadata.factor_for`):

```math
K_{\text{air}} = K\,\frac{V_{\text{susp}}}{V_{\text{air}}\, f_{\text{filter}}}\ \ [\text{L}^{-1}],
\qquad
K_{\text{soil}} = K\,\frac{V_{\text{susp}}}{m_{\text{dry}}}\ \ [\text{g}^{-1}].
```

A profile-likelihood interval does not change under such a rescaling, so the
bounds are scaled by the same factor.

## 15. Assumptions in one place

1. Singular freezing: each INP has a fixed activation temperature, and the
   cooling rate does not matter.
2. INPs are Poisson-distributed among droplets, and droplets freeze independently.
3. The water background is proportional to droplet volume, does not depend on
   dilution, and is the same for all wells in one run and cycle.
4. Each droplet is counted once, at its first freezing. The droplet sets in
   one curve are physically different droplets.
5. Uncertainty comes from Wilks' theorem: approximate and pointwise.

## 16. Code map

| What | Where |
|---|---|
| Build droplet-set histories, latest-warmer rule, dropping no-event points | `src/inptk/curve_fit.py` (`_trajectory`, `fit_curve`) |
| Design vectors $a_{s,j}$, $c$, $\Delta_e$, merged and unbounded columns | `CurveLikelihood.__init__`, `_reduced` in `src/inptk/_engine/curve_likelihood.py` |
| $\ell(x)$, gradient, L-BFGS-B fit | `_Likelihood.value_gradient`, `_Likelihood.fit` |
| Profile bounds | `_Likelihood.endpoint`, `CurveLikelihood.estimate` |
| $\delta = z^2/2$ | `fit_curve` (`z**2 / 2`); 95 % value `PROFILE_LIKELIHOOD_DROP_95` in `_engine/math.py` |
| Reporting interval, output table | `src/inptk/estimation.py` (`estimate_concentration`) |
