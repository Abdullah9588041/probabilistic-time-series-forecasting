# Mathematical notes

Derivations behind `tsforecast.statespace` and `tsforecast.evaluation`.
Notation: $y_{1:t} = (y_1, \dots, y_t)$.

## 1. State-space form of the structural model

With period-$s$ seasonality the state is
$x_t = [\mu_t, \beta_t, \gamma_t, \gamma_{t-1}, \dots, \gamma_{t-s+2}]^\top \in \mathbb{R}^{s+1}$:

$$
\begin{aligned}
y_t &= H x_t + \varepsilon_t, &
\varepsilon_t &\sim \mathcal{N}(0, \sigma_\varepsilon^2), \\
x_{t+1} &= F x_t + w_t, &
w_t &\sim \mathcal{N}(0, Q),
\end{aligned}
$$

$$
F = \begin{bmatrix}
1 & 1 &        &        &   \\
0 & 1 &        &        &   \\
0 & 0 & -1 & \cdots & -1 \\
0 & 0 &  1 &        &   \\
  &   &    & \ddots &   \\
0 & 0 &    &  1     & 0
\end{bmatrix},
\qquad
Q = \operatorname{diag}(\sigma_\ell^2, \sigma_b^2, \sigma_s^2, 0, \dots, 0),
\qquad
H = [1, 0, 1, 0, \dots, 0].
$$

The third row enforces the **sum-to-zero seasonal constraint**:
$\gamma_{t+1} = -\sum_{j=0}^{s-2}\gamma_{t-j} + \omega_t$, so the $s$ seasonal
effects always sum (in expectation) to zero and are identified. The remaining
rows are a shift register carrying past seasonals forward. Setting
$\sigma_b^2 = 0$ gives a random-walk trend; $\sigma_\ell^2 = \sigma_b^2 = 0$
gives a deterministic trend; all state variances $0$ collapses to linear
regression on seasonal dummies.

## 2. Kalman filter from Gaussian conditioning

Assume $x_t \mid y_{1:t-1} \sim \mathcal{N}(x_{t|t-1}, P_{t|t-1})$.
The joint distribution of $(x_t, y_t)$ given $y_{1:t-1}$ is Gaussian with

$$
\begin{bmatrix} x_t \\ y_t \end{bmatrix} \sim
\mathcal{N}\!\left(
\begin{bmatrix} x_{t|t-1} \\ H x_{t|t-1} \end{bmatrix},
\begin{bmatrix}
P_{t|t-1} & P_{t|t-1} H^\top \\
H P_{t|t-1} & H P_{t|t-1} H^\top + \sigma_\varepsilon^2
\end{bmatrix}
\right).
$$

For jointly Gaussian $(a, b)$, $a \mid b \sim \mathcal{N}(\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(b - \mu_b),\
\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})$. Applying this with
$a = x_t$, $b = y_t$ gives the **update**:

$$
\begin{aligned}
v_t &= y_t - H x_{t|t-1}, \qquad
S_t = H P_{t|t-1} H^\top + \sigma_\varepsilon^2, \\
K_t &= P_{t|t-1} H^\top S_t^{-1} \quad\text{(Kalman gain)}, \\
x_{t|t} &= x_{t|t-1} + K_t v_t, \\
P_{t|t} &= P_{t|t-1} - K_t H P_{t|t-1},
\end{aligned}
$$

and the **predict** step is the linear-Gaussian pushforward:

$$
x_{t+1|t} = F x_{t|t}, \qquad
P_{t+1|t} = F P_{t|t} F^\top + Q.
$$

The Kalman gain optimally trades off prior uncertainty ($P$) against
measurement noise ($R$): $K_t \to 1$ (in the scalar case) when the state is
uncertain relative to the observation, and $K_t \to 0$ when the observation
is pure noise.

## 3. Likelihood via prediction-error decomposition

Because $y_t \mid y_{1:t-1} \sim \mathcal{N}(H x_{t|t-1}, S_t)$,

$$
\log p(y_{1:n} \mid \theta)
= \sum_{t=1}^{n} \log \mathcal{N}(y_t; H x_{t|t-1}, S_t)
= -\frac{1}{2}\sum_{t=1}^{n}\left(\log 2\pi S_t + \frac{v_t^2}{S_t}\right),
$$

where $\theta = (\sigma_\varepsilon^2, \sigma_\ell^2, \sigma_b^2, \sigma_s^2)$.
We maximise this over $\log\theta \in \mathbb{R}^4$ (unconstrained) with
L-BFGS-B from several random restarts. The first $2s$ terms are excluded from
the sum so the diffuse initialisation ($P_0 = 10^6 I$) does not dominate —
a standard burn-in for exact diffuse initialisation.

## 4. RTS smoother (sketch)

The Rauch–Tung–Striebel smoother runs backwards:
$G_t = P_{t|t} F^\top P_{t+1|t}^{-1}$,
$x_{t|n} = x_{t|t} + G_t (x_{t+1|n} - x_{t+1|t})$.
It follows from the same Gaussian conditioning, now on the joint law of
$(x_t, x_{t+1})$ given all data. Smoothed states use future information, so
their MSE is no larger than the filter's — verified in
`tests/test_statespace.py::test_filter_recovers_true_level_and_smoother_improves`.

## 5. Multi-step predictive distribution

Iterating the predict step $h$ times from the terminal filter state,

$$
x_{T+h|T} = F^h x_{T|T}, \qquad
P_{T+h|T} = F^h P_{T|T}(F^h)^\top + \sum_{j=0}^{h-1} F^j Q (F^j)^\top,
$$

$$
y_{T+h} \mid y_{1:T} \sim \mathcal{N}\!\left(H x_{T+h|T},\; H P_{T+h|T} H^\top + \sigma_\varepsilon^2\right).
$$

Uncertainty grows with $h$ (the accumulated $Q$ terms) — the fan chart.
A central $(1-\alpha)$ prediction interval is
$\hat y_{T+h} \pm z_{1-\alpha/2}\,\hat\sigma_{T+h}$.

## 6. CRPS and why it is a proper scoring rule

For predictive CDF $F$ and outcome $y$,

$$
\operatorname{CRPS}(F, y) = \int_{-\infty}^{\infty} \bigl(F(z) - \mathbf{1}\{z \ge y\}\bigr)^2 dz
= \mathbb{E}_F|X - y| - \tfrac12 \mathbb{E}_F|X - X'|,
$$

with $X, X' \stackrel{iid}{\sim} F$. CRPS is **strictly proper**
(Gneiting & Raftery, 2007): $\mathbb{E}_{Y \sim G}[\operatorname{CRPS}(F, Y)]$
is uniquely minimised at $F = G$. Unlike MAE/RMSE, it rewards the *whole*
predictive distribution — a model with identical point forecasts but honest,
well-calibrated uncertainty scores better. For $F = \mathcal{N}(\mu, \sigma^2)$,

$$
\operatorname{CRPS} = \sigma\left[z(2\Phi(z) - 1) + 2\phi(z) - \frac{1}{\sqrt{\pi}}\right],
\quad z = \frac{y-\mu}{\sigma},
$$

which is what `evaluation.crps_gaussian` implements. As $\sigma \to 0$,
CRPS $\to |y - \mu|$ (MAE) — also covered by a unit test.

## 7. Interval calibration

A $(1-\alpha)$ prediction interval is *calibrated* if
$\mathbb{P}(y \in [l, u]) = 1 - \alpha$ over repeated forecasts.
We report empirical coverage at 80% and 95% nominal levels: systematic
under-coverage means overconfident (too narrow) intervals; over-coverage
means needlessly wide ones. Sharpness (mean width) breaks ties between
models with equal coverage — the classic calibration–sharpness trade-off.

## References

- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge.
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477).
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. (OTexts).
