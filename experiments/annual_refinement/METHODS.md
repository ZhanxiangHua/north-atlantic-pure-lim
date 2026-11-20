# How ocean maps enter the monthly wind

The emulator represents one North Atlantic calendar month at a time. Give it a global mean warming level $g$ in kelvin above 1850–1900. It samples a monthly ocean and atmosphere, calculates a wind mean consistent with that background, then generates a sequence of daily winds around that mean. A February model and a September model have different numerical coefficients. All twelve use the same set of equations.

## 1. Compress a training map

Take one monthly map, retain the ocean cells shared across the training data, and stack its numbers into a column $x$. Let $\bar x$ be the training mean of each component and $W$ a diagonal matrix of positive weights. A weight accounts for cell area and the typical scale of the variable. The mixed-layer depth $h$ enters as $\log(\max(h,0.01\,\mathrm{m})/1\,\mathrm{m})$. A difference of one unit in this coordinate means a multiplicative change in depth. SST and the subsurface temperature gradient use their original coordinates.

An EOF is a spatial pattern. Put the patterns into the columns of a matrix $U$ whose columns have length one and are mutually perpendicular. The two operations encode the map as a vector $z$ and reconstruct it from those coefficients.

$$
z = U^\top W(x-\bar x), \qquad
\widehat x=\bar x+W^{-1}Uz .
$$

The superscript $\top$ means transpose. Each entry of $z$ measures how much of one pattern is present. The product $UU^\top$ retains the part of the weighted map that the chosen patterns can express. Adding a pattern can improve reconstruction even if it explains little about tomorrow’s wind.

One tested method weights a mixed-layer log-depth residual by a fractional power of typical local depth. A small log-depth error $\delta$ near depth $h$ corresponds to roughly $h\delta$ metres. Weighting by $h^{0.7}$ partially accounts for that change of units. Another method keeps the published 18 SST and 18 joint-ocean patterns, projects training maps onto their perpendicular residual space, and learns extra patterns there. The latter construction preserves the original coarse coordinates exactly. The pressure and temperature/humidity blocks have separate coordinates.

## 2. Generate a monthly background

Stack the four blocks into one vector $z$. At warming $g$, the model draws

$$
z=m(g)+F(g)e, \qquad e\sim\mathcal N(0,I), \qquad C(g)=F(g)F(g)^\top .
$$

The vector $m(g)$ is the fitted mean, $e$ contains independent standard normal numbers, and $C(g)$ is the covariance between coordinates. Its diagonal entries are variances. Positive off-diagonal entries mean that two coordinates tend to move together. The fitted mean is a quadratic function of standardized warming, optionally shrunk toward the linear fit. The variance response uses a bounded temperature coordinate when it extrapolates.

In the published covariance law, $L$ is a lower-triangular factor of the pooled covariance and the columns of $V$ are orthonormal directions that matter to the monthly wind.

$$
F(g)=L\,[I+(\sqrt{s(g)}-1)VV^\top],\qquad
s(g)=a\exp[b(g_{\rm clip}-g_0)] .
$$

The quantity $a$ sets the variance level near the centre of the training temperatures, $b$ is a log-variance change per kelvin, and $g_0$ is the centring temperature. The matrix $VV^\top$ selects the active directions. Multiplying their amplitudes by $\sqrt{s}$ multiplies their variances by $s$.

In an ocean-only data experiment, 57 years train an ocean marginal and 30 years supply paired ocean and wind observations. The Cholesky factors $L_{o,0}$ and $L_{o,1}$ describe the paired and enlarged ocean covariances. Acting with $L_{o,1}L_{o,0}^{-1}$ on the ocean part of the original joint covariance installs the new ocean marginal while retaining a valid joint covariance. The separate 57-year fully paired experiment also uses the added daily wind and atmospheric fields.

## 3. Fit the background-to-wind relationship

Let $w_t$ contain the wind EOF coefficients on day $t$. The model predicts a monthly wind mean in those coordinates.

$$
\mu(g,z)=a_0+a_1(g-g_0)/s_g+K^\top[z-m(g)] .
$$

Here $a_0$ is an intercept vector, $a_1$ is a vector of forced wind responses, $s_g$ is the standard deviation of training warming, and $K$ maps background anomalies into wind anomalies. Training pairs each actual monthly background with that member’s mean wind. Ridge regression penalizes large coefficients. A separate penalty on the additional ocean modes lets the fine-scale maps improve without requiring every new mode to influence the wind.

At fixed warming the wind-mean covariance supplied by the background is

$$
C_{\mu}(g)=K^\top C(g)K .
$$

To obtain a number in $\mathrm{m^2\,s^{-2}}$, multiply by $G$ and take the trace. The trace is the sum of diagonal entries. The matrix $G$ converts wind EOF coordinates to the area-weighted energy of steering, which uses 20% upper-level and 80% lower-level wind. Cross-covariances between ocean and atmosphere can have either sign. The attribution tables keep these cross terms so that the block contributions sum to the full variance.

For a small example, let the monthly wind respond to two dimensionless scores with coefficients $k_1=0.6$ and $k_2=0.3$ m/s. At the cooler temperature let both variances be 1 and their covariance be 0.2. The wind variance is $0.6^2(1)+0.3^2(1)+2(0.6)(0.3)(0.2)=0.522$ m²/s². At the warmer temperature let the variances be 1.4 and 1.2 and their covariance 0.3. The same wind responses give $0.504+0.108+0.108=0.720$ m²/s². The factor of two counts both symmetric off-diagonal entries of the covariance matrix. These numbers illustrate the calculation. The archived field-block table applies it to the fitted North Atlantic model at each warming level. [Illustrated calculation](../../reports/annual_refinement/figures/covariance_to_wind.png)

One refinement fits spread inside directions unseen by the wind. A column $v$ qualifies when $K^\top Lv=0$ and $V^\top v=0$. Increasing variance along the physical direction $Lv$ adds a multiple of $(Lv)(Lv)^\top$ to the background covariance. Sandwich that change between $K^\top$ and $K$ and the result is zero. Training uses this freedom to fit ocean gridpoint variances and spatial differences. The implementation verifies the wind covariance identity at cold, central and warm temperatures before writing the fit.

## 4. Generate daily weather with a pure LIM

For a two-day autoregression, stack the most recent two wind-anomaly vectors into a state $u_t$. The daily transition $A$ propagates that state and the innovation covariance $Q$ supplies fresh Gaussian variation. Its stationary covariance $\Sigma$ satisfies

$$
u_t=A u_{t-1}+\eta_t,\quad \eta_t\sim\mathcal N(0,Q),
\qquad \Sigma=A\Sigma A^\top+Q .
$$

All transition eigenvalues must have magnitude below one. The model draws its pre-event history from the stationary law. A fitted retention $r$ replaces $A$ with $rA$ and adjusts the innovation covariance to preserve $\Sigma$. If $H$ extracts the current wind coordinate block, one common output filter is

$$
w_t=\mu(g,z)+d(g)\,k\left[Hu_t-
       \frac{b}{16}\sum_{j=5}^{20}Hu_{t-j}\right] .
$$

The coefficient $b$ controls feedback from recent history, $k$ normalizes the daily steering variance, and $d(g)$ controls weather amplitude at the requested warming. Every term is linear in a Gaussian state at fixed $g$. Deployment needs the monthly numerical weights and a random seed. Temperature is the only scenario input.

The experiments also test seasonal pooling of weather coefficients and separate gains for slow and shear-sensitive wind modes. A gain matrix $S$ can be absorbed into a new state with $T=\mathrm{diag}(S,S)$. The transformed matrices are $TAT^{-1}$, $TQT^\top$ and $T\Sigma T^\top$. These obey the same stationary equation and have the same transition eigenvalues. The validation tables determine whether a more elaborate fit earns its extra parameters.

## 5. Check each prediction on the quantity it is meant to reproduce

A map reconstruction RMSE asks how well observed EOF coordinates recover the observed field, in kelvin, metres or K per 100 m. The continuous ranked probability score (CRPS) asks whether distributions sampled using only warming place their mass near observed values. The calculation uses 20 equally spaced probability levels in physical units. Lower is better for both scores. The empirical coverage of a nominal 90% interval gives a separate calibration check. Spatial variogram scores compare observed and model-expected absolute differences at three grid separations.

For an averaging window of $d$ days, stationarity and the lag covariance $c_j$ of steering give its weather variance as

$$
V_W(d)=\frac{d c_0+2\sum_{j=1}^{d-1}(d-j)c_j}{d^2}.
$$

The factor $d-j$ counts the pairs of days separated by $j$ days. The full variance at fixed warming equals $V_W(d)+\mathrm{tr}[G C_{\mu}(g)]$. Across different warming levels, the forced mean adds another variance term. Within-month scores first subtract each realization’s own monthly mean from every grid point. The entire-month window then has zero within-month variance by construction.

The published and research fits see the same MPI cases. Uncertainty intervals resample whole ensemble members and whole calendar years with the same weights for both fits and all months. Storm tracks use the same starting disturbances and FAST settings. The 2050 SSP2-4.5 reference is a separate daily MPI scenario replay. A stronger ocean reconstruction, calibrated 15-day variance, a better forecast conditional on yesterday's winds, and a better track distribution are four distinct empirical results.

## 6. What can another calendar month teach an ocean EOF?

A warm August map and a cool September map can share a spatial pattern even when their climatologies differ. For each target month $m$, we preserve its original 18 large-scale coordinates in each ocean block. Let $R_m$ contain the weighted map residuals after projecting away those coordinates. Let $C_m=R_m^\top R_m/(n_m-1)$, where $n_m$ is the number of training maps. For each neighbor, subtract its own climatology before computing $C$. Rescale each neighbor covariance to the trace of $C_m$, giving $\tilde C_{m-1}$ and $\tilde C_{m+1}$. Trace is total weighted spatial variance. This scaling gives high-variance winter months the stated contribution.

$$
C_m^{\mathrm{pool}} = 0.70 C_m + 0.15\tilde C_{m-1} + 0.15\tilde C_{m+1}.
$$

January takes December and February as neighbors. The leading eigenvectors of this matrix give candidate fine-scale patterns. Each month's own data still determine how often its patterns occur at a given warming and how their scores relate to wind. The validation table compares pooled and single-month fits at the same rank and with the same weather filter.

The all-month 2050 SSP2-4.5 panel contains eight MPI members and one scenario year. Its uncertainty intervals can measure differences between members. They do not estimate variation across future scenario years. TC formation and landfall use every attempted disturbance as their denominator. A monthly intensity or displacement distribution requires at least one qualifying storm on each side of the comparison. The tables report how many resamples support each interval.

## 7. Give Fourier the same inputs, then test what each wind model can represent

Both weather generators receive the same conditional monthly wind and the same instantaneous four-component covariance. In the TC package, 15 harmonics share a 20-day period. Harmonic number $j$ has power proportional to $j^{-3}$. A separate training experiment estimates the fundamental period $P$ and power exponent $q$ once per calendar month. It minimizes the squared logarithmic error of the total and within-month training variances. The number of harmonics stays at 15. The daily wind test samples whole MPI members and years together for both generators. The TC track replay continues to use the package's native hourly generator.

Let $x_t$ be the four-component wind vector at one grid cell on day $t$. For lag $h$ we use $C_h=E[x_{t+h}x_t^\top]$. Rows refer to the future component and columns to the earlier one. The entry $C_h[1,3]$, using one-based indices and the order $(u_{250},v_{250},u_{850},v_{850})$, measures upper-level zonal wind after lower-level zonal wind. The reverse ordering measures a different time sequence. Their half-difference is one entry of

$$
D_h = \tfrac12(C_h-C_h^\top).
$$

For independent Fourier factors sharing a scalar lag function $r_h$, a constant loading matrix $B$ mixes factors into the four physical winds. Then $C_h=B(r_hI)B^\top=r_hBB^\top$. Its transpose is identical and $D_h$ is zero, whatever period or exponent the training selects. This prediction can be tested on actual MPI daily winds. A zero directional covariance is a stronger assumption than matching the variance of one area-averaged steering speed.

If the LIM state has transition $A$, stationary covariance $\Sigma$, and physical observation map $H$, its unfiltered covariance at positive lag is $H A^h\Sigma H^\top$. The negative lag uses its transpose. With memory-filter coefficients $a_i$, the emitted covariance is $\sum_{i,j}a_i a_j C_{h-i+j}$, multiplied by the weather amplitude squared. The code computes every entry before taking a regional area average.

The MPI calculation removes each realization's own monthly mean. For $n$ days, define $M=I-\mathbf1\mathbf1^\top/n$. This matrix applies the same demeaning to the model expectation. For each ordered pair $t+h,t$, the model weights the raw covariance at lag $i-j$ by $M_{t+h,i}M_{t,j}$, sums over $i,j$, then averages across all $n-h$ eligible pairs. We divide covariance entry $(a,b)$ by the product of the two observed MPI standard deviations. Errors in the resulting dimensionless correlations can be compared between levels and regions. The directional score is the RMS error across the six distinct component pairs.

A separate training-only ablation varies the importance of this matrix loss while refitting two existing weather parameters, retention and feedback. Only the training years enter the objective. The validation tables compare the resulting changes in short-lag wind structure and in 5–15 day and monthly variability. The reported primary checkpoint follows the previously fixed selection rule.

## 8. Measure the variability lost at an EOF truncation

A fitted basis can reconstruct a map well and still give prediction intervals that are too narrow. One validation experiment measures how much of the missing width lies outside the retained spatial patterns. Write $X$ for a matrix with one area-weighted, centered map per row and $U$ for the retained EOF columns, with $U^\top U=I$. The residual maps are $R=X-XUU^\top$. The calculation uses separate matrices for sea surface temperature and for the joint log-depth and subsurface-gradient block. It subtracts each training year’s mean residual from all member residuals for that year. These centered rows form $R_c$.

With $N$ member-years and $Y$ distinct years, the estimated within-year covariance is $S=R_c^\top R_c/(N-Y)$. Multiplying its leading orthonormal patterns by the square roots of their eigenvalues gives a matrix $A_r$ with $r$ columns. A draw $A_r e$, where $e$ has $r$ independent standard normal entries, is a coherent residual map. Each column is orthogonal to the previously retained EOFs. For a grid cell whose training weight is $w_i$, its additional variance in the unweighted coordinates is $\sum_{j=1}^{r} A_{ij}^{2}/w_i^2$. The experiment adds this quantity to the existing marginal variance and applies the same physical transformations and quantiles used for the main model. A separate arm fits a bounded log-linear warming response to each ocean block’s training residual energy. These calculations score ocean probabilities on validation members. The frozen wind and storm simulations use their archived primary checkpoint.

## 9. Hold the wind response fixed and vary the background covariance

Let $K_b$ be the rows of the published wind regression for source block $b$ and let $x_{ib}$ be member $i$’s background anomaly in that block. Within each climate year, subtract the member mean and apply $K_b$. The result is one wind contribution per member and source block. The inner product defined by the physical steering metric $G$ converts those wind coefficients into squared metres per squared second. Applying this inner product to every pair of source blocks gives a four-by-four matrix $D$ whose entries are $D_{bc}=\mathrm{tr}(G K_b^\top C_{bc}K_c)$. The sum of all entries is the monthly steering variance implied by the backgrounds. The validation calculation checks it against explicit steering maps computed from actual MPI backgrounds.

One model uses the same pooled $D_0$ at every temperature. A second fits one slope to the logarithm of total variance and multiplies $D_0$ by its positive, training-normalized scale. A third fits one slope per field block and scales the matrix on both sides by the diagonal matrix of square-root amplitudes. Another averages year-specific covariances in three warming groups and linearly interpolates neighboring matrices. A half-pooled version averages this estimate with $D_0$. Each construction keeps the covariance positive semidefinite. The comparison includes the published wind-active response, uses the same validation members for all fits and keeps the wind regression fixed for every training-data size.

## 10. Forecast from two real daily wind maps

Fix the monthly MPI background and its conditional wind mean. Subtract that mean from two preceding daily wind maps, giving score anomalies $y_t$ and $y_{t-1}$. To predict $h$ days ahead, the best linear least-squares estimate multiplies the stacked observations by the future-to-past covariance and the inverse of the past-to-past covariance. With $C_j=\operatorname{Cov}(y_{t+j},y_t)$ this is

$$
\widehat y_{t+h}=\begin{bmatrix}C_h&C_{h+1}\end{bmatrix}
\begin{bmatrix}C_0&C_1\\C_1^\top&C_0\end{bmatrix}^{-1}
\begin{bmatrix}y_t\\y_{t-1}\end{bmatrix}.
$$

For the LIM, each $C_j$ is a matrix. It includes the effect of the saved weather filter. For a Fourier process that gives every component the same spectral shape, $C_j=r_j B$, where $B$ is its shared spatial covariance. The spatial factor cancels, leaving two scalar coefficients $a,b$ obtained by solving a two-by-two system with diagonal $r_0$, off-diagonal $r_1$ and right-hand side $(r_h,r_{h+1})^\top$. Its predicted anomaly is $a y_t+b y_{t-1}$. The code solves both linear systems numerically. It adds the same monthly wind mean back to each prediction, reconstructs physical wind, and compares steering with the subsequent MPI maps at leads of 1, 5, 10 and 15 days. Every driver has the same observations at initialization. The original source member and climate year are the resampling units, so overlapping daily starts stay together. This conditional experiment supplies the actual monthly climate. Using it operationally would also require a prediction of that monthly climate.
## 11. Can neighboring calendar months improve an estimated warming response?

Suppose seven MPI members experience the same warming in a given year. They have seven different monthly background maps. We first send each map through the same fixed wind regression. Call the resulting wind coordinates $w_i$ for member $i$. The quantity of interest is the variance in wind that these backgrounds induce across members. The positive semidefinite matrix $G$ converts a pair of wind-coordinate vectors into the area-weighted physical steering inner product. Positive semidefinite means that the squared steering amplitude $w^\top G w$ is always at least zero. With $n$ members and their mean vector $\bar w$, the variance is

$$
v = \frac{1}{n-1}\sum_{i=1}^n(w_i-\bar w)^\top G(w_i-\bar w).
$$

We compute this quantity twice, once with all background coordinates and once with only the SST and joint ocean coordinates. Each value has units of $\mathrm{m^2\,s^{-2}}$. We hold the published spatial basis and wind regression fixed, and use 42 additional training years to study the warming response. In a split, fitting members and years are both disjoint from the members and years used to score the fitted slope.

For each month $m$, regress $\log v_{m,y}$ on the annual warming $g_y$ over the fitting years $y$. The ordinary least-squares slope $\hat b_m$ is the fractional change on a logarithmic scale per kelvin. Twelve separate slope estimates may be noisy. In a second fit, allow each month to borrow information from its neighbors. Stack the twelve slopes into a column $b$ and choose the column minimizing

$$
\sum_{m=1}^{12}(b_m-\hat b_m)^2+
\lambda\sum_{m=1}^{12}(b_m-b_{m-1})^2,\qquad b_0=b_{12}.
$$

The first sum keeps each estimate close to its own data. The second penalizes abrupt changes around the calendar, including December to January. Let $D$ be the matrix whose multiplication takes these twelve successive differences, and $I$ the $12\times12$ identity. Differentiating the two squared lengths gives the linear system $(I+\lambda D^\top D)b=\hat b$. Setting $\lambda=0$ retains every monthly estimate. A single annual slope gives all months the mean of their estimated slopes. Each month retains its own average level of variance under every method.

To turn a slope into a positive variance, use $\hat v_m(g)=a_m\exp[b_m(g-\bar g)]$. Here $\bar g$ is the mean fitting warming. Select $a_m$ so the average of $\hat v_m(g_y)$ across fitting years equals their measured mean variance. At extrapolation, the evaluation bounds the warming input to at most 1.25 K beyond the fitting interval at either end. The score is $|\log(\hat v/v)|$ averaged over five held years. The published table pairs every method on exactly the same years, members and months. This diagnostic concerns the component of the wind variance induced by the background. The full 15-day statistic also contains an independent contribution from weather persistence.
## 12. Give ocean fronts more influence when fitting a fixed number of EOFs

A front here is a grid location where the monthly training climatology changes rapidly over distance. For each field separately, take the difference between a wet cell and each adjacent wet cell and divide by the separation of their centers in hundreds of kilometres. Take the root mean square of the available differences, then rank the cell within that field's wet grid. Its percentile $q_j$ runs from zero to one. We call the upper fifth of these cells fronts. This definition applies to monthly mean SST, mixed-layer depth in metres, and subsurface gradient in K per 100 m. The fitting inputs for depth continue to use log metres.

Let $R$ hold the weighted map residuals left by the original 18 large-scale modes, with rows for training maps and columns for cells. Multiply column $j$ by $f_j=1+p q_j$, scaling the factors within each physical field to a weighted root mean square of one. Write $F=\operatorname{diag}(f_j)$. Take the leading right singular vectors $V$ of $RF$. Undo the extra weighting with $F^{-1}V$, remove their projections onto the original modes, and orthonormalize the remaining vectors. These are the additional EOFs in the ordinary deployment coordinates. The new basis still has 48 SST modes and 72 joint ocean modes. Setting $p=0$ reconstructs the ordinary fit on the validation maps.

The front-region score averages squared physical reconstruction errors over its wet cells and takes the square root separately for each member-month. A complementary score asks whether the contrast between two neighboring wet cells has the right magnitude. For an edge from $i$ to $j$ with distance $d_{ij}$ in hundreds of kilometres, the gradient error is $(e_i-e_j)/d_{ij}$, where $e$ is reconstructed minus actual field. Both zonal and meridional edges contribute to the physical gradient RMSE.
## 13. Check very deep mixed layers in physical metres

At a given cell and warming, the model gives log depth a mean $\mu$ and standard deviation $\sigma$. With a standard normal variable $Z$, its depth is $H=\exp(\mu+\sigma Z)$, subject to the 0.01 m floor. Its upper tail has no finite ceiling. To investigate the tail without changing the fitted covariance, the development trial compares this decoder with two local alternatives. The first uses $H'=\min(H,M)$, where $M$ is the greatest depth in that cell and calendar month's 57 training years. This maximum describes the sample of training climates. Bathymetry would require a separate seafloor-depth field.

The second alternative changes only the top five percent of the Gaussian coordinate. Let $t=\Phi^{-1}(0.95)$, where $\Phi$ is the standard normal cumulative distribution function, and set $T_\alpha(z)=z+(\alpha-1)\max(z-t,0)$. Above $t$ the slope is $\alpha$ and below it the slope is one. Since this map is monotone, the predicted depth at quantile $\tau$ is $\exp[\mu+\sigma T_\alpha(\Phi^{-1}(\tau))]$. The chance of exceeding a fixed depth $h$ is $1-\Phi[T_{1/\alpha}((\log h-\mu)/\sigma)]$. If a ceiling is also imposed at or below $h$, that exceedance probability is zero.

For an observed depth $h$ and predicted quantile $q_\tau$, define $d=h-q_\tau$. The quantile loss is $\max(\tau d,(\tau-1)d)$, in metres. For a threshold, compare the model's exceedance probability $p$ to the observed indicator $o$ using the Brier score $(p-o)^2$. These two checks distinguish an inaccurate local quantile from too many or too few very deep events over a region. The trial evaluates both on the existing validation members and keeps the wind law fixed.
## 14. Change the variance seen by wind while preserving the other random directions

At a fixed warming, write the random background anomaly as $F\varepsilon$, where $F F^\top=C$ and the entries of $\varepsilon$ have independent unit normal distributions. The wind anomaly contributed by this background is $K^\top F\varepsilon$. Let $Q$ have orthonormal columns spanning the columns of $F^\top K$ and let $P=QQ^\top$. Multiplying a standardized random vector by $P$ extracts the part that can affect wind. The remaining part, $(I-P)\varepsilon$, has zero response under $K^\top F$.

For a positive variance gain $s$, use

$$
F' = F[I+(\sqrt{s}-1)P].
$$

In a direction selected by $P$, the amplitude of the noise is multiplied by $\sqrt{s}$. In the orthogonal complement it is multiplied by one. Since $K^\top F P=K^\top F$, the wind contribution after the change is $\sqrt{s}K^\top F\varepsilon$. Its covariance is exactly $sK^\top C K$. The background covariance $F'F'^\top$ is positive semidefinite by construction. Physical ocean fields also depend on these directions, so any gain that helps wind must be checked against ocean probabilities.

For calendar month $m$ and fitting year $y$, measure $v_{m,y}$ by applying the actual wind regression to the 35 MPI backgrounds, then taking the area-weighted variance across members. Denote the model's own transmitted variance by $u_{m,y}$. Regress $\log(v_{m,y}/u_{m,y})$ on centered warming. The resulting slopes can be used directly or smoothed around the calendar as in Section 11. The gain is $s_m(g)=a_m\exp[b_m(g-\bar g)]$. Set $a_m$ so that $\sum_y s_m(g_y)u_{m,y}=\sum_y v_{m,y}$. This uses only the 57 fitting years. The separate validation measures compare both transmitted and full 15-day variance, and the probabilities of SST, mixed-layer depth and subsurface temperature gradient.

## 15. How much happens between the first and second wind lags?

Let $x_t$ be the 120 wind EOF coefficients after removing the conditional monthly wind mean. The selected model carries two consecutive latent weather states, $s_t=(x_t^\top,x_{t-1}^\top)^\top$. Its transition is $s_{t+1}=A s_t+\eta_t$. The random innovation $\eta_t$ has mean zero and is independent of $s_t$. Write $\Sigma=\mathbb E[s_t s_t^\top]$ for the stationary covariance and $H=[I\;0]$ for the matrix that takes the current 120 coefficients out of the 240-coordinate state. The current and one-day-later covariances are

$$
C_0=H\Sigma H^\top,\qquad C_1=H A\Sigma H^\top=\mathbb E[x_{t+1}x_t^\top].
$$

Suppose we must predict the next latent day from the current day alone. Minimizing $\mathbb E[\|x_{t+1}-Bx_t\|^2]$ gives the normal equations $B C_0=C_1$. Solve this linear system for $B$. The new process $x_{t+1}=Bx_t+\epsilon_t$ has covariance $C_0$ when the independent innovation $\epsilon_t$ has covariance $Q_1=C_0-BC_0B^\top$. These formulas reproduce the original $C_0$ and $C_1$ before output filtering. At two days the smaller model gives $B^2 C_0$, while the original gives $H A^2\Sigma H^\top$. Agreement at one day leaves room for their longer memories to differ. We check numerical stability, positive semidefiniteness of $Q_1$ and the covariance identity for every month.

We keep the ocean and atmospheric background, its regression into monthly wind, and all 120 wind EOFs fixed. One reduced model inherits the original linear output filter; a second fits the filter to the original 57 training years with the same variance and autocorrelation loss. For a forecast both receive the same two observed, filtered daily wind maps. Even for a one-lag latent process, filtering can make the second observed map informative. For a 15-day variance check we instead integrate the unconditional lag covariances and add the predicted contribution from monthly backgrounds. The validation integrates all 12 months and compares the same seven members and six years across the three methods. Each forecast interval resamples whole members and whole years together across calendar months. Each cold- and warm-edge stratum includes one validation year, so its interval only measures sampling across members.

### Keeping selected information from the previous day

Write $u=x_{t-1}$, $v=x_t$ and $w=x_{t+1}$. Each is a 120-entry column vector in the original stationary wind process. First remove from $u$ and $w$ whatever is predictable from $v$ alone. For the past, set $K=\operatorname{Cov}(u,v)C_0^{-1}$ and $u'=u-Kv$. For the future make the same regression to get $w'$. Let $S_u$ and $S_w$ be their covariance matrices and let $D=\mathbb E[w'u'^\top]$. Whiten each residual by multiplying by its symmetric inverse covariance square root. For a positive matrix $S=U\Lambda U^\top$, this square root is $S^{-1/2}=U\Lambda^{-1/2}U^\top$.

The matrix $M=S_w^{-1/2}DS_u^{-1/2}$ now relates standardized future and previous days. Its singular value decomposition orders the pairs of directions by their conditional canonical correlation. Put the leading $r$ right singular vectors in the columns of $V_r$. Only $v$ and the $r$ standardized contrasts $V_r^\top S_u^{-1/2}(u-Kv)$ enter the reduced state $y_t$. Call the linear map taking $s_t$ to $y_t$ the matrix $T$. The state has $120+r$ entries. Its training covariance and first lag are $S_0=T\Sigma T^\top$ and $S_1=T A\Sigma T^\top$. Solve $B S_0=S_1$ for its transition and set $Q=S_0-BS_0B^\top$. Compute its forecasts and window statistics with the original physical wind EOFs and output filter.

At $r=0$ this gives the preceding single-day model. At $r=120$ the coordinate change is invertible and every lag covariance of the original observable winds is recovered. We verified both endpoints against the actual forecasts and 15-day variances for all twelve months. The development study fixed the allowed day-one forecast RMSE increase at 0.025 m/s, using the upper paired 95% bound, and the allowed increase in each 15-day absolute log variance error at 0.005. These are experimental tolerances specified before scoring the four intermediate ranks. They express a choice about how much accuracy a smaller state may trade away.

## 16. What does the simulated mixed layer do at the sea floor?

The physical background decoder returns a mixed-layer depth $H=\exp(\mu+\sigma Z)$ at an individual ocean grid cell, with a floor of 0.01 m. In this expression $Z$ is standard normal, $\mu$ is the model's mean log depth at that point, and $\sigma$ is the length of the row that maps independent Gaussian noise to its log depth. The separate bathymetry experiment uses the MPI model's own static sea-floor depth $D$ in metres. The source grid was checked against historical MPI mixed-layer-depth coordinates. Its wet vertices were remapped to the common scoring grid with exactly the same interpolation weights.

A bounded diagnostic reports $H_b=\min(H,D)$. Consequently its 99th percentile is $\min[\exp(\mu+\sigma\Phi^{-1}(0.99)),D]$ on the wet cells in this analysis. For $h<D$, the probability of exceeding $h$ agrees with the original decoder. For $h\ge D$, that probability is zero. If an unconstrained realization crosses the bottom, the bounded distribution assigns its probability to the bottom itself. The operation uses the local model depth and costs one comparison at each grid cell. It acts on the decoded mixed layer after sampling. The monthly wind regression still receives the identical background coefficients.

Compare predicted bottom crossings to the actual MPI frequency, including a count with 1 m tolerance for remapping precision. For an observed depth $h$ and predicted 99th percentile $q$, the loss in metres is $\max[0.99(h-q),-0.01(h-q)]$. We average over wet cells with latitude-area weights, then over complete member-months. The shelf subset includes basin cells with $D\le500$ m. The subpolar box uses 45–60°N and 65–20°W. The source record includes the CMIP6 tracking identifier, native-file hash, transformed-grid hash and license. [MPI bathymetry provenance](../../reports/annual_refinement/data/mpi_bathymetry_source.json)
