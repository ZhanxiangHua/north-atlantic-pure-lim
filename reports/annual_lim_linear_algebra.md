# The linear algebra of a month-conditioned North Atlantic weather emulator

Suppose the annual global temperature is 2 K above its historical baseline. You draw an August ocean and atmosphere, then ask how much the mean steering wind could vary over the next 15 days. You can answer this twice. Hold the ocean and atmosphere fixed to measure weather variability alone. Draw a new background as well to include uncertainty about which August you get. In the released model the two answers are **2.068** and **3.476** in squared metres per squared second. This note derives both numbers from the stored matrices.

An empirical orthogonal function (EOF) is a spatial pattern. Its principal component (PC) is the scalar coefficient telling us how much of that pattern a field contains. The linear inverse model (LIM) evolves a vector of wind coefficients from one day to the next. In this project, “pure LIM” describes the daily weather generator. The monthly ocean and atmosphere still have their own warming-conditioned distribution. We use EOFs on both layers. [Worked English notebook](../notebooks/Annual_Pure_LIM_Tutorial.ipynb)

## Where the ESD paper enters

Geogdzhayev, Souza, Flierl and Ferrari compress climate maps, fit the distribution of their coefficients as a function of global temperature, and reconstruct fields in physical space. Their published application computes a separate basis for each variable from one historical ensemble member, shares that basis across months, and keeps 1,000 modes per variable. They fit the statistics separately by calendar month. In their covariance model the factor is affine in temperature, $L_0+T_g L_1$. The paper describes the basis construction in section 3 and the covariance fit in section 4.1. [ESD paper](https://esd.copernicus.org/articles/17/235/2026/) · [Authors’ archived EarthCovar code](https://doi.org/10.5281/zenodo.18391766)

Our implementation adopts the reduced-coordinate idea. For each of the twelve months it fits **five new bases** on regional MPI-ESM1-2-LR training data. Four describe monthly background fields. The fifth describes the four daily wind components together. All 60 bases are in the release, in twelve `representation.npz` files. They are estimated by the project's NumPy/SciPy code. The published EarthCovar basis arrays and its Julia/JAX training software are not runtime inputs. We also use a different covariance rule that estimates the warming response of background directions that contribute most to steering wind. [Monthly basis fitting](../src/annual_lim/train.py) · [EOF implementation](../src/tc_emulator/representation.py) · [Deployed covariance rule](../src/annual_lim/model.py)

## Read a map as a vector

Fix a calendar month $m$. Index a member-year by $i$, a calendar year by $y$, and a day within a member-month by $t$. A training August has $N=35\times15=525$ monthly backgrounds and $525\times31=16{,}275$ daily wind maps. All quantities below have their own fitted values for each month.

| Symbol | Shape or units | Meaning |
|---|---|---|
| $f,\bar f$ | $q\times1$ | One field block, flattened over valid channels and cells, and its training mean |
| $D,U,c$ | $q\times q,\ q\times j,\ j\times1$ | Diagonal feature weights, retained EOF basis, and coefficients of one block |
| $z,w_t$ | $p\times1,\ k\times1$ | Concatenated background PCs with $p=78$, and daily wind PCs with $k=120$ |
| $g,\tilde g,g_*$ | K, dimensionless, K | Input warming, standardized warming for mean regression, and capped warming for variance |
| $F,K$ | $p\times p,\ p\times k$ | Background covariance factor and background-to-wind regression coefficients |
| $X_t,A,Q,\Sigma$ | $2k\times1$ and three $2k\times2k$ matrices | Base two-day state, transition, innovation covariance and stationary covariance |
| $G,h,d$ | $k\times k$, days, days | Steering-wind metric, time lag and averaging-window length |

A superscript $\mathsf T$ transposes a matrix. The product $u^\mathsf T v$ is the sum of the pairwise products of the entries of two columns. The outer product $vv^\mathsf T$ is a matrix whose $(a,b)$ entry is $v_a v_b$. Averaging centered outer products produces a covariance matrix. Its diagonal contains variances, its off-diagonal entries describe joint variation. The trace $\operatorname{tr}(C)$ adds the diagonal entries. We write samples as columns here. NumPy stores a batch of them as rows, so the formula $c=U^\mathsf T x$ becomes `X @ U` for a batch `X`.

The forcing scalar $g_y$ is computed from global monthly surface air temperature. Let $T_{iym}$ be the global area mean for member $i$, year $y$ and month $m$, and $d_{ym}$ the number of calendar days. Define

$$
T_{iy}=\frac{\sum_m d_{ym}T_{iym}}{\sum_m d_{ym}},
\qquad \bar T_y=\frac{1}{35}\sum_i T_{iy},
\qquad g_y=\bar T_y-\frac{1}{51}\sum_{s=1850}^{1900}\bar T_s .
$$

The baseline is 13.50738 °C. The 35 fitting members define the forcing series for training and evaluation. For inference a user can supply $g$ directly. The background and daily wind field statistics still depend on the requested month. [Forcing reconstruction](../src/annual_lim/forcing.py) · [Input fields, grids and calendar](../src/annual_lim/prepare.py)

## Derive the spatial coordinates

### Weight the grid before choosing directions

Within each block, $a_\ell=\cos(\phi_\ell)$ is proportional to the area of grid cell $\ell$ at latitude $\phi_\ell$. A *channel* $c$ is one field or one pressure level of a field. Let $\mathcal V_c$ be the cells with finite data in at least 95% of the training samples, and let $\sigma_c$ be the square root of the area-weighted mean of the local training variances on those cells. Let $v_c$ be the channel weight. The four winds each get $1/4$. Temperature and humidity profiles each get half of their block, with level weights proportional to the magnitude of the pressure-coordinate gradient.

For each valid channel-cell pair, form the diagonal entry

$$
D_{c\ell,c\ell}=\frac{1}{\sigma_c}
  \sqrt{\frac{a_\ell v_c}{\sum_{j\in\mathcal V_c}a_j}} .
$$

The implementation floors $\sigma_c$ at $10^{-8}$. The square root appears because a squared length multiplies a weight twice. Thus $\|D(f-\bar f)\|^2$ is a sum of area-weighted, scale-adjusted squared anomalies. This lets a multilevel field or a variable with large numerical units share a block with another field. Before fitting, mixed-layer depth uses $\log(\max(h,0.01\ {\rm m}))$ in the numerical metre convention, and specific humidity uses $\log(\max(q_v,10^{-10}))$ in kg/kg. Here $q_v$ denotes humidity. A missing value at a retained cell contributes zero anomaly for that sample. Excluded cells are restored as missing.

### Why the leading eigenvectors minimize reconstruction error

Set $x_i=D(f_i-\bar f)$. Stack the rows $x_i^\mathsf T$ in a matrix $X$ of size $n\times q$, where $n$ is the number of maps in this block. Consider a unit column $u$, so $u^\mathsf T u=1$. The closest point to $x_i$ along that line is $u(u^\mathsf T x_i)$. Its squared residual is $\|x_i\|^2-(u^\mathsf T x_i)^2$. Summing over maps gives

$$
\sum_i \|x_i-uu^\mathsf T x_i\|^2
=\|X\|_F^2-u^\mathsf T X^\mathsf T X u .
$$

The Frobenius norm $\|X\|_F$ squares and sums every matrix entry before taking the square root. Maximizing the last term over unit vectors selects the leading eigenvector of $X^\mathsf T X$. For $j$ orthonormal directions the retained term is $\operatorname{tr}(U^\mathsf T X^\mathsf T X U)$, maximized by the first $j$ eigenvectors. Equivalently these are the leading right singular vectors of $X$. The sum of their eigenvalues divided by the full sum is the fraction of weighted training variance represented.

Encoding, reconstruction and the physical-space anomaly basis $J$ now follow directly.

$$
c=U^\mathsf T D(f-\bar f),\qquad
J=D^{-1}U,\qquad \hat f=\bar f+Jc .
$$

$UU^\mathsf T$ is an orthogonal projection in the weighted coordinates. In physical coordinates the projection is $D^{-1}UU^\mathsf T D$. If $u$ is an EOF, replacing $u$ by $-u$ and its coefficient by its negative gives exactly the same map.

For the background, $n=525$ is much smaller than the number of features in large blocks. If $XX^\mathsf T v=s^2v$ and $s>0$, multiplying by $X^\mathsf T$ shows that $X^\mathsf T v/s$ is a unit spatial eigenvector when $v$ has unit length. This identity makes the smaller sample-space eigensolve practical. For daily winds the implementation approximates the leading spatial eigenspace with three block subspace iterations, followed by a small symmetric eigensolve. The reported retained wind fraction is computed for the fitted subspace. [Training scales, eigensolvers and reconstruction](../src/tc_emulator/representation.py)

### Two cells you can calculate by hand

Let $\bar f=(25,26)^\mathsf T$ °C, $f=(27,26)^\mathsf T$ °C, $D=I_2/\sqrt2$ and

$$
U=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}.
$$

The two columns describe a common change and a contrast between the cells. The weighted anomaly is $(\sqrt2,0)^\mathsf T$ and $c=(1,1)^\mathsf T$. Keeping both columns reproduces $(27,26)^\mathsf T$. Keeping only the first gives $(26,27)^\mathsf T$. Both cells then inherit a common +1 °C anomaly. This is what spatial truncation does to a local feature.

For the actual August weights, the following dimensions and explained fractions come from the fitted files. A feature is one valid channel-cell pair. The wind block has $4\times36\times58=8{,}352$ features. The 38 profile channels cover temperature and humidity on 19 pressure levels each.

| Block | Valid August features | Retained columns | August weighted training variance retained |
|---|---:|---:|---:|
| Sea surface temperature | 1,479 | 18 | 88.84% |
| Mixed-layer depth and subsurface temperature gradient | 2,869 | 18 | 56.02% |
| Sea-level pressure | 2,088 | 14 | 91.41% |
| Air temperature and humidity profiles | 79,344 | 28 | 88.88% |
| Daily $u_{250},v_{250},u_{850},v_{850}$ | 8,352 | 120 | 86.36% |

The first four coefficient vectors concatenate to give $z$. The wind vector $w_t$ has its own basis $U_w$ and physical anomaly map $J_w$. An 86% retained wind fraction describes a spatial approximation to training maps. Accuracy of a 15-day average also depends on temporal covariances.

## Fit the monthly background and its contribution to wind

### A conditional mean and a covariance for member spread

Let $g_0$ and $s_g$ be the mean and standard deviation of warming over the fitting records. Define $\tilde g=(g-g_0)/s_g$ and $\phi(g)=(1,\tilde g,\tilde g^2)^\mathsf T$. With $\Phi$ containing one $\phi(g_i)^\mathsf T$ per row and $Z$ containing $z_i^\mathsf T$, estimate the $3\times p$ matrix $B$ that minimizes $\|\Phi B-Z\|_F^2$. Differentiating gives the normal equations $\Phi^\mathsf T\Phi B=\Phi^\mathsf T Z$. The code uses a least-squares solver. The expected background in PC space is $m_z(g)=B^\mathsf T\phi(g)$.

Two ensemble members in the same year share the same $g$ but experience different weather. To estimate their spread, center their PCs at that year's member mean $\bar z_y$. If $n_y$ members contribute then

$$
C_y=\frac1{n_y-1}\sum_{i\in y}(z_i-\bar z_y)(z_i-\bar z_y)^\mathsf T,\qquad
C_{\rm pool}=\sum_y\omega_y C_y,\quad
\omega_y=\frac{n_y-1}{\sum_{s}(n_s-1)} .
$$

A single year's $C_y$ has rank at most 34 because it is estimated from 35 centered members in 78 dimensions. Pooling years and shrinking the off-diagonal terms gives the reference $C_0=0.95C_{\rm pool}+0.05\operatorname{diag}(C_{\rm pool})$. Here $\operatorname{diag}$ builds the diagonal matrix from the diagonal of its argument. Compute a Cholesky factor $L$ satisfying $LL^\mathsf T=C_0$. For a vector $e$ of independent standard normal entries, $\mathbb E[Lee^\mathsf TL^\mathsf T]=LL^\mathsf T$. This is why multiplying noise by a matrix can create the observed spatial correlations. [Pooled covariance and conditional regression](../src/tc_emulator/warming_circulation.py)

### Regress the conditional mean daily wind

For fitting member-month $i$, the target is the mean of its daily wind PCs, $\bar w_i=T_i^{-1}\sum_t w_{it}$, where $T_i$ is its actual number of days. Write $\delta z_i=z_i-m_z(g_i)$ and let $S_z$ be diagonal with the training standard deviations of the columns of $\delta z$ (floor $10^{-8}$). The $i$th design row is $[1,\tilde g_i,(S_z^{-1}\delta z_i)^\mathsf T]$. Stack the rows in $H$, and write $Y$ for the matrix of targets. The fitted $(p+2)\times k$ coefficient matrix $\Theta$ solves

$$
(H^\mathsf T H+0.1N R)\Theta=H^\mathsf T Y,\qquad
R=\operatorname{diag}(0,1,\ldots,1).
$$

This follows by differentiating $\|H\Theta-Y\|_F^2+0.1N\|R\Theta\|_F^2$, using $R^\mathsf T R=R$. The intercept is unpenalized. Let $a_0,a_1$ be the first two coefficient rows, written as columns, and let $K=S_z^{-1}\Theta_{\rm bg}$, where $\Theta_{\rm bg}$ contains the remaining $p$ rows. For a supplied background,

$$
\mu(g,z)=a_0+a_1\tilde g+K^\mathsf T[z-m_z(g)].
$$

This is the expected wind PC on any day of the event. A finite simulated month's average includes residual weather and can differ from $\mu$. [Circulation design and prediction](../src/tc_emulator/background.py)

### Choose the background directions whose spread responds to warming

The steering field at each cell has components $0.2u_{250}+0.8u_{850}$ and $0.2v_{250}+0.8v_{850}$. Let $S$ be the linear operator that takes four physical wind maps to these two maps. Let $a_\ell^{\rm reg}$ be normalized cosine-latitude weights on the scoring region, 260–360°E and 0–60°N. Let $W$ repeat $a_\ell^{\rm reg}/2$ for the two steering components. A wind PC anomaly $v$ then satisfies

$$
G=J_w^\mathsf T S^\mathsf T W S J_w,\qquad
v^\mathsf T Gv=\|W^{1/2}SJ_wv\|^2 .
$$

The right-hand side squares the local wind at each cell before area averaging. It retains spatial variance that would cancel in a single basin-mean wind. $G$ is positive semidefinite since its quadratic form is a squared length. Its physical output is in $\mathrm{m^2\,s^{-2}}$.

A unit vector $u$ in the whitened background coordinates produces background anomaly $Lu$, then wind anomaly $K^\mathsf T Lu$. Its squared steering effect is $u^\mathsf T M u$ where $M=L^\mathsf T K G K^\mathsf T L$. Diagonalize this $p\times p$ symmetric matrix. Sort its eigenvalues $\lambda_j$ from largest to smallest and take the fewest eigenvectors accounting for at least 80% of their sum. Call the matrix of those columns $V$ and set $P=VV^\mathsf T$. In August seven directions account for 82.38% of this **background-induced steering variance**. That percentage has a different denominator from the wind EOF fraction in the preceding table.

To estimate how these directions respond to warming, form $\tilde C_y=L^{-1}C_yL^{-\mathsf T}$. If $v_j$ is column $j$ of $V$, compute $q_y=\sum_{j\ {\rm retained}}\lambda_j v_j^\mathsf T\tilde C_yv_j$. Define $x_y=g_y-\sum_s\omega_sg_s$ and $E_V=\sum_{j\ {\rm retained}}\lambda_j$. The fitted log-variance slope $b_g$ and positive scale $a_g$ are

$$
b_g=\frac{\sum_y\omega_y x_y\log q_y}{\sum_y\omega_y x_y^2},
\qquad
a_g=\frac{\sum_y\omega_y q_y}{E_V\sum_y\omega_y\exp(b_g x_y)} .
$$

Centering $x_y$ gives the usual slope with a free intercept. The second expression matches the *average projected variance in physical units*. In this balanced design the weighted year center equals $g_0$. For sampling, clip warming to $g_*=\operatorname{clip}(g,g_{\min}-1.25,\ g_{\max}+1.25)$ in K and put $s_b(g)=a_g\exp[b_g(g_*-g_0)]$. The fitted mean regressions continue to use the actual input $g$.

The sampled coordinates and their covariance are

$$
F(g)=L[I_p+(\sqrt{s_b(g)}-1)P],\qquad
z=m_z(g)+F(g)e,\qquad C_z(g)=F(g)F(g)^\mathsf T .
$$

To see why this works, use $P^\mathsf T=P$ and $P^2=P$. Squaring the bracket gives $I_p+(s_b-1)P$. The active directions therefore receive exactly the prescribed variance factor, and the orthogonal complement keeps its reference variance. The factor $F$ need not itself be triangular. Pushing it through the wind regression gives the **between-background** wind-PC covariance $K^\mathsf T C_z K$. [Selection of directions and temperature fit](../src/tc_emulator/warming_circulation.py) · [Sampling factor](../src/annual_lim/model.py)

For a two-coordinate check, take $L=\left(\begin{smallmatrix}2&0\\0.5&1\end{smallmatrix}\right)$ and one wind response $k=(0.6,0.8)^\mathsf T$. Then $LL^\mathsf T=\left(\begin{smallmatrix}4&1\\1&1.25\end{smallmatrix}\right)$ and $k^\mathsf TLL^\mathsf T k=3.20$. Equivalently $L^\mathsf Tk=(1.6,0.8)^\mathsf T$ and $1.6^2+0.8^2=3.20$. If only the first white-noise direction gets four times the variance, the same calculation gives $4(1.6)^2+(0.8)^2=10.88$. A change in the spread of the background can therefore change the wind variance even with fixed regression coefficients.

## Learn the sequence of daily wind anomalies

### From a regression on two days to a transition matrix

Subtract each member-month's fitted conditional wind from each of its observed daily PCs. Write $x_{it}^{\rm obs}=w_{it}-\mu(g_i,z_i)$. A two-lag regression predicts today's anomaly from yesterday and the day before. Define a stacked predictor $u_{it}=[(x_{i,t-1}^{\rm obs})^\mathsf T,(x_{i,t-2}^{\rm obs})^\mathsf T]^\mathsf T$ and target $v_{it}=x_{it}^{\rm obs}$. Both days and the target must belong to the same member, year and month. August contributes $525(31-2)=15{,}225$ triples. February contributes 13,895 because fitting includes both 28- and 29-day months.

Let $n_{\rm tr}$ count these triples. Define $C_{uu}=n_{\rm tr}^{-1}\sum uu^\mathsf T$ and $C_{vu}=n_{\rm tr}^{-1}\sum vu^\mathsf T$, summing over the retained triples. A coefficient matrix $M_w$ minimizes $n_{\rm tr}^{-1}\sum\|v-M_wu\|^2+\lambda\|M_w\|_F^2$. Setting its derivative to zero gives

$$
M_w(C_{uu}+\lambda I_{2k})=C_{vu},\qquad
M_w=[A_1\;\;A_2].
$$

$A_1,A_2$ are $k\times k$. The implementation solves the linear equations and tries penalty multipliers $0,0.001,0.01,0.1$ in that order, setting $\lambda$ to the multiplier times $\operatorname{tr}(C_{uu})/(2k)$. It uses the first fit with a valid stable numerical solution. Form the residual $\epsilon=v-M_wu$ and its empirical covariance $Q_x=n_{\rm tr}^{-1}\sum\epsilon\epsilon^\mathsf T$. The code obtains the same expression from accumulated matrix products, so it can combine months of different lengths without pairing across a boundary.

Stacking the current and previous *base* wind PCs gives the companion representation

$$
X_t=\begin{pmatrix}x_t\\x_{t-1}\end{pmatrix},\quad
A=\begin{pmatrix}A_1&A_2\\I_k&0\end{pmatrix},\quad
Q=\begin{pmatrix}Q_x&0\\0&0\end{pmatrix},\quad
H_x=[I_k\;\;0].
$$

In the base model $X_t=AX_{t-1}+\eta_t$ with independent $\eta_t\sim\mathcal N(0,Q)$ and $x_t=H_xX_t$. The first block predicts today's wind. The second copies yesterday's wind. A zero lower block in $Q$ is therefore expected. The model works directly with a discrete one-day transition. [Sufficient statistics and stable fitting](../src/tc_emulator/wind_lim.py) · [Calendar-month fitting loop](../src/annual_lim/train.py)

### Why a random first day can already be stationary

If yesterday has covariance $\Sigma$, and today's innovation is independent of yesterday, then today's covariance is $A\Sigma A^\mathsf T+Q$. Equating the two gives the discrete Lyapunov equation. When the spectral radius $\rho(A)$, the largest absolute eigenvalue, is below one, its stationary solution is

$$
\Sigma=A\Sigma A^\mathsf T+Q
=\sum_{j=0}^{\infty} A^j Q(A^j)^\mathsf T .
$$

The sum tracks the surviving contribution of innovations from each past day. In one dimension, $x_t=0.8x_{t-1}+\epsilon_t$ with innovation variance $1.44$ has stationary variance $1.44/(1-0.8^2)=4$. Starting with $x_0\sim\mathcal N(0,4)$ gives day one the same spread as later days. The released program solves the matrix equation, checks its relative residual, and samples the initial state with covariance $\Sigma$. For a positive semidefinite matrix $C=E\Lambda E^\mathsf T$, $E\sqrt{\Lambda}\,\xi$ has covariance $C$ if $\xi\sim\mathcal N(0,I)$. This construction also works with a singular innovation covariance. [Stationary initialization and NumPy/PyTorch propagation](../src/tc_emulator/wind_lim.py)

### Change the memory while preserving the base daily covariance

The base fit carries information about tomorrow. Matching the spread of a 15-day mean also requires the correct *shape* of the lag covariance. Let $r$ be a fitted retention coefficient and define

$$
A_r=rA,\qquad Q_r=r^2Q+(1-r^2)\Sigma,\qquad X_t=A_rX_{t-1}+\eta^r_t,
\qquad \eta^r_t\sim\mathcal N(0,Q_r).
$$

Substituting in the stationary equation gives $A_r\Sigma A_r^\mathsf T+Q_r=r^2(A\Sigma A^\mathsf T+Q)+(1-r^2)\Sigma=\Sigma$. In the scalar example, setting $r=0.75$ changes the effective daily coefficient to $0.6$ and the innovation variance to $2.56$. The stationary variance is still $4$, while the lag-one covariance falls from $3.2$ to $2.4$. In the damped 240-dimensional model, innovations can also enter the lower state block. After this adjustment that block is an auxiliary state. The observation remains $H_xX_t$.

Write $x_t=H_xX_t$ for this observed damped state. Let $b$ be the fitted feedback coefficient. Define taps $h_0=1$, $h_j=0$ for $j=1,\ldots,4$, and $h_j=-b/16$ for $j=5,\ldots,20$. Daily residual weather and total daily wind are

$$
u_t^{\rm weather}=s_w(g)\,\kappa\sum_{j=0}^{20}h_jx_{t-j},
\qquad
w_t=\mu(g,z)+u_t^{\rm weather},
\qquad
s_w(g)=\exp\!\left(\frac{\alpha_0+\alpha_1(g_*-g_0)}{2}\right).
$$

The factor of one half converts a log-*variance* slope into a multiplier of *standard deviation*. The model generates its own 20 days of stationary history before forming the first output. Every term in this filter is a linear transformation of Gaussian initial conditions and innovations. This is how the generator produces a new, correlated sequence without sampling a sequence of historical days. Physical fields obtained by exponentiating humidity or applying a thermodynamic solver can subsequently have non-Gaussian distributions. [Retention and innovation covariance](../src/tc_emulator/pure_lim.py) · [Causal output filter](../src/tc_emulator/pure_lim_memory.py)

## Derive the variance of a 15-day mean

### From lag matrices to a scalar with physical units

For lag $\ell\geq0$, iterating the stationary transition gives $\operatorname{Cov}(X_{t+\ell},X_t)=A_r^\ell\Sigma$. Future innovations are independent of $X_t$, which is why they disappear from this cross-covariance. Using the steering metric $G$ and the observation $H_x$, define the base and damped steering lag covariances

$$
c_\ell^0=\operatorname{tr}\!\left(G H_x A^\ell\Sigma H_x^\mathsf T\right),
\qquad c_\ell^{r}=r^\ell c_\ell^0 .
$$

The trace expresses the area-weighted covariance of the two steering components in physical units. It follows from $\mathbb E[v^\mathsf TGv]=\operatorname{tr}(G\,\mathbb E[vv^\mathsf T])$. A lagged matrix can be asymmetric. Reversing the lag transposes it, and the scalar trace is unchanged because $G$ is symmetric.

Multiply the two filtered outputs, expand both sums and take expectations. Before applying $s_w$ and $\kappa$, the filtered scalar covariance is

$$
\tilde c_\ell=\sum_{i=0}^{20}\sum_{j=0}^{20}h_i h_j\,c^r_{|\ell-i+j|},
\qquad
\kappa^2=\frac{c_0^0}{\tilde c_0},\qquad
\gamma_\ell(g)=s_w(g)^2\kappa^2\tilde c_\ell .
$$

Thus the taps redistribute persistence while $\kappa$ restores the reference daily *steering* variance before the warming amplitude is applied. Other wind components use the same fitted weather parameters. The code also checks their individual lag correlations. Up to lag 50 is needed for a 31-day statistic that includes a 20-day filter. [Lag traces and steering metric](../src/annual_lim/statistics.py) · [Filter covariance and window formula](../src/tc_emulator/pure_lim_memory.py)

### Count the pairs of days in the average

Let $\bar u_d=d^{-1}\sum_{t=1}^d u_t^{\rm weather}$. In the quadratic form $\mathbb E[\bar u_d^\mathsf T G\bar u_d]$, $d$ terms have lag zero. For a positive lag $\ell$ there are $d-\ell$ ordered pairs in each direction. Therefore the weather variance at a **fixed background** is

$$
V_{\rm weather}(d,g)=
\frac{d\,\gamma_0(g)+2\sum_{\ell=1}^{d-1}(d-\ell)\gamma_\ell(g)}{d^2}.
$$

For two days with variance 4 on each day and lag-one covariance 3.2, the answer is $(4+4+2\times3.2)/4=3.6$. If the two days were independent, the answer would be 2. This is why correcting the one-day variance alone cannot determine how well the model performs at 15 days.

For an ensemble of **new backgrounds at fixed warming**, the random conditional mean $\mu(g,z)$ is constant during the window. Its variability survives averaging. The law of total covariance gives

$$
V_{\rm total}(d,g)=V_{\rm weather}(d,g)
+\operatorname{tr}\!\left(G K^\mathsf T F(g)F(g)^\mathsf T K\right).
$$

One way to prove this is to write the deviation as the sum of a background anomaly and residual weather. Conditional on the background, the latter has mean zero, so the expected cross term vanishes. If a test cohort also mixes years with different $g$, average the fixed-$g$ variances over records and add the variance of their forced mean steering maps. A pooled validation ratio and the single-$g$ calculation below answer different questions. [Expected variances used in evaluation](../src/annual_lim/validate.py)

### Read these quantities from the trained August model

For $g=2$ K, the fitted August model has $\rho(A)=0.94713$, $r=0.84056$, $b=0.78533$ and $\kappa=0.95557$. Its fitted between-background steering variance is $1.40792\ \mathrm{m^2\,s^{-2}}$. Each row below uses those same weights. The Monte Carlo column is the executed notebook's average over 384 independent weather paths with the background fixed. All variance columns use $\mathrm{m^2\,s^{-2}}$.

| Window | Analytic weather | Sampled weather | Monte Carlo standard error | Analytic total, new backgrounds |
|---:|---:|---:|---:|---:|
| 1 day | 13.613 | 13.609 | 0.063 | 15.021 |
| 5 days | 6.292 | 6.293 | 0.051 | 7.700 |
| 10 days | 3.398 | 3.404 | 0.038 | 4.806 |
| 15 days | 2.068 | 2.070 | 0.026 | 3.476 |
| 31 days | 0.558 | 0.561 | 0.010 | 1.966 |

The background accounts for about 40.5% of the total 15-day variance and 71.6% at 31 days in this August example. The increase in its share arises because averaging suppresses a large part of the daily weather. The absolute background contribution is the same in both rows. [Executed notebook](../notebooks/Annual_Pure_LIM_Tutorial.ipynb)

The following calculation can be run from the full release or the original project with `PYTHONPATH=src`. It loads the saved spatial bases as well as the dynamics, since the physical steering metric depends on the wind EOFs.

~~~python
from pathlib import Path
import numpy as np
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics, trace_lags
from tc_emulator.warming_circulation import transfer_matrix

weights = next(d for d in (Path("weights"), Path("runs/annual_lim/weights"))
               if (d / "08/model.json").is_file())
model = MonthlyModel(weights, 8)
G = metrics(model.rep)[2][0]
K, F = transfer_matrix(model.mean), model.factor(2.0)
between = np.trace(G @ K.T @ F @ F.T @ K)
lags = trace_lags(model.base, G[None], 51)[0]
weather = model.cal.window_variance(2.0, lags, [1, 5, 10, 15, 31])[0]
print(between, weather[3], between + weather[3])
~~~

## What training matches and what validation measures

Once the EOFs, background model, wind regression and base transition are fixed, four parameters remain, $\alpha_0,\alpha_1,r,b$. For every training year $y$, let $V_{yd}^{\rm obs}$ be the MPI steering variance at window $d$. The five windows are 1, 5, 10, 15 and the actual calendar month length. Let $V_{yd}^{\rm fit}$ be the analytic weather variance plus the empirical variance of the conditional means of **that year's training members**. For each wind component $c$, let $\rho^{\rm obs}_{yc\ell}$ be its observed ocean-weighted lag correlation, and $\rho^{\rm fit}_{yc\ell}$ its model counterpart including the between-member contribution. The fitted loss is proportional to

$$
\sum_y\left[
\frac15\sum_d\left(\log\frac{V_{yd}^{\rm fit}}{V_{yd}^{\rm obs}}\right)^2+
\frac1{60}\sum_{c=1}^4\sum_{\ell=1}^{15}
(\rho^{\rm fit}_{yc\ell}-\rho^{\rm obs}_{yc\ell})^2
\right].
$$

Bounded least squares uses $\alpha_0\in[-0.7,0.7]$, $\alpha_1\in[-0.35,0.35]$ per K, $r\in[0.65,1]$ and $b\in[0,1.2]$. The logarithm treats reciprocal multiplicative errors symmetrically. At deployment and in held-out evaluation the background term comes from the fitted $F(g)$. Separating these two calculations matters when interpreting an extrapolation result.

Fitting used 35 members, validation used seven different members, and testing used eight other members at 15 held-out years. For each window a ratio of one means that generated and observed steering variance agree. The train-derived reference bands come from pairs of disjoint eight-member groups. Test uncertainty comes from resampling whole members and years. Across the pooled test years, the 15-day ratios run from 0.949 to 1.073, with all twelve months inside their reference bands and closer to one than independent days. In the warm extrapolation group, January and December have 15-day ratios of 1.191 and 1.187. These figures measure within-month weather statistics. The natural-year sampler independently initializes each month's background and weather history, so a path crossing a month boundary has a reset. [Monthly scores and temperature groups](annual_lim_results.md) · [Window statistics](../src/annual_lim/statistics.py)

## Reproduce one event from the saved matrices

1. Select the calendar month's directory and supply a year and $g$ in K relative to 1850–1900. Load the five spatial bases, feature masks, scales and means from `representation.npz`. Load the regression, covariance and transition matrices from `parameters.npz` and the four weather parameters and normalization from `model.json`.
2. Draw a standard normal background vector $e$, form $z=m_z(g)+F(g)e$, and compute $\mu(g,z)$. Decode the four background blocks. Undo the logarithms for mixed-layer depth and humidity, apply the physical bounds, and derive the thermodynamic fields from those maps.
3. Generate a stationary history and independent future innovations, propagate with $(A_r,Q_r)$, filter and scale the observed state, and add $\mu(g,z)$. Decode each $w_t$ with the wind EOFs. The random streams for the monthly draw, initial weather history and new weather innovations are separately keyed. Keeping the seed, year, month and background ID fixed lets you vary only the weather ID.
4. An event of $d$ integration days saves $d+3$ wind nodes, including padding for interpolation. The first actual day is at `t_seconds=0`. A sampled year discards padding and places winds on the actual 365 or 366 daily midpoints. A deployed model needs its fitted matrices, fixed thermodynamic resources and the warming input.

For a first experiment, examine two wind maps on the same day with different weather IDs and the same background ID. Then average 15 days of each realization. The first comparison shows the scale of daily weather. The second combines the spatial patterns from the EOFs with the persistence learned by the LIM. [Event and year sampler](../src/annual_lim/sample.py) · [Physical decoding and thermodynamics](../src/tc_emulator/physical.py)
