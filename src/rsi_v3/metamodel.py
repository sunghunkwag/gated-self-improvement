"""
rsi_v3.metamodel -- the shared continuous contextual meta-predictor.

One ridge regressor shared by every improvement action (no per-action or
per-failure-mode table). A training row is ONE (candidate, probe) pair from
a counterfactual test:

    x = features(round context, candidate action, candidate deltas, probe)
    y = progress(candidate, probe) - progress(incumbent, probe)

with both searches run on the SAME stream, so y is a paired counterfactual
outcome. The model is refit in closed form after every round and ranks the
next round's proposals by predicted generalisation gain on cross-family
META-VALIDATION probes. Pure stdlib, deterministic.
"""
import math


class Ridge(object):
    def __init__(self, dim, lam):
        self.dim = dim
        self.lam = float(lam)
        self.A = [[(self.lam if i == j else 0.0) for j in range(dim)]
                  for i in range(dim)]
        self.b = [0.0] * dim
        self.w = [0.0] * dim
        self.n = 0
        self.frozen = False
        # meta-controller compute accounting (deterministic): the predictor
        # never executes a program; its cost is floating-point arithmetic
        self.ops = {"rows": 0, "refits": 0, "predictions": 0, "flops": 0}

    def add(self, x, y, wt=1.0):
        if self.frozen:
            raise AssertionError("frozen meta-predictor received data")
        A, b = self.A, self.b
        nz = [(i, v) for i, v in enumerate(x) if v != 0.0]
        for i, vi in nz:
            row = A[i]
            for j, vj in nz:
                row[j] += wt * vi * vj
            b[i] += wt * vi * y
        self.n += 1
        self.ops["rows"] += 1
        self.ops["flops"] += 2 * len(nz) * len(nz) + 2 * len(nz)

    def refit(self):
        if self.frozen:
            return
        L = cholesky(self.A)
        self.L = L
        self.w = chol_solve(L, self.b)
        d = self.dim
        self.ops["refits"] += 1
        self.ops["flops"] += d * d * d // 3 + 2 * d * d

    def predict(self, x):
        self.ops["predictions"] += 1
        self.ops["flops"] += 2 * self.dim
        return sum(wi * xi for wi, xi in zip(self.w, x))

    # ---- Bayesian view (used by the process controller): with prior
    # w ~ N(0, noise_var / lam) and per-row noise noise_var, A = lam I +
    # sum wt x x^T is the posterior precision / noise_var.
    def pred_sd(self, x, noise_var):
        """Posterior sd of x^T w (not of a new observation)."""
        L = getattr(self, "L", None)
        if L is None:
            L = cholesky(self.A)
            self.L = L
        z = forward(L, x)
        self.ops["flops"] += self.dim * self.dim
        return math.sqrt(noise_var * sum(v * v for v in z))

    def sample_w(self, noise_var, prng):
        """One Thompson draw w ~ N(w_hat, noise_var A^-1)."""
        L = getattr(self, "L", None)
        if L is None:
            L = cholesky(self.A)
            self.L = L
        xi = [gauss(prng) for _ in range(self.dim)]
        z = backward(L, xi)                   # L^T z = xi -> cov A^-1
        self.ops["flops"] += self.dim * self.dim
        sd = math.sqrt(noise_var)
        return [wi + sd * zi for wi, zi in zip(self.w, z)]


def gauss(prng):
    """Standard normal from a deterministic stream (Box-Muller)."""
    u1 = max(prng.unit(), 1e-300)
    u2 = prng.unit()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def forward(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        Li = L[i]
        for k in range(i):
            s -= Li[k] * y[k]
        y[i] = s / Li[i]
    return y


def backward(L, y):
    n = len(L)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i]
    return x


def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            Lj = L[j]
            s = A[i][j]
            for k in range(j):
                s -= Li[k] * Lj[k]
            if i == j:
                Li[i] = math.sqrt(max(s, 1e-12))
            else:
                Li[j] = s / Lj[j]
    return L


def chol_solve(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        Li = L[i]
        for k in range(i):
            s -= Li[k] * y[k]
        y[i] = s / Li[i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i]
    return x
