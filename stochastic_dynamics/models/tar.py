import numpy as np

class TAR:
    """
    Threshold AutoRegressive (TAR) model with 2 regimes.
    """

    @staticmethod
    def lag_matrix_with_threshold(x, p, d=1):
        """
        Returns X (lags), y (target), and z (threshold variable x_{t-d}) aligned.
        """
        N = len(x)
        y = x[p:].copy()
        X = np.zeros((N-p, p))
        for i in range(p):
            X[:, i] = x[p-1-i : N-1-i]
        # threshold var aligned with y: z_t = x_{t-d}
        z = x[p - d : N - d]
        return X, y, z

    @staticmethod
    def ols_with_intercept(X, y):
        D = np.column_stack([np.ones(len(X)), X])
        w = np.linalg.lstsq(D, y, rcond=None)[0]
        return w  # [c, a1..ap]

    @staticmethod
    def predict_from_params(X, w):
        D = np.column_stack([np.ones(len(X)), X])
        return D @ w

    @staticmethod
    def aic_bic(y, yhat, k):
        n = len(y)
        rss = np.sum((y - yhat)**2)
        sigma2 = rss / max(n,1)
        bic = n*np.log(sigma2 + 1e-12) + k*np.log(max(n,2))
        return bic

    @staticmethod
    def fit(x, p=8, d=1, thresh=0.0, train_frac=0.8):
        X, y, z = TAR.lag_matrix_with_threshold(x, p, d=d)
        N = len(y)
        ntr = int(train_frac * N)
        X_tr, y_tr, z_tr = X[:ntr], y[:ntr], z[:ntr]
        X_te, y_te, z_te = X[ntr:], y[ntr:], z[ntr:]

        # masks
        m1 = z_tr <= thresh
        m2 = ~m1
        # Ensure both regimes have samples; if not, relax threshold slightly
        if m1.sum() < p+5 or m2.sum() < p+5:
            tval = np.median(z_tr)
            m1 = z_tr <= tval; m2 = ~m1

        w1 = TAR.ols_with_intercept(X_tr[m1], y_tr[m1])
        w2 = TAR.ols_with_intercept(X_tr[m2], y_tr[m2])

        # residuals per regime (for bootstrap)
        r1 = y_tr[m1] - TAR.predict_from_params(X_tr[m1], w1)
        r2 = y_tr[m2] - TAR.predict_from_params(X_tr[m2], w2)

        # BIC (rough) for info
        bic1 = TAR.aic_bic(y_tr[m1], TAR.predict_from_params(X_tr[m1], w1), k=p+1)
        bic2 = TAR.aic_bic(y_tr[m2], TAR.predict_from_params(X_tr[m2], w2), k=p+1)
        info = {"w1": w1, "w2": w2, "r1": r1, "r2": r2, "ntr": ntr,
                "y_all": y, "z_all": z, "p": p, "d": d,
                "bic1": bic1, "bic2": bic2, "thresh": thresh}
        return info

    @staticmethod
    def simulate_tar_free_run(info, init_lags, n_steps, seed=0, variance_match_to=None):
        rng = np.random.default_rng(seed)
        w1, w2 = info["w1"], info["w2"]
        r1, r2 = info["r1"], info["r2"]
        p, d, thresh = info["p"], info["d"], info["thresh"]

        state = np.array(init_lags, dtype=float).copy()  # [x_{t-1},...,x_{t-p}]
        out = np.empty(n_steps, dtype=float)
        for t in range(n_steps):
            z_t = state[d-1] if d-1 < p else state[-1]  # x_{t-d}
            w = w1 if z_t <= thresh else w2
            # bootstrap residual from the regime distribution
            resid_pool = r1 if z_t <= thresh else r2
            eps = rng.choice(resid_pool) if len(resid_pool) > 0 else 0.0
            xt = w[0] + np.dot(w[1:], state[:p]) + eps
            out[t] = xt
            state[1:p] = state[0:p-1]
            state[0] = xt
        # variance match (simple affine rescale)
        if variance_match_to is not None:
            mu_t, sd_t = np.mean(variance_match_to), np.std(variance_match_to)
            mu_o, sd_o = np.mean(out), np.std(out)
            sd_o = sd_o if sd_o > 0 else 1.0
            out = (out - mu_o) * (sd_t/sd_o) + mu_t
        return out