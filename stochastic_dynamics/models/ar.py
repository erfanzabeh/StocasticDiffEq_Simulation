import numpy as np

class AR:
    
    @staticmethod
    def lag_matrix(x, p):
        N = len(x)
        y = x[p:].copy()
        X = np.zeros((N-p, p))
        for i in range(p):
            X[:, i] = x[p-1-i : N-1-i]
        return X, y

    @staticmethod
    def fit(X, y):
        D = np.column_stack([np.ones(len(X)), X])
        w = np.linalg.lstsq(D, y, rcond=None)[0]
        return w  # [c, a1..ap]

    @staticmethod
    def predict_from_params(X, w):
        D = np.column_stack([np.ones(len(X)), X])
        return D @ w

    @staticmethod
    def metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred)**2)
        mae = np.mean(np.abs(y_true - y_pred))
        yt = y_true - np.mean(y_true)
        yp = y_pred - np.mean(y_pred)
        r = (yt @ yp) / np.sqrt((yt @ yt) * (yp @ yp))
        return mse, mae, r

    @staticmethod
    def acf(sig, nlags=60):
        s = sig - np.mean(sig)
        ac = np.correlate(s, s, mode="full")
        ac = ac[ac.size//2:]
        ac0 = ac[0] if ac[0] != 0 else 1.0
        ac = ac / ac0
        return ac[:nlags+1]

    @staticmethod
    def aic_bic(y, yhat, k):
        n = len(y)
        rss = np.sum((y - yhat)**2)
        sigma2 = rss / max(n, 1)
        aic = n*np.log(sigma2 + 1e-12) + 2*k
        bic = n*np.log(sigma2 + 1e-12) + k*np.log(max(n, 2))
        return aic, bic
    
    @staticmethod
    def hybrid_predict(series, w, p, start_idx, n_steps, refresh_every=1):
        preds = []
        hist = series[start_idx - p : start_idx].tolist()
        for t in range(n_steps):
            if (t % max(int(refresh_every), 1)) == 0:
                abs_idx = start_idx + t
                hist = series[abs_idx - p : abs_idx].tolist()
            xhat = w[0] + np.dot(w[1:], list(reversed(hist)))
            preds.append(xhat)
            hist.pop(0)
            hist.append(xhat)
        return np.array(preds)