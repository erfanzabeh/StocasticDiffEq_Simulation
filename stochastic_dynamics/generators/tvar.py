import numpy as np

def tvar(p, T=600, W=40, P0=0.5, sigma_noise=0.1, burnin=300, max_attempts=100):
    """Generate one TVAR(p) sample with power constraint and burnin.
    
    Args:
        p (int): Order of the TVAR model.
        T (int): Length of the desired output time series.
        W (int): Window size for power constraint.
        P0 (float): Maximum allowed power in any window of size W.
        sigma_noise (float): Standard deviation of the Gaussian noise.
        burnin (int): Number of initial samples to discard.
        max_attempts (int): Maximum number of attempts to generate a valid sample.
    """
    
    T_total = T + burnin
    
    # Coefficient functions - scaled for stability
    base_freqs = [1/150, 1/200, 1/180, 1/220, 1/170, 1/190]
    base_amps = [0.35, 0.25, 0.20, 0.15, 0.12, 0.10]
    base_offsets = [0.6, -0.5, 0.3, -0.2, 0.15, -0.1]
    scale = 1.0 / np.sqrt(p)
    
    for attempt in range(max_attempts):
        # Generate coefficients
        coeffs_full = np.zeros((T_total, p))
        for k in range(p):
            freq = base_freqs[k % 6]
            amp = base_amps[k % 6] * scale
            offset = base_offsets[k % 6] * scale
            phase = k * np.pi / 4
            t_arr = np.arange(T_total)
            if k % 2 == 0:
                coeffs_full[:, k] = offset + amp * np.sin(2 * np.pi * freq * t_arr + phase)
            else:
                coeffs_full[:, k] = offset + amp * np.cos(2 * np.pi * freq * t_arr + phase)
        
        # Simulate signal
        x_full = np.zeros(T_total)
        for i in range(T_total):
            val = np.random.normal(scale=sigma_noise)
            for k in range(p):
                if i > k:
                    val += coeffs_full[i, k] * x_full[i - k - 1]
            x_full[i] = val
            
            if i >= W - 1:
                window = x_full[i - W + 1:i + 1]
                current_power = np.mean(window ** 2)
                if current_power > 0:
                    s = np.clip(np.sqrt(P0 / current_power), 0.8, 1.2)
                    x_full[i] *= s
        
        # Discard burnin
        x = x_full[burnin:]
        coeffs = coeffs_full[burnin:]
        
        # Check power constraint (same logic as original)
        power = np.zeros(T)
        for i in range(T):
            start = max(0, i - W + 1)
            power[i] = np.mean(x[start:i + 1] ** 2)
        
        if power.max() <= P0:
            return x, coeffs
    
    raise ValueError(f"Failed to generate valid sample after {max_attempts} attempts")
