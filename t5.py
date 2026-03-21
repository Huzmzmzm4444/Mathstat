import numpy as np
import scipy.stats as st

np.random.seed(123)

# Параметры
n = 100
beta = 0.95
theta_true = 22
data = st.uniform(loc=theta_true, scale=theta_true).rvs(size=n)

# Оценки
sample_mean = np.mean(data)
sample_max = np.max(data)
theta_mom = 2/3 * sample_mean
theta_mle = (n+1) * sample_max / (2*n+1)
sample_var = np.sum((data - sample_mean)**2) / n

# 1. Точный доверительный интервал
gamma1 = ((1 + beta)/2)**(1/n)
gamma2 = ((1 - beta)/2)**(1/n)
accurate_left = sample_max / (1 + gamma1)
accurate_right = sample_max / (1 + gamma2)
accurate_length = accurate_right - accurate_left

print(f"Точный доверительный интервал для theta: {accurate_left:.6f} < theta < {accurate_right:.6f}")
print(f"Длина интервала: {accurate_length:.6f}")

# 2. Асимптотический доверительный интервал (ОММ)
z_crit = st.norm.ppf((1 + beta)/2)
std_error = 2/3 * np.sqrt(sample_var / n)
asym_left = theta_mom - z_crit * std_error
asym_right = theta_mom + z_crit * std_error
asymptotic_length = asym_right - asym_left

print(f"Асимптотический доверительный интервал для theta: {asym_left} < theta < {asym_right}")
print(f"Длина интервала: {asymptotic_length}")

# 3. Непараметрический бутстрап
N_bootstrap = 1000

def bootstrap_omm(data_sample, B):
    bootstrap_stats = []
    n_sample = len(data_sample)
    for _ in range(B):
        bootstrap_sample = np.random.choice(data_sample, size=n_sample, replace=True)        
        bootstrap_stats.append(2/3 * np.mean(bootstrap_sample) - theta_mom)
    return sorted(np.array(bootstrap_stats))

bootstrap_arr_omm = bootstrap_omm(data, N_bootstrap)
idx_l = int((1 - beta)/2 * N_bootstrap - 1)
idx_r = int((1 + beta)/2 * N_bootstrap - 1)
bs_omm_left = theta_mom - bootstrap_arr_omm[idx_r]
bs_omm_right = theta_mom - bootstrap_arr_omm[idx_l]
bs_omm_length = bs_omm_right - bs_omm_left

print(f"Непараметрический бутстраповский доверительный интервал для theta(ОММ): {bs_omm_left} < theta < {bs_omm_right}")
print(f"l = {bs_omm_length}")

def bootstrap_omp(data_sample, B):
    bootstrap_stats = []
    n_sample = len(data_sample)
    for _ in range(B):
        bootstrap_sample = np.random.choice(data_sample, size=n_sample, replace=True)        
        bootstrap_stats.append((n_sample + 1) * np.max(bootstrap_sample) / (2 * n_sample + 1) - theta_mle)
    return sorted(np.array(bootstrap_stats))

bootstrap_arr_omp = bootstrap_omp(data, N_bootstrap)
idx_l_p = int((1 - beta)/2 * N_bootstrap)
idx_r_p = int((1 + beta)/2 * N_bootstrap)
bs_omp_left = theta_mle - bootstrap_arr_omp[idx_r_p]
bs_omp_right = theta_mle - bootstrap_arr_omp[idx_l_p]
bs_omp_length = bs_omp_right - bs_omp_left

print(f"Непараметрический бутстраповский доверительный интервал для theta(ОМП): {bs_omp_left} < theta < {bs_omp_right}")
print(f"l = {bs_omp_length}")

# 4. Сравнение
intervals_lengths = [
    (accurate_length, "Точный"),
    (asymptotic_length, "Асимптотический"),
    (bs_omm_length, "Бутстрап ОММ"),
    (bs_omp_length, "Бутстрап ОМП")
]
intervals_lengths.sort()

print("\nРейтинг доверительных интервалов:")
for i, (length, name) in enumerate(intervals_lengths, 1):
    print(f"{i}) {name} (l = {np.round(length, 3)})")