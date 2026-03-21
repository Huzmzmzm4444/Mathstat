import numpy as np
import scipy.stats as st

np.random.seed(123)

def inv_pareto(u, th):
    return (1 - u)**(1 / (1 - th))

def med(th):
    return 2 ** (1/(th-1))

# Параметры
n = 100
conf = 0.95
th_true = 5
x = inv_pareto(st.uniform(loc=0, scale=1).rvs(size=n), th=th_true)

# ОМП и Медиана
th_hat = 1 + 1 / (np.mean(np.log(x)))
med_true = med(th_true)
med_hat = med(th_hat)
z_low = st.norm.ppf((1-conf)/2)
z_high = st.norm.ppf((1+conf)/2)

# 1. Асимптотический интервал для медианы
ci_asym_med_left = med_hat - med_hat*np.log(2)/(np.sqrt(n)*(th_hat-1))*z_high
ci_asym_med_right = med_hat - med_hat*np.log(2)/(np.sqrt(n)*(th_hat-1))*z_low
ci_asym_med_len = ci_asym_med_right - ci_asym_med_left

print(f"Медиана: {med_true}, оценка медианы: {med_hat}")
print(f"Асимптотический ДИ для медианы: {ci_asym_med_left} < med < {ci_asym_med_right}")

# 2. Асимптотический интервал для theta
ci_asym_left = th_hat - z_high / (np.sum(np.log(x))/np.sqrt(n))
ci_asym_right = th_hat - z_low / (np.sum(np.log(x))/np.sqrt(n))
ci_asym_len = ci_asym_right - ci_asym_left

print(f"theta: {th_true}, MLE theta: {th_hat}")
print(f"Асимптотический ДИ для theta: {ci_asym_left} < theta < {ci_asym_right}")

# 3. Бутстраповские интервалы
B = 1000
idx_low = int((1 - conf)/2 * B - 1)
idx_high = int((1 + conf)/2 * B - 1)

# Непараметрический
def boot_np(samp, B):
    diffs = []
    m = len(samp)
    for _ in range(B):
        samp_b = np.random.choice(samp, size=m, replace=True)        
        diffs += [(1 + 1 / (np.mean(np.log(samp_b)))) - th_hat]
    return sorted(np.array(diffs))

diffs_np = boot_np(x, B)
ci_np_left = th_hat - diffs_np[idx_high]
ci_np_right = th_hat - diffs_np[idx_low]
ci_np_len = ci_np_right - ci_np_left

# Параметрический
def boot_p(samp, B, th_est):
    diffs = []
    m = len(samp)
    for _ in range(B):
        samp_b = inv_pareto(st.uniform(loc=0, scale=1).rvs(size=m), th=th_est)        
        diffs += [(1 + 1 / (np.mean(np.log(samp_b)))) - th_est]
    return sorted(np.array(diffs))

diffs_p = boot_p(x, B, th_hat)
ci_p_left = th_hat - diffs_p[idx_high]
ci_p_right = th_hat - diffs_p[idx_low]
ci_p_len = ci_p_right - ci_p_left

print(f"Параметрический бутстрап: {ci_p_left} < theta < {ci_p_right}")
print(f"Непараметрический бутстрап: {ci_np_left} < theta < {ci_np_right}")

# 4. Рейтинг
lens = sorted([(ci_asym_len, "Асимптотический"), 
               (ci_np_len, "Бутстрап непараметрический"), 
               (ci_p_len, "Бутстрап параметрический")])
print("\nРейтинг (для theta):")
for i in range(len(lens)):
    print(f"{i+1}) {lens[i][1]} (len = {np.round(lens[i][0],3)})")