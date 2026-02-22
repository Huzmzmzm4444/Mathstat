import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import factorial

np.random.seed(42)
n = 25
x = np.random.exponential(scale=1.0, size=n)

print("Выборка (n=25):")
print(np.round(x, 6))

sample_median = float(np.median(x))
sample_range = float(np.max(x) - np.min(x))

bin_edges = np.histogram_bin_edges(x, bins="fd")
counts, edges = np.histogram(x, bins=bin_edges)
imax = int(np.argmax(counts))
mode_hist = float((edges[imax] + edges[imax + 1]) / 2.0)

sample_skew = float(stats.skew(x, bias=False))

print("\n(a) По выборке:")
print(f"Мода: {mode_hist:.6f}")
print(f"Медиана: {sample_median:.6f}")
print(f"Размах: {sample_range:.6f}")
print(f"Асимметрия: {sample_skew:.6f}")

x_sorted = np.sort(x)
ecdf_y = np.arange(1, n + 1) / n

plt.figure()
plt.step(x_sorted, ecdf_y, where="post")
plt.xlabel("x")
plt.ylabel("F_n(x)")
plt.title("Эмпирическая функция распределения (ЭФР)")
plt.grid(True)

plt.figure()
plt.hist(x, bins=bin_edges, density=True, alpha=0.7, label="Гистограмма (оценка плотности)")
grid_x = np.linspace(0, max(x) * 1.1, 400)
plt.plot(grid_x, np.exp(-grid_x), label="Теоретическая плотность: e^{-x}, x >= 0")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.title("Гистограмма и теоретическая плотность Exp(1)")
plt.grid(True)
plt.legend()

plt.figure()
plt.boxplot(x, vert=True)
plt.ylabel("x")
plt.title("Ящик с усами (boxplot) выборки")
plt.grid(True)

B = 20000
rng = np.random.default_rng(123)

boot_means = np.array([np.mean(rng.choice(x, size=n, replace=True)) for _ in range(B)])

xbar = float(np.mean(x))
s = float(np.std(x, ddof=1))
clt_mu = xbar
clt_sigma = s / np.sqrt(n)

grid_mean = np.linspace(min(boot_means.min(), clt_mu - 4 * clt_sigma),
                        max(boot_means.max(), clt_mu + 4 * clt_sigma), 400)

clt_pdf = stats.norm.pdf(grid_mean, loc=clt_mu, scale=clt_sigma)
kde_mean = stats.gaussian_kde(boot_means)
boot_pdf_mean = kde_mean(grid_mean)

plt.figure()
plt.plot(grid_mean, clt_pdf, label="ЦПТ (нормальное приближение)")
plt.plot(grid_mean, boot_pdf_mean, label="Бутстрап (KDE)")
plt.xlabel("Среднее выборки")
plt.ylabel("Плотность")
plt.title("Плотность среднего: ЦПТ vs бутстрап")
plt.grid(True)
plt.legend()

boot_skew = np.array([stats.skew(rng.choice(x, size=n, replace=True), bias=False) for _ in range(B)])
p_skew_less_1 = float(np.mean(boot_skew < 1.0))

print(f"\n(d) P(асимметрия < 1): {p_skew_less_1:.6f}")

grid_skew = np.linspace(boot_skew.min(), boot_skew.max(), 400)
kde_skew = stats.gaussian_kde(boot_skew)

plt.figure()
plt.plot(grid_skew, kde_skew(grid_skew), label="Бутстрап (KDE)")
plt.axvline(1.0, linestyle="--", label="Порог 1")
plt.xlabel("Коэффициент асимметрии")
plt.ylabel("Плотность")
plt.title("Бутстрап-оценка плотности асимметрии")
plt.grid(True)
plt.legend()

boot_median = np.array([np.median(rng.choice(x, size=n, replace=True)) for _ in range(B)])

k = n // 2 + 1
coef = factorial(n) / (factorial(k - 1) * factorial(n - k))

def median_pdf_exp1(t):
    t = np.asarray(t)
    out = np.zeros_like(t, dtype=float)
    mask = t >= 0
    tt = t[mask]
    Ft = 1.0 - np.exp(-tt)
    ft = np.exp(-tt)
    out[mask] = coef * (Ft ** (k - 1)) * ((1.0 - Ft) ** (n - k)) * ft
    return out

grid_med = np.linspace(0.0, max(boot_median.max(), x.max()) * 1.1, 500)
theory_med_pdf = median_pdf_exp1(grid_med)

plt.figure()
plt.hist(boot_median, bins="fd", density=True, alpha=0.7, label="Бутстрап (гистограмма)")
plt.plot(grid_med, theory_med_pdf, label="Теоретическая плотность медианы")
plt.xlabel("Медиана выборки")
plt.ylabel("Плотность")
plt.title("Плотность медианы: теория и бутстрап")
plt.grid(True)
plt.legend()

plt.show()
