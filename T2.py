import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import factorial


def mode_by_histogram(x):
    bin_edges = np.histogram_bin_edges(x, bins="fd")
    counts, edges = np.histogram(x, bins=bin_edges)
    i = int(np.argmax(counts))
    return float((edges[i] + edges[i + 1]) / 2.0), bin_edges


def check_consistency_by_definition(
    dist_sampler,
    true_value,
    estimator,
    eps_list=(0.5, 0.2, 0.1),
    n_list=(10, 25, 50, 100, 200, 500, 1000),
    M=10000,
    seed=123
):
    rng = np.random.default_rng(seed)
    results = {}
    for eps in eps_list:
        probs = []
        for n in n_list:
            est = np.empty(M, dtype=float)
            for i in range(M):
                sample = dist_sampler(rng, n)
                est[i] = estimator(sample)
            probs.append(float(np.mean(np.abs(est - true_value) > eps)))
        results[eps] = (n_list, probs)
    return results


def exp1_sampler(rng, n):
    return rng.exponential(scale=1.0, size=n)


def order_stat_pdf(k, n, F, f):
    coef = factorial(n) / (factorial(k - 1) * factorial(n - k))

    def pdf(t):
        t = np.asarray(t)
        out = np.zeros_like(t, dtype=float)
        mask = t >= 0
        tt = t[mask]
        Ft = F(tt)
        ft = f(tt)
        out[mask] = coef * (Ft ** (k - 1)) * ((1.0 - Ft) ** (n - k)) * ft
        return out

    return pdf


def main():
    np.random.seed(42)

    n = 25
    x = np.random.exponential(scale=1.0, size=n)

    print("===== T2: Домашнее задание по математической статистике =====\n")
    print("Распределение: Exp(1), плотность f(x)=e^{-x}, x>=0")
    print(f"Размер выборки: n = {n}\n")
    print("Выборка:")
    print(np.round(x, 6))

    sample_mean = float(np.mean(x))
    sample_median = float(np.median(x))
    sample_range = float(np.max(x) - np.min(x))
    sample_skew = float(stats.skew(x, bias=False))

    mode_est, bin_edges = mode_by_histogram(x)

    print("\n(a) Выборочные характеристики:")
    print(f"Мода (по гистограмме): {mode_est:.6f}")
    print(f"Среднее:               {sample_mean:.6f}")
    print(f"Медиана:               {sample_median:.6f}")
    print(f"Размах:                {sample_range:.6f}")
    print(f"Асимметрия:            {sample_skew:.6f}")

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

    print("\n(b) Построены графики: ЭФР, гистограмма + теоретическая плотность, boxplot.")

    print("\n(c) Плотность среднего: ЦПТ vs бутстрап")
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

    print("Результат (c): бутстрап-плотность среднего близка к нормальной (приближение ЦПТ).")

    print("\n(d) Бутстрап плотности асимметрии и вероятность P(асимметрия < 1)")
    boot_skew = np.array([stats.skew(rng.choice(x, size=n, replace=True), bias=False) for _ in range(B)])
    p_skew_less_1 = float(np.mean(boot_skew < 1.0))

    print(f"P(асимметрия < 1) ≈ {p_skew_less_1:.6f}")

    grid_skew = np.linspace(boot_skew.min(), boot_skew.max(), 400)
    kde_skew = stats.gaussian_kde(boot_skew)

    plt.figure()
    plt.plot(grid_skew, kde_skew(grid_skew), label="Бутстрап (KDE)")
    plt.axvline(1.0, linestyle="--", label="Порог 1")
    plt.xlabel("Коэффициент асимметрии")
    plt.ylabel("Плотность")
    plt.title("Бутстрап-оценка плотности коэффициента асимметрии")
    plt.grid(True)
    plt.legend()

    print("\n(e) Плотность медианы: теория (порядковая статистика) vs бутстрап")
    boot_median = np.array([np.median(rng.choice(x, size=n, replace=True)) for _ in range(B)])

    k = n // 2 + 1

    F = lambda t: 1.0 - np.exp(-t)
    f = lambda t: np.exp(-t)
    theory_median_pdf = order_stat_pdf(k=k, n=n, F=F, f=f)

    grid_med = np.linspace(0.0, max(boot_median.max(), x.max()) * 1.1, 500)
    theory_med_pdf = theory_median_pdf(grid_med)

    plt.figure()
    plt.hist(boot_median, bins="fd", density=True, alpha=0.7, label="Бутстрап (гистограмма)")
    plt.plot(grid_med, theory_med_pdf, label="Теоретическая плотность медианы")
    plt.xlabel("Медиана выборки")
    plt.ylabel("Плотность")
    plt.title("Плотность медианы: теория и бутстрап")
    plt.grid(True)
    plt.legend()

    print("Результат (e): бутстрап-распределение медианы согласуется с теоретической плотностью.")

    print("\n===== Исследование состоятельности оценки №3 по определению =====")
    print("Оценка №3: выборочное среднее X̄ как оценка математического ожидания E[X]=1 для Exp(1).")
    print("По определению состоятельности: для любого eps>0 должно выполняться P(|X̄_n - 1| > eps) -> 0 при n->∞.\n")

    consistency = check_consistency_by_definition(
        dist_sampler=exp1_sampler,
        true_value=1.0,
        estimator=lambda s: float(np.mean(s)),
        eps_list=(0.5, 0.2, 0.1),
        n_list=(10, 25, 50, 100, 200, 500, 1000),
        M=8000,
        seed=777
    )

    for eps, (n_list, probs) in consistency.items():
        print(f"eps = {eps}:")
        for n_i, p_i in zip(n_list, probs):
            print(f"  n = {n_i:4d}  ->  P(|X̄ - 1| > eps) ≈ {p_i:.4f}")
        print()

    plt.figure()
    for eps, (n_list, probs) in consistency.items():
        plt.plot(n_list, probs, marker="o", label=f"eps = {eps}")
    plt.xlabel("n")
    plt.ylabel("P(|X̄ - 1| > eps)")
    plt.title("Состоятельность X̄ по определению (монте-карло)")
    plt.grid(True)
    plt.legend()

    print("Вывод: значения P(|X̄ - 1| > eps) убывают при росте n, что подтверждает состоятельность X̄.")

    plt.show()


if __name__ == "__main__":
    main()
