from model import Model

from scipy.optimize import curve_fit

import pandas as pd

import statsmodels.api as sm

import numpy as np

import plot

import sys


def create_model(model_name):
    print(f"Modelo: {model_name}")

    df = pd.read_csv(f"{model_name}.csv", names=["mcd", "peso"], header=0)
    x, y = df.mcd.to_list(), df.peso.to_list()

    return Model(model_name, x, y)


def model_manual(model):
    print("-" * 10)
    print("MUESTRA")
    print("-" * 10)
    print(f"MCD: {model.x}")
    print(f"PESO: {model.y}")

    print("-" * 10)

    print(f"x̄ = {model.x_avg():.2f}")
    print(f"ȳ = {model.y_avg():.2f}")
    print(f"Sxx = {model.sxx():.2f}")
    print(f"Sxy = {model.sxy():.2f}")

    print("-" * 10)
    print("ESTIMADORES")
    print("-" * 10)

    print(f"β1 = {model.beta_1():.2f}")
    print(f"β0 = {model.beta_0():.2f}")

    print("-" * 10)
    print("MODELO FINAL")
    print("-" * 10)

    print(f"y = {model.beta_0():.2f} + {model.beta_1():.2f}x")

    print("-" * 10)
    print(f"R2: {model.r_squared():.3f}")
    print("-" * 10)

    print("-" * 10)
    print(f"INTERVALOS DE CONFIANZA: {model.r_squared():.3f}")
    print("-" * 10)

    ic = model.confidence_interval_beta_1(0.95)

    print(f"β1 IC 95%: [{ic[0]:.2f}, {ic[1]:.2f}]")

    ic = model.confidence_interval_beta_1(0.99)
    print(f"β1 IC 99%: [{ic[0]:.2f}, {ic[1]:.2f}]")

    print(f"t={model.test_t():.2f}")


def model_curvefit(model):
    param_opt, param_cov = curve_fit(
        lambda x, beta_0, beta_1: beta_0 + beta_1 * x,
        model.x,
        model.y
    )

    beta_0, beta_1 = param_opt

    print(f'β1: {beta_1:.2f}')
    print(f'β0: {beta_0:.2f}')


def model_statsmodels(model):
    model_fit = sm.OLS(model.y, sm.add_constant(model.x)).fit()
    print(model_fit.summary())

    ic = model_fit.conf_int(0.05)[1]
    print(f"β1 Intervalo confianza 95%: [{ic[0]:.2f}, {ic[1]:.2f}]")

    ic = model_fit.conf_int(0.01)[1]
    print(f"β1 Intervalo confianza 99%: [{ic[0]:.2f}, {ic[1]:.2f}]")

    print(f"β1 P-Valor: {model_fit.pvalues[1]:.8f}")


def model_numpy(model):
    print(f"R2: {np.corrcoef(model.x, model.y)[0][1] ** 2:.3f}")


def solve(model_name):
    model = create_model(model_name)

    print("=" * 40)
    print("Manualmente:")
    model_manual(model)

    print("=" * 40)
    print("Con curve_fit:")
    model_curvefit(model)

    print("=" * 40)
    print("Con statsmodels:")
    model_statsmodels(model)

    print("=" * 40)
    print("Con numpy:")
    model_numpy(model)

    plot.plot_model(model)
    plot.plot_residuos(model)


def main():
    solve("raza_a")
    solve("raza_b")


if __name__ == '__main__':
    main()
