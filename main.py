from model import Model
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plot
import statsmodels.api as sm


def graph(model):
    fig, axis = plot.subplots()

    axis.set_title(model.name)
    axis.set_xlabel("Medida del cráneo en cm.")
    axis.set_ylabel("Peso en kg.")
    axis.plot(model.x, model.y, marker="o", linestyle='None', color="red")
    axis.plot(model.x, model.regression_line(), linestyle="-", color="blue")

    plot.show()


def create_model(model_name):
    print(f"Modelo: {model_name}")

    df = pd.read_csv(f"{model_name}.csv", names=["mcd", "peso"], header=0)
    x, y = df.mcd.to_list(), df.peso.to_list()

    model = Model(model_name, x, y)

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

    return model


def main():
    print("Manualmente:")

    model = create_model("raza_b")

    print("=" * 40)
    print("Con curve_fit:")

    param_opt, param_cov = curve_fit(
        lambda x, beta_0, beta_1: beta_0 + beta_1 * x,
        model.x,
        model.y
    )

    beta_0, beta_1 = param_opt

    print(f'β1: {beta_1:.2f}')
    print(f'β0: {beta_0:.2f}')

    print("=" * 40)
    print("Con statsmodels:")

    model_fit = sm.OLS(model.y, sm.add_constant(model.x)).fit()
    print(model_fit.summary())

    # graph(model)


if __name__ == '__main__':
    main()
