import matplotlib.pyplot as plot


def plot_model(model):
    fig, axis = plot.subplots()

    axis.set_title(model.name)
    axis.set_xlabel("Medida del cráneo en cm.")
    axis.set_ylabel("Peso en kg.")
    axis.plot(model.x, model.y, marker="o", linestyle='None', color="red")
    axis.plot(model.x, model.estimated_y(), linestyle="-", color="blue")

    plot.show()


def plot_residuos(model):
    fig, axis = plot.subplots()

    zero_constant = [0] * len(model.x)

    axis.set_title(f"Residuos de {model.name}")
    axis.plot(model.x, model.residuos(), marker="o",
              linestyle='None', color="red")
    axis.plot(model.x, zero_constant, linestyle="-", color="blue")

    plot.show()
