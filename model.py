import statistics


class Model:

    def __init__(self, model_name, x, y):
        self.name = model_name
        self.x, self.y = x, y

    def x_avg(self):
        return statistics.mean(self.x)

    def y_avg(self):
        return statistics.mean(self.y)

    def sxx(self):
        return sum([(xi - self.x_avg()) ** 2 for xi in self.x])

    def sxy(self):
        x_y = zip(self.x, self.y)

        return sum([(x - self.x_avg()) * (y - self.y_avg()) for x, y in x_y])

    def beta_0(self):
        return self.y_avg() - (self.x_avg() * self.beta_1())

    def beta_1(self):
        return self.sxy() / self.sxx()

    def regression_line(self):
        return [self.beta_0() + self.beta_1() * xi for xi in self.x]
