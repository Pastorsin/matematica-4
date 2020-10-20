import statistics
from scipy import stats
from math import sqrt


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

    def syy(self):
        return sum([(yi - self.y_avg()) ** 2 for yi in self.y])

    def beta_0(self):
        return self.y_avg() - (self.x_avg() * self.beta_1())

    def beta_1(self):
        return self.sxy() / self.sxx()

    def estimated_y(self):
        return [self.beta_0() + self.beta_1() * xi for xi in self.x]

    def srr(self):
        return self.syy() - (self.sxy() ** 2) / self.sxx()

    def r_squared(self):
        return 1 - self.srr() / self.syy()

    def residuos(self):
        y_y = zip(self.y, self.estimated_y())

        return [y - estimated_y for y, estimated_y in y_y]

    def sr(self):
        return sqrt(self.srr() / (len(self.x) - 2))

    def t(self, confidence):
        n = len(self.x) - 2
        alpha = (1 - confidence) / 2

        return abs(stats.t.ppf(alpha, n))

    def confidence_interval_beta_1(self, confidence):
        t, sr, sxx = self.t(confidence), self.sr(), self.sxx()

        print(f"t={t:.3f}")

        low = self.beta_1() - t * (sr / sqrt(sxx))
        up = self.beta_1() + t * (sr / sqrt(sxx))

        return low, up

    def test_t(self):
        beta_1, sr, sxx = self.beta_1(), self.sr(), self.sxx()

        return beta_1 / (sr / sqrt(sxx))
