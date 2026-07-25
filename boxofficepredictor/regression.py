from __future__ import annotations

import math


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(matrix)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Training matrix is singular and cannot be solved")

        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for cursor in range(column, size + 1):
            augmented[column][cursor] /= pivot_value

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            for cursor in range(column, size + 1):
                augmented[row][cursor] -= factor * augmented[column][cursor]

    return [augmented[row][size] for row in range(size)]


class RidgeRegressor:
    def __init__(self, regularization: float = 1.0) -> None:
        self.regularization = regularization
        self.coefficients: list[float] | None = None

    def fit(self, samples: list[list[float]], targets: list[float]) -> None:
        if not samples:
            raise ValueError("At least one sample is required")

        width = len(samples[0])
        xtx = [[0.0 for _ in range(width)] for _ in range(width)]
        xty = [0.0 for _ in range(width)]

        for sample, target in zip(samples, targets, strict=True):
            for row in range(width):
                xty[row] += sample[row] * target
                for column in range(width):
                    xtx[row][column] += sample[row] * sample[column]

        for diagonal in range(1, width):
            xtx[diagonal][diagonal] += self.regularization

        self.coefficients = _solve_linear_system(xtx, xty)

    def predict(self, sample: list[float]) -> float:
        if self.coefficients is None:
            raise ValueError("Model must be fit before prediction")
        return sum(weight * value for weight, value in zip(self.coefficients, sample, strict=True))

    def rmse(self, samples: list[list[float]], targets: list[float]) -> float:
        squared_error = [
            (self.predict(sample) - target) ** 2
            for sample, target in zip(samples, targets, strict=True)
        ]
        return math.sqrt(sum(squared_error) / len(squared_error))
