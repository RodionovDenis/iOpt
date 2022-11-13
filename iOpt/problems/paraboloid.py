import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class Paraboloid(Problem):

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.numberOfFloatVariables = dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0

        self.floatVariableNames = np.ndarray(shape=(dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(-2.2)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(dimension), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(1.8)

        self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(dimension), dtype=np.double)
        pointfv.fill(0)
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 0
        self.knownOptimum[0] = Trial(KOpoint, KOfunV)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """Compute selected function at given point."""
        sum: np.double = 0
        for i in range(self.dimension):
            sum += point.floatVariables[i] * point.floatVariables[i]

        functionValue.value = sum
        return functionValue
