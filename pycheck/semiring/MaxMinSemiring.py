from pycheck.semiring import Semiring


class MaxMinSemiring(Semiring):
    def isBottom(self, element):
        return element == - float('Inf')

    def isTop(self, element):
        return element == float('Inf')

    def isInSet(self, element):
        return True

    def oPlus(self, left, right):
        return max(left, right)

    def oTimes(self, left, right):
        return min(left, right)

    def isIncluded(self, left, right):
        return self.oPlus(left, right) == right

    def accumulator(self, accumulated, element):
        return self.oTimes(accumulated, element)