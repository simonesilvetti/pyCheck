from pycheck.semiring import Semiring


class TropicalNaturalSemiring(Semiring):
    def isBottom(self, element):
        return element == float('Inf')

    def isTop(self, element):
        return element == 0

    def isInSet(self, element):
        return element % 1 == 0 and element >= 0 or self.isBottom(element)

    def oPlus(self, left, right):
        return min(left, right)

    def oTimes(self, left, right):
        return left + right

    def accumulator(self, accumulated, element):
        return self.oTimes(accumulated, element)
