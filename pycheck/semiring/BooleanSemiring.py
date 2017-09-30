from pycheck.semiring import Semiring


class BooleanSemiring(Semiring):
    def isBottom(self, element):
        return element == 0

    def isTop(self, element):
        return element == 1

    def isInSet(self, element):
        return element in (0, 1)

    def oPlus(self, left, right):
        return max(left, right)

    def oTimes(self, left, right):
        return min(left, right)

    def accumulator(self, accumulated, element):
        return self.oTimes(accumulated, element)
