class Semiring:

    def getBottom(self, element):
        pass

    def getTop(self, element):
        pass

    def getSet(self, element):
        pass

    def isBottom(self, element):
        pass

    def isTop(self, element):
        pass

    def isInSet(self, element):
        pass


    def oPlus(self, left, right):
        pass

    def oTimes(self,left, right):
        pass

    def oNeg(self, element):
        pass

    def isIncluded(self,left, right):
        return self.oPlus(left, right) == right

    def accumulator(self, accumulated, element):
        pass

