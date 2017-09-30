from pycheck.series.SpatialSignal import *


class Monitoring:
    def __init__(self, semiring):
        self.semiring = semiring

    def monitor(self, formula, locServ, spTimeSignal):
        return None

    def atomic(self, mu, spTimeSignal):
        return spTimeSignal.map(mu)

    def conjunction(self, spTimeSignal1, spTimeSignal2):
        return spTimeSignal1.merge(self.semiring.oTimes, spTimeSignal2)

    def disjunction(self, spTimeSignal1, spTimeSignal2):
        return spTimeSignal1.merge(self.semiring.oPlus, spTimeSignal2)

    def neg(self, spTimeSignal):
        return spTimeSignal.map(self.semiring.oNeg)

    def eventually(self, spTimeSignal, a, b):
        return None

    def always(self, spTimeSignal, a, b):
        return None

    def until(self, spTimeSignal1, spTimeSignal2, a, b):
        return None

    def reach(self, spTimeSignal1, spTimeSignal2, f, d):
        return None

    def escape(self, spTimeSignal, f, d):
        return None
