import copy

from  matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from scipy.integrate import odeint

from pycheck.semiring.TropicalRealSemiring import TropicalRealSemiring
from pycheck.series.TimeSeries import TimeSeries


def sir(b, g, y0, time, n):
    def sirEquation(y, t, b, g):
        S, I, R = y
        N = S + I + R
        dydt = [-b * S * I / N, b * S * I / N - g * I, g * I]
        return dydt

    t = np.linspace(time[0], time[1], n)
    sol = odeint(sirEquation, y0, t, args=(b, g))
    return t, sol


t, sol = sir(1, 2, [95, 5, 0], [0, 300], 500)
serie = TimeSeries(['A', 'B', 'C'], t, sol)

G = nx.Graph()
G.add_node(1, p1=1, p2=-1)
G.add_node(2, p1=1, p2=-1)
G.add_node(3, p1=1, p2=-1)
G.add_node(4, p1=-10, p2=-1)
# G.add_node(5, p1=2, p2=7)
# G.add_node(6, p1=-2, p2=-1)
# G.add_node(7, p1=-2, p2=-1)
# G.add_node(8, p1=-2, p2=-1)

G.add_edge(1, 2, w=2)
G.add_edge(1, 4, w=1.1)
G.add_edge(2, 3, w=2)
G.add_edge(4, 3, w=1)
# G.add_edge(5, 6, w=1)
# G.add_edge(6, 8, w=2)
# G.add_edge(8, 5, w=10)
# G.add_edge(6, 7, w=6)


def distancePredicate(value, d):
    return value <= d


def reach(G, d):
    semiring = TropicalRealSemiring()
    datab = {}
    for l in G.nodes():
        locdatab = {}
        locdatab[(l, G.node[l]['p2'])] = 0
        datab[l] = locdatab
    stable = False
    while not stable:
        stable = True
        databCopy = copy.deepcopy(datab)
        for l in G.nodes():
            for ln in G.neighbors(l):
                for tLoc, tValue in datab[ln]:
                    newDistance = semiring.accumulator(datab[ln][(tLoc, tValue)], G[l][ln]['w'])
                    if distancePredicate(newDistance, d):
                        newValue = min(tValue, G.node[l]['p1'])
                        if ((tLoc, newValue) in databCopy[l]):
                            databCopy[l][(tLoc, newValue)] = semiring.oPlus(databCopy[l][(tLoc, newValue)], newDistance)
                        else:
                            databCopy[l][(tLoc, newValue)] = newDistance
            if (databCopy[l] != datab[l]):
                stable = False
        datab = databCopy
    GCopy = G.copy()
    nx.set_node_attributes(GCopy, 'eval', 0)
    for l in GCopy.nodes():
        GCopy.node[l]['eval'] = max([value for p, value in datab[l]])
    return GCopy


def escape(G, d):
    semiring = TropicalRealSemiring()
    datab = {}
    for l in G.nodes():
        locdatab = {}
        locdatab[l] = (G.node[l]['p1'], 0)
        datab[l] = locdatab
    stable = False
    while not stable:
        stable = True
        databCopy = copy.deepcopy(datab)
        for l in G.nodes():
            for ln in G.neighbors(l):
                for tLoc in datab[ln]:
                    newDistance = semiring.accumulator(datab[ln][tLoc][1], G[l][ln]['w'])
                    newValue = min(datab[ln][tLoc][0], G.node[l]['p1'])
                    if tLoc in databCopy[l]:
                        newNewValue = max(databCopy[l][tLoc][0], newValue)
                        newNewDistance = semiring.oPlus(databCopy[l][tLoc][1], newDistance)
                        databCopy[l][tLoc] = (newNewValue, newNewDistance)
                    else:
                        databCopy[l][tLoc] = (newValue, newDistance)
            if (databCopy[l] != datab[l]):
                stable = False
        datab = databCopy
    GCopy = G.copy()
    nx.set_node_attributes(GCopy, 'eval', 0)
    for l in GCopy.nodes():
        [print(databCopy[l][ldata]) for ldata in databCopy[l]]
        z = [databCopy[l][ldata] for ldata in databCopy[l]]
        fz=[a for a,b in z if not distancePredicate(b, d)]
        if not(fz):
            GCopy.node[l]['eval'] = - float('Inf')
        else:
            GCopy.node[l]['eval'] = max(fz)
    return GCopy

# GG = reach(G, 10)
# print(GG.nodes())
# print(GG.nodes(data=True))
# node_labels = dict((n, str(n) + ' (' + str(d['eval']) + ')') for n, d in GG.nodes(data=True))
# # # labels=nx.draw_networkx_edge_labels(GG,pos=nx.spring_layout(GG))
# # nx.draw(GG,labels=labels,node_size=1300,font_color='white',pos=nx.spring_layout(GG))
# # plt.show()
#
#
# pos = nx.spring_layout(GG)
# nx.draw(GG, pos, node_size=1500)
# # node_labels = nx.get_node_attributes(G,'state')
# nx.draw_networkx_labels(GG, pos, labels=node_labels, font_color='white')
# edge_labels = nx.get_edge_attributes(GG, 'w')
# nx.draw_networkx_edge_labels(GG, pos, labels=edge_labels)
# # nx.draw(GG,node_size=1300)
# # nx.draw_networkx_nodes(GG,node_size=1500,pos=pos)
# plt.savefig('this.png')
# plt.show()

GG = escape(G, 2)
print(GG.nodes())
print(GG.nodes(data=True))
node_labels = dict((n, str(n) + ' (' + str(d['eval']) + ')') for n, d in GG.nodes(data=True))
# # labels=nx.draw_networkx_edge_labels(GG,pos=nx.spring_layout(GG))
# nx.draw(GG,labels=labels,node_size=1300,font_color='white',pos=nx.spring_layout(GG))
# plt.show()


pos = nx.spring_layout(GG)
nx.draw(GG, pos, node_size=1500)
# node_labels = nx.get_node_attributes(G,'state')
nx.draw_networkx_labels(GG, pos, labels=node_labels, font_color='white')
edge_labels = nx.get_edge_attributes(GG, 'w')
nx.draw_networkx_edge_labels(GG, pos, labels=edge_labels)
# nx.draw(GG,node_size=1300)
# nx.draw_networkx_nodes(GG,node_size=1500,pos=pos)
plt.savefig('that.png')
plt.show()
