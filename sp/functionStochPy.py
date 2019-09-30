import stochpy
from antlr4 import FileStream

from provas.BooleanSemantics import BooleanSemantics
from provas.FilterSemantics import FilterSemantics
from provas.QuantitativeSemantics import QuantitativeSemantics
from provas.StochSerie import StochSerie
from provas.STLLexer import STLLexer, CommonTokenStream
from provas.STLParser import STLParser

def stocchizza(a):
    smod = stochpy.SSA()
    smod.Model('dsmts-003-03.xml.psc')
    smod.DoStochSim(end=50, mode='time', trajectories=1)
    smod.ChangeParameter('k1', a[0])
    smod.ChangeParameter('k2', a[1])
    input_stream = FileStream('t.expr')
    lexer = STLLexer(input_stream)

    token_stream = CommonTokenStream(lexer)
    parser = STLParser(token_stream)
    tree = parser.prog()
    aa=StochSerie(smod.data_stochsim.species_labels, smod.data_stochsim.time, smod.data_stochsim.species)
    visitor = QuantitativeSemantics(
        series=aa)
    quant = visitor.visit(tree)
    visitor = FilterSemantics(
        series=aa)
    fil = visitor.visit(tree)
    return quant, fil