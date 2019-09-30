import stochpy
from antlr4 import FileStream
import numpy as np

from provas.BooleanSemantics import BooleanSemantics
from provas.FilterSemantics import FilterSemantics
from provas.FilterSemanticsNaive import FilterSemanticsNaive
from provas.QuantitativeSemantics import QuantitativeSemantics
from provas.StochSerie import StochSerie
from provas.STLLexer import STLLexer, CommonTokenStream
from provas.STLParser import STLParser

smod = stochpy.SSA()
smod.Model('dsmts-003-03.xml.psc')
smod.DoStochSim(end=50,mode='time',trajectories=1)
smod.ChangeParameter('k1',10)
smod.ChangeParameter('k2',0.1)

input_stream = FileStream('tProva.expr')
lexer = STLLexer(input_stream)

token_stream = CommonTokenStream(lexer)
parser = STLParser(token_stream)
tree = parser.prog()

# lisp_tree_str = tree.toStringTree(recog=parser)
# print(lisp_tree_str)
# smod = sp.SSA()
serie = StochSerie(['P'],np.array([0,1,2,3,4])[:,None],np.array([2,-5,1,-1,-1])[:,None])
visitor = BooleanSemantics(series=serie)
print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
visitor = QuantitativeSemantics(series=serie)
print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
visitor = FilterSemantics(series=serie)
print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
visitor = FilterSemanticsNaive(series=serie)
print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
# smod.data_stochsim.species_labels