import stochpy
from antlr4 import FileStream

from provas.BooleanSemantics import BooleanSemantics
from provas.FilterSemantics import FilterSemantics
from provas.QuantitativeSemantics import QuantitativeSemantics
from provas.StochSerie import StochSerie
from provas.STLLexer import STLLexer, CommonTokenStream
from provas.STLParser import STLParser

smod = stochpy.SSA()
smod.Model('epidemic.psc')
smod.DoStochSim(end=130,mode='time',trajectories=300)
input_stream = FileStream('t.expr')
lexer = STLLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser = STLParser(token_stream)
tree = parser.prog()
count = 0
for i in range(1,301):
    smod.GetTrajectoryData(i)
    # if(max(smod.data_stochsim.time)<130):
    #     print("CAZZO")
    visitor = BooleanSemantics(series=StochSerie(smod.data_stochsim.species_labels, smod.data_stochsim.time, smod.data_stochsim.species))
    count+=1 if visitor.visit(tree) else 0
print (count/300)
# lisp_tree_str = tree.toStringTree(recog=parser)
# print(lisp_tree_str)
# smod = sp.SSA()

# visitor = BooleanSemantics(series=StochSerie(smod.data_stochsim.species_labels, smod.data_stochsim.time, smod.data_stochsim.species))
# print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
# visitor = QuantitativeSemantics(series=StochSerie(smod.data_stochsim.species_labels, smod.data_stochsim.time, smod.data_stochsim.species))
# print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
# visitor = FilterSemantics(series=StochSerie(smod.data_stochsim.species_labels, smod.data_stochsim.time, smod.data_stochsim.species))
# print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
# smod.data_stochsim.species_labels