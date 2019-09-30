from smothed.smoothedMC import smoothedMC, smoothedMCNaive, drowSquares, drowSquares2

modelName = 'epidemic.psc'
paramterName= 'ki'
timeEnd=40
trajectoriesNumber=600
mitlFormula = 'F_[20,30](R>30)\n'
precision = 0.1
interval=[0.005,0.3]
# smoothedMC(modelName,paramterName,timeEnd,trajectoriesNumber,mitlFormula,precision,interval)
# smoothedMCNaive(modelName,paramterName,timeEnd,trajectoriesNumber,mitlFormula,precision,interval,10)
drowSquares2(interval,200,precision,modelName,paramterName,timeEnd,trajectoriesNumber,mitlFormula)

# input_stream = InputStream('G_[0,1] (S > G) \n')
# lexer = STLLexer(input=input_stream)
# token_stream = CommonTokenStream(lexer)
# parser = STLParser(token_stream)
# objectiveTree = parser.prog()