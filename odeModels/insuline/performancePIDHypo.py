from odeModels.insuline.hovorka import average, prob

from odeModels.insuline.simulation1Day import pidC1,pidC2,pidC3,pidC4,hyperGlicemia,hypoGlicemia

#
# print("pid1-hypo-v80-p0.8-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(80), 0.8, 0)))
# print("pid2-hypo-v80-p0.8-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(80), 0.8, 0)))
# print("pid3-hypo-v80-p0.8-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(80), 0.8, 0)))
# print("pid4-hypo-v80-p0.8-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(80), 0.8, 0)))
#
# #VIENE SEMPRE 0
# print("pid1-hypo-v80-p0.999-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(80), 0.999, 0)))
# print("pid2-hypo-v80-p0.999-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(80), 0.999, 0)))
# print("pid3-hypo-v80-p0.999-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(80), 0.999, 0)))
# print("pid4-hypo-v80-p0.999-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(80), 0.999, 0)))
#
#
# print("pid1-hypo-v70-p0.8-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.8, 0)))
# print("pid2-hypo-v70-p0.8-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.8, 0)))
# print("pid3-hypo-v70-p0.8-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.8, 0)))
# print("pid4-hypo-v70-p0.8-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.8, 0)))
# #
#
# print("pid1-hypo-v70-p0.999-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
# print("pid2-hypo-v70-p0.999-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
# print("pid3-hypo-v70-p0.999-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
# print("pid4-hypo-v70-p0.999-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
#
# print("pid1-hypo-v60-p0.95-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.95, 0)))
# print("pid2-hypo-v60-p0.95-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.95, 0)))
# print("pid3-hypo-v60-p0.95-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.95, 0)))
# print("pid4-hypo-v60-p0.95-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.95, 0)))
#
# print("pid1-hypo-v60-p0.999-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.999, 0)))
# print("pid2-hypo-v60-p0.999-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.999, 0)))
# print("pid3-hypo-v60-p0.999-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.999, 0)))
# print("pid4-hypo-v60-p0.999-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.999, 0)))


# print("pid1-hypo-v60-p0.8: "+str(prob(pidC1,200, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.8, 0)))
# print("pid2-hypo-v60-p0.8: "+str(prob(pidC2,200, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.8, 0)))
# print("pid3-hypo-v60-p0.8: "+str(prob(pidC3,200, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.8, 0)))
# print("pid4-hypo-v60-p0.8: "+str(prob(pidC4,200, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.8, 0)))
#
# print("pid1-hypo-v70-p0.999: "+str(average(pidC1,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
# print("pid2-hypo-v70-p0.999: "+str(average(pidC2,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
# print("pid3-hypo-v70-p0.999: "+str(average(pidC3,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
# print("pid4-hypo-v70-p0.999: "+str(average(pidC4,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
#
#
#
# print("pid1-hypo-v70-p0.8: "+str(average(pidC1,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.8, 0)))
# print("pid2-hypo-v70-p0.8: "+str(average(pidC2,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.8, 0)))
# print("pid3-hypo-v70-p0.8: "+str(average(pidC3,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.8, 0)))
# print("pid4-hypo-v70-p0.8: "+str(average(pidC4,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.8, 0)))
#
# print("pid1-hypo-v60-p0.8: "+str(average(pidC1,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.8, 0)))
# print("pid2-hypo-v60-p0.8: "+str(average(pidC2,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.8, 0)))
# print("pid3-hypo-v60-p0.8: "+str(average(pidC3,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.8, 0)))
# print("pid4-hypo-v60-p0.8: "+str(average(pidC4,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(60), 0.8, 0)))
#
# print("pid1-hypo-v70-p0.999: "+str(average(pidC1,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
# print("pid2-hypo-v70-p0.999: "+str(average(pidC2,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
# print("pid3-hypo-v70-p0.999: "+str(average(pidC3,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
# print("pid4-hypo-v70-p0.999: "+str(average(pidC4,100, 0, 0, 1400, lambda x: 1, hypoGlicemia(70), 0.999, 0)))
#
# print("pid1-hyper-v180-p0.7: "+str(average(pidC1,100, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.7, 0)))
# print("pid2-hyper-v180-p0.7: "+str(average(pidC2,100, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.7, 0)))
# print("pid3-hyper-v180-p0.7: "+str(average(pidC3,100, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.7, 0)))
# print("pid4-hyper-v180-p0.7: "+str(average(pidC4,100, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.7, 0)))
#
#
# print("pid1-hyper-v180-p0.999: "+str(average(pidC1,100, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.999, 0)))
# print("pid2-hyper-v180-p0.999: "+str(average(pidC2,100, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.999, 0)))
# print("pid3-hyper-v180-p0.999: "+str(average(pidC3,100, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.999, 0)))
# print("pid4-hyper-v180-p0.999: "+str(average(pidC4,100, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.999, 0)))


print("pid1-hyper-v180-p0.999: "+str(prob(pidC1,200, 0, 0, 1438, lambda x: 1, hyperGlicemia(300), 0.87, 0)))
print("pid2-hyper-v180-p0.999: "+str(prob(pidC2,200, 0, 0, 1438, lambda x: 1, hyperGlicemia(300), 0.87, 0)))
print("pid3-hyper-v180-p0.999: "+str(prob(pidC3,200, 0, 0, 1438, lambda x: 1, hyperGlicemia(300), 0.87, 0)))
print("pid4-hyper-v180-p0.999: "+str(prob(pidC4,200, 0, 0, 1438, lambda x: 1, hyperGlicemia(300), 0.87, 0)))
