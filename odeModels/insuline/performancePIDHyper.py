from odeModels.insuline.hovorka import average, prob

from odeModels.insuline.simulation1Day import pidC1,pidC2,pidC3,pidC4,hyperGlicemia,hyperGlicemia

# print("pid1-hyper-v180-p0.8-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.8, 0)))
# print("pid2-hyper-v180-p0.8-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.8, 0)))
# print("pid3-hyper-v180-p0.8-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.8, 0)))
# print("pid4-hyper-v180-p0.8-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.8, 0)))

# print("pid1-hyper-v180-p0.95-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.95, 0)))
# print("pid2-hyper-v180-p0.95-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.95, 0)))
# print("pid3-hyper-v180-p0.95-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.95, 0)))
# print("pid4-hyper-v180-p0.95-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.95, 0)))

print("pid1-hyper-v180-p0.90-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.90, 0)))
print("pid2-hyper-v180-p0.90-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.90, 0)))
print("pid3-hyper-v180-p0.90-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.90, 0)))
print("pid4-hyper-v180-p0.90-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.90, 0)))



#VIENE SEMPRE 0
# print("pid1-hyper-v180-p0.999-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.999, 0)))
# print("pid2-hyper-v180-p0.999-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.999, 0)))
# print("pid3-hyper-v180-p0.999-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.999, 0)))
# print("pid4-hyper-v180-p0.999-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(180), 0.999, 0)))


# print("pid1-hyper-v300-p0.8-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(300), 0.8, 0)))
# print("pid2-hyper-v300-p0.8-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(300), 0.8, 0)))
# print("pid3-hyper-v300-p0.8-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(300), 0.8, 0)))
# print("pid4-hyper-v300-p0.8-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(300), 0.8, 0)))
# #
#
# print("pid1-hyper-v300-p0.999-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(300), 0.999, 0)))
# print("pid2-hyper-v300-p0.999-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(300), 0.999, 0)))
# print("pid3-hyper-v300-p0.999-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(300), 0.999, 0)))
# print("pid4-hyper-v300-p0.999-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(300), 0.999, 0)))
#
# print("pid1-hyper-v60-p0.95-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(60), 0.95, 0)))
# print("pid2-hyper-v60-p0.95-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(60), 0.95, 0)))
# print("pid3-hyper-v60-p0.95-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(60), 0.95, 0)))
# print("pid4-hyper-v60-p0.95-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(60), 0.95, 0)))
#
# print("pid1-hyper-v60-p0.999-prob: "+str(prob(pidC1,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(60), 0.999, 0)))
# print("pid2-hyper-v60-p0.999-prob: "+str(prob(pidC2,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(60), 0.999, 0)))
# print("pid3-hyper-v60-p0.999-prob: "+str(prob(pidC3,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(60), 0.999, 0)))
# print("pid4-hyper-v60-p0.999-prob: "+str(prob(pidC4,500, 0, 0, 1400, lambda x: 1, hyperGlicemia(60), 0.999, 0)))