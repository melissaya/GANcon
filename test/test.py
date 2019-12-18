from gancon import *
model = gancon.get_model()
A = gancon.predict(model,"./1a2pA.fasta","./1a2pA.aln")
print(A)
gancon.predict_rr(model,"./1a2pA.fasta","./1a2pA.aln","./1a2pA.rr")
