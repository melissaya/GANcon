import os
import tempfile
import sys
import linecache
from math import log,exp
import numpy as np
import gaussdca
from keras.models import load_model

    
def get_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_file_name = base_path + "/model/gancon.h5"
    model = load_model(model_file_name)
    return model

def _zero_padding_3D(X,len):
    result = np.zeros((len,len,X.shape[2]),X.dtype)
    result[:X.shape[0],:X.shape[1],:X.shape[2]] = X
    return result
    
def _cutting_2D(X,len):
    return X[:len,:len]

def predict(model,fasta_file_name,aln_file_name):
    feat_file_name = gen_feat(fasta_file_name,aln_file_name)
    return _predict(model,feat_file_name)

def predict_rr(model,fasta_file_name,aln_file_name,rr_file_name):
    array = predict(model,fasta_file_name,aln_file_name)
    seq = linecache.getline(fasta_file_name, 2).strip().upper()
    _predict_rr(array,seq,rr_file_name)

def _predict_rr(y,seq,rr_file):
    L  = len(seq)
    rr = open(rr_file, 'w')
    rr.write("PFRMAT RR\n")
    rr.write("TARGET T0000\n")
    rr.write("MODEL 0\n")
#    rr.write("AUTHOR UNKNOWN\n")
#    rr.write("METHOD UNKNOWN\n")
    for i in range(len(seq)):
        rr.write(seq[i])
        if i%50 == 49:
            rr.write("\n")
    rr.write("\n")
    out_dict = {}
    for i in range(0, L):
        for j in range(i, L):
            if abs(i - j) < 1:
                continue
            out_dict[str(i+1)+' '+str(j+1)] = (y[i][j] + y[j][i]) / 2.0
    out_list = sorted(out_dict.items(),key=lambda item:item[1])
    out_list.reverse()
    for (k,v) in out_list:
        rr.write("%s 0 8 %.6f\n" %(k,v))
    rr.write("END\n")
    rr.close()

def _predict(model,feat_file_name,MIN_FEAT_SIZE=16):
    base_path = os.path.dirname(os.path.abspath(__file__))
    mean_file_name = base_path + "/stat/mean.txt"
    std_file_name = base_path + "/stat/std.txt"
    mean = np.loadtxt(mean_file_name).reshape(60)
    std = np.loadtxt(std_file_name).reshape(60)
    feature = _load_feat(feat_file_name, mean, std)
    length = feature.shape[0]
    new_length = (int(length / MIN_FEAT_SIZE) + 1) * MIN_FEAT_SIZE
    new_feature = np.array([_zero_padding_3D(feature,new_length)])
    y = model.predict(x=[new_feature], verbose=0).reshape((new_length,new_length))
    new_y = _cutting_2D(y,length)
    return new_y

def _load_feat(feature_file,mean,std):
    L = 0
    with open(feature_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            L = line.strip().split()
            L = int(round(exp(float(L[0]))))
            break
    reject_list = []
    Data = []
    with open(feature_file) as f:
        accept_flag = 1
        index = -1
        for line in f:
            if line.strip() != "":
                if line.startswith('#'):
                    if line.strip() in reject_list:
                        accept_flag = 0
                    else:
                        accept_flag = 1
                    continue
                this_line = line.strip().split()
                if len(this_line) == L:
                    index = index + 2
                else:
                    index = index + 1
                if accept_flag == 0:
                    continue
                if len(this_line) == 1:
                    # 0D feature
                    feature2D = np.zeros((L, L),dtype=np.float32)
                    feature2D[:, :] = float(this_line[0])
                    #Data.append(feature2D)
                    Data.append((feature2D - mean[index])/std[index])
                elif len(this_line) == L:
                    # 1D feature
                    feature2D1 = np.zeros((L, L),dtype=np.float32)
                    feature2D2 = np.zeros((L, L),dtype=np.float32)
                    for i in range (0, L):
                        feature2D1[i, :] = float(this_line[i])
                        feature2D2[:, i] = float(this_line[i])
                    #Data.append(feature2D1)
                    #Data.append(feature2D2)
                    Data.append((feature2D1 - mean[index])/std[index])
                    Data.append((feature2D2 - mean[index])/std[index])
                elif len(this_line) == L * L:
                    # 2D feature
                    feature2D = np.asarray(this_line,dtype=np.float32).reshape(L, L)
                    #Data.append(feature2D)
                    Data.append((feature2D - mean[index])/std[index])
                else:
                    print (line)
                    print ('Error!! Unknown length of feature in !!' + feature_file)
                    print ('Expected length 0, ' + str(L) + ', or ' + str (L*L) + ' - Found ' + str(len(this_line)))
                    sys.exit()
    F = len(Data)
    X = np.zeros((L, L, F),dtype=np.float32)
    for i in range (0, F):
        X[0:L, 0:L, i] = Data[i]
    return X

def gen_feat(fasta_file_name,aln_file_name):
    if not os.path.isfile(aln_file_name):
            raise IOError("Alignment file does not exist.")
    tmpdir = tempfile.mkdtemp() + "/"
    target = fasta_file_name.split("/")[-1].split(".")[0]
    base_path = os.path.dirname(os.path.abspath(__file__))
    alnstats = base_path + "/lib/alnstats"
    os.system("chmod +x " + alnstats)
    os.system("cp " + aln_file_name + " " + tmpdir + "target.aln")
    colstats_file_name = tmpdir + target + ".colstats"
    pairstats_file_name = tmpdir + target + ".pairstats"
    os.system(alnstats + " " + aln_file_name + " " 
                             + colstats_file_name + " "
                             + pairstats_file_name)
    feat_file_name = tmpdir + target + ".feat"
    feat_file = open(feat_file_name,"w")
    ####################################################################################################
    seq = linecache.getline(fasta_file_name, 2).strip().lower()
    feat_file.write("# Sequence Length (log)"+"\n")
    feat_file.write(str(round(log(len(seq)),6))+"\n")
    feat_file.write("# alignment-count (log)"+"\n")
    feat_file.write(str(round(log(int(linecache.getline(colstats_file_name, 2).strip())),6))+"\n")
    feat_file.write("# effective-alignment-count (log)"+"\n")
    feat_file.write(str(round(log(float(linecache.getline(colstats_file_name, 3).strip())),6))+"\n")
    ####################################################################################################
    feat_file.write("# AA composition"+"\n")
    ass = "ACDEFGHIKLMNPQRSTVWY"
    for s in ass:
        feat_file.write(str(round((float(seq.count(s.lower()))/len(seq)),6))+"\n")
    ####################################################################################################
    feat_file.write("# Atchley factors"+"\n")
    factor = {
        "A" : [-0.591,-1.302,-0.733,1.570,-0.146],
        "C" : [-1.343,0.465,-0.862,-1.020,-0.255],
        "D" : [1.050,0.302,-3.656,-0.259,-3.242],
        "E" : [1.357,-1.453,1.477,0.113,-0.837],
        "F" : [-1.006,-0.590,1.891,-0.397,0.412],
        "G" : [-0.384,1.652,1.330,1.045,2.064],
        "H" : [0.336,-0.417,-1.673,-1.474,-0.078],
        "I" : [-1.239,-0.547,2.131,0.393,0.816],
        "K" : [1.831,-0.561,0.533,-0.277,1.648],
        "L" : [-1.019,-0.987,-1.505,1.266,-0.912],
        "M" : [-0.663,-1.524,2.219,-1.005,1.212],
        "N" : [0.945,0.828,1.299,-0.169,0.933],
        "P" : [0.189,2.081,-1.628,0.421,-1.392],
        "Q" : [0.931,-0.179,-3.005,-0.503,-1.853],
        "R" : [1.538,-0.055,1.502,0.440,2.897],
        "S" : [-0.228,1.399,-4.760,0.670,-2.647],
        "T" : [-0.032,0.326,2.213,0.908,1.313],
        "V" : [-1.337,-0.279,-0.544,1.242,-1.262],
        "W" : [-0.595,0.009,0.672,-2.128,-0.184],
        "Y" : [0.260,0.830,3.097,-0.838,1.512]
    }
    for i in range(5):
        factors = np.empty((len(seq)))
        for j in range(len(seq)):
            if seq[j].upper() in factor:
                factors[j] = factor[seq[j].upper()][i]
            else:
                factors[j] = 0
        for j in range(len(seq)):
            feat_file.write(str(round(factors[j],6))+' ')
        feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# Relative sequence separation"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            feat_file.write(str(round(abs(i-j)/float(len(seq)),6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# Sequence separation 5-"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)<5:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation =5"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)==5:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation =6"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)==6:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation =7"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)==7:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation =8"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)==8:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation =9"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)==9:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation =10"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)==10:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation =11"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)==11:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation =12"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)==12:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation =13"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)==13:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation between 14 and 18"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)>=14 and abs(i-j)<18:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation between 18 and 23"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)>=18 and abs(i-j)<23:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation between 23 and 28"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)>=23 and abs(i-j)<28:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation between 28 and 38"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)>=28 and abs(i-j)<38:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation between 38 and 48"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)>=38 and abs(i-j)<48:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    feat_file.write("# Sequence separation 48+"+"\n")
    for i in range(len(seq)):
        for j in range(len(seq)):
            if abs(i-j)>=48:
                feat_file.write("1 ")
            else:
                feat_file.write("0 ")
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# pref score"+"\n")
    def residue_residue_contacts(a,b):
        a = a.upper()
        b = b.upper()
        alphabet = "IVLFCMAGTSWYPHEQDNKR"
        i_index = alphabet.find(a)
        j_index = alphabet.find(b)
        if i_index < 0 or j_index < 0:
            i_index = 10
            j_index = 10
        preference_score = [[.78,1.52,1.37,.82,.23,.50,1.74,1.55,1.09,.87,.32,.77,1.10,.35,.73,.56,.70,.56,.49,.60],
                            [0,1.79,1.93,1.09,.46,.63,2.52,1.82,1.60,1.84,.23,.81,1.56,.52,1.13,.79,.99,.82,1.00,1.01],
                            [0,0,1.80,1.10,.45,.76,2.56,1.78,1.30,1.43,.43,.83,1.38,.74,1.07,.81,.85,.99,.72,1.18],
                            [0,0,0,.62,.27,.38,1.36,1.01,.88,.78,.22,.61,1.04,.27,.51,.49,.39,.60,.40,.53],
                            [0,0,0,0,.43,.11,.61,.59,.33,.59,.06,.18,.47,.20,.30,.16,.21,.17,.18,.23],
                            [0,0,0,0,0,.28,.72,.75,.41,.74,.11,.30,.53,.22,.40,.30,.21,.31,.27,.27],
                            [0,0,0,0,0,0,2.28,2.45,2.03,2.15,.47,1.06,1.95,.83,1.47,1.03,1.52,1.63,1.08,1.10],
                            [0,0,0,0,0,0,0,1.92,2.31,1.98,.43,1.15,1.88,.84,1.16,1.17,1.65,1.40,1.29,1.47],
                            [0,0,0,0,0,0,0,0,1.23,1.82,.42,.42,.74,1.62,.51,1.15,.63,1.71,.92,1.01],
                            [0,0,0,0,0,0,0,0,0,1.47,.32,.78,1.53,.42,1.38,.84,1.76,1.27,.95,1.04],
                            [0,0,0,0,0,0,0,0,0,0,.07,.21,.76,.17,.11,.08,.18,.21,.21,.43],
                            [0,0,0,0,0,0,0,0,0,0,0,.55,.91,.43,.66,.26,.41,.60,.52,.56],
                            [0,0,0,0,0,0,0,0,0,0,0,0,.97,.51,1.18,.89,.94,1.29,.90,1.02],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,.28,.30,.31,.69,.34,.22,.39],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,.56,.42,.46,.79,.87,1.03],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.36,.67,.66,.40,.54],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.55,1.22,.74,1.01],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.93,.59,.74],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.36,.31],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.83]]
        if i_index < j_index:
            return preference_score[i_index][j_index]
        else:
            return preference_score[j_index][i_index]
    for i in range(len(seq)):
        for j in range(len(seq)):
            feat_file.write(str(round((residue_residue_contacts(seq[i],seq[j]) - 0.06)/2.5,6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# scld lu con pot"+"\n")
    def lu_contact_potential(a,b):
        a = a.upper()
        b = b.upper()
        alphabet = "GASCVTIPMDNLKEQRHFYW"
        i_index = alphabet.find(a)
        j_index = alphabet.find(b)
        if i_index < 0 or j_index < 0:
            i_index = 0
            j_index = 16
        contact_potential = [[0.4,0.4,0.7,-1.0,-0.1,0.6,-0.4,0.6,-0.4,1.3,0.7,-0.4,1.1,1.3,0.6,0.1,-0.0,-0.8,-0.8,-1.0],
                            [0,-0.4,0.4,-1.2,-0.7,0.4,-0.9,0.5,-0.9,1.2,0.5,-1.0,1.3,1.1,0.5,-0.0,-0.1,-1.2,-1.0,-0.9],
                            [0,0,0.1,-1.2,-0.2,0.5,-0.6,0.6,-0.2,0.9,0.6,-0.5,1.1,0.8,0.6,0.1,-0.4,-0.9,-0.7,-0.8],
                            [0,0,0,-4.4,-1.9,-0.8,-2.2,-1.0,-2.2,0.0,-0.3,-2.0,-0.2,0.2,-0.8,-1.6,-1.9,-2.4,-2.3,-2.6],
                            [0,0,0,0,-2.0,-0.5,-1.9,-0.3,-1.7,0.4,-0.2,-1.9,0.3,0.4,-0.3,-0.8,-0.9,-2.1,-1.8,-2.0],
                            [0,0,0,0,0,-0.1,-0.8,0.6,-0.5,1.0,0.5,-0.7,0.8,1.0,0.5,0.1,-0.3,-0.9,-0.7,-1.1],
                            [0,0,0,0,0,0,-2.7,-0.6,-2.0,0.3,-0.3,-2.3,-0.0,-0.0,-0.6,-1.1,-1.3,-2.4,-2.0,-2.2],
                            [0,0,0,0,0,0,0,0.1,-0.8,1.3,0.8,-0.8,1.1,1.3,0.4,0.1,0.1,-0.8,-1.0,-1.0],
                            [0,0,0,0,0,0,0,0,-2.9,0.3,-0.4,-2.2,-0.0,-0.0,-0.7,-0.7,-1.3,-2.4,-2.1,-2.6],
                            [0,0,0,0,0,0,0,0,0,1.1,1.0,0.2,0.7,1.9,0.9,-0.2,0.3,0.1,-0.4,-0.2],
                            [0,0,0,0,0,0,0,0,0,0,0.3,-0.3,1.0,1.1,0.6,0.1,0.3,-0.7,-0.5,-0.6],
                            [0,0,0,0,0,0,0,0,0,0,0,-2.7,0.0,-0.0,-0.6,-1.0,-1.1,-2.3,-2.1,-2.4],
                            [0,0,0,0,0,0,0,0,0,0,0,0,1.6,0.8,1.0,0.8,0.7,-0.2,-0.2,-0.2],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,1.2,1.1,0.0,0.2,-0.1,-0.4,0.0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.2,0.1,-0.0,-0.9,-0.6,-1.2],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.6,-0.4,-1.2,-1.3,-1.4],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.3,-1.4,-1.6,-1.9],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3.0,-2.3,-2.5],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2.3,-2.4],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3.3]]
        if i_index < j_index:
            return contact_potential[i_index][j_index]
        else:
            return contact_potential[j_index][i_index]
    for i in range(len(seq)):
        for j in range(len(seq)):
            feat_file.write(str(round((lu_contact_potential(seq[i],seq[j]) + 4.4)/6.3,6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# levitt con pot"+"\n")
    def levitt_contact_potential(a,b):
        a = a.upper()
        b = b.upper()
        alphabet = "GAVLIPDENQKRSTMCYWHF"
        i_index = alphabet.find(a)
        j_index = alphabet.find(b)
        if i_index < 0 or j_index < 0:
            i_index = 0
            j_index = 4
        contact_potential = [[.1,.7,.1,.1,0,.5,.4,.6,.1,0,.4,-0.1,.4,.2,-0.1,-0.1,-0.4,-0.7,0,-0.3],
                             [0,.5,-0.3,-0.4,-0.4,.6,.3,.6,.3,0,1.0,.2,.5,0,-0.5,.3,-0.7,-0.8,0,-0.8],
                             [0,0,1.1,-1.2,-1.2,0,.4,0,0,-0.4,0.1,-0.5,0,-0.3,-1.0,-0.5,-1.2,-1.6,-0.5,-1.5],
                             [0,0,0,-1.4,-1.4,-0.1,0,-0.1,-0.1,-0.6,0.1,-0.6,0,-0.3,-1.3,-0.8,-1.4,-1.7,-0.7,-1.6],
                             [0,0,0,0,-1.5,-0.1,0,-0.2,-0.1,-0.4,0,-0.7,-0.1,-0.6,-1.4,-0.8,-1.4,-1.8,-0.8,-1.7],
                             [0,0,0,0,0,.1,.1,.1,-0.1,-0.3,.6,-0.2,.2,0,-0.5,0,-1.0,-1.3,-0.4,-0.7],
                             [0,0,0,0,0,0,0,0,-0.6,-0.3,-1.0,-1.4,-0.3,-0.3,0.1,0,-1.0,-0.6,-1.1,-0.3],
                             [0,0,0,0,0,0,0,0.1,-0.6,-0.4,-1.1,-1.5,-0.2,-0.3,-0.3,0.1,-1.0,-0.8,-1.0,-0.5],
                             [0,0,0,0,0,0,0,0,-0.7,-0.7,-0.3,-0.8,-0.1,-0.4,-0.3,0,-0.8,-0.8,-0.8,-0.6],
                             [0,0,0,0,0,0,0,0,0,-0.5,-0.4,-0.9,0,-0.5,-0.6,-0.2,-1.1,-1.0,-0.5,-0.8],
                             [0,0,0,0,0,0,0,0,0,0,.7,.1,.1,0,-0.1,.5,-1.0,-0.8,0,-0.4],
                             [0,0,0,0,0,0,0,0,0,0,0,-0.9,-0.4,-0.6,-0.5,0,-1.4,-1.3,-1.0,-0.9],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,-0.2,-0.1,-0.1,-0.6,-0.6,-0.6,-0.4],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,-0.6,-0.3,-0.8,-0.9,-0.7,-0.7],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-0.8,-1.5,-2.0,-0.9,-1.9],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2.7,-0.8,-1.3,-0.6,-1.2],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.6,-1.8,-1.5,-1.7],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2.2,-1.5,-2.0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.6,-1.2],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2.0]]
        if i_index < j_index:
            return contact_potential[i_index][j_index]
        else:
            return contact_potential[j_index][i_index]
    for i in range(len(seq)):
        for j in range(len(seq)):
            feat_file.write(str(round((levitt_contact_potential(seq[i],seq[j]) + 2.7)/3.8,6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# braun con pot"+"\n")
    def braun_contact_potential(a,b):
        a = a.upper()
        b = b.upper()
        alphabet = "GAVLIFYWMCPSTNQHKRDE"
        i_index = alphabet.find(a)
        j_index = alphabet.find(b)
        if i_index < 0 or j_index < 0:
            i_index = 11
            j_index = 0
        contact_potential = [[-0.29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [-0.14,-0.18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [-0.10,-0.15,-0.48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [-0.04,-0.24,-0.29,-0.43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0.27,-0.25,-0.31,-0.45,-0.48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [-0.09,-0.16,-0.31,-0.28,-0.05,-0.50,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [-0.21,-0.18,0.00,-0.10,-0.34,-0.27,-0.11,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [-0.34,-0.01,0.18,-0.18,-0.28,0.16,-0.30,-0.53,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0.25,-0.02,-0.02,-0.32,0.21,-0.36,0.01,-0.73,-0.75,0,0,0,0,0,0,0,0,0,0,0],
                             [-0.42,0.08,0.08,0.36,-0.16,-0.28,0.69,-0.74,0.27,-1.77,0,0,0,0,0,0,0,0,0,0],
                             [0.06,0.28,0.76,0.30,0.99,0.65,-0.02,0.70,-0.78,0.31,-0.78,0,0,0,0,0,0,0,0,0],
                             [0.04,0.38,0.18,0.30,0.57,0.15,-0.03,0.44,0.00,0.12,0.21,-0.68,0,0,0,0,0,0,0,0],
                             [0.28,0.06,0.19,0.57,0.34,0.25,0.23,0.74,0.43,0.28,0.04,-0.23,-0.58,0,0,0,0,0,0,0],
                             [0.49,-0.04,0.48,0.25,1.45,0.12,-0.14,0.46,-0.52,0.07,0.59,-0.21,-0.06,-0.45,0,0,0,0,0,0],
                             [0.54,0.35,0.41,0.35,0.44,-0.04,-0.06,-0.09,0.07,0.39,0.73,0.19,-0.31,0.20,-0.17,0,0,0,0,0],
                             [-0.09,0.44,0.37,0.10,0.24,0.25,0.33,-0.34,1.07,-0.45,-0.21,-0.13,-0.22,-0.56,0.28,-0.15,0,0,0,0],
                             [0.56,0.28,0.53,0.37,-0.00,0.75,-0.00,0.02,0.44,0.68,0.26,-0.05,-0.26,-0.27,0.05,0.57,0.21,0,0,0],
                             [0.40,0.59,0.43,0.37,0.05,0.31,0.03,-0.20,0.53,0.92,0.34,0.24,-0.31,-0.00,0.56,-0.11,0.58,-0.03,0,0],
                             [-0.26,0.24,0.51,0.80,0.26,0.33,0.61,0.74,0.21,0.53,0.87,-0.03,0.32,-0.43,-0.03,-0.61,-0.43,-0.79,0.11,0],
                             [0.21,0.53,0.37,0.51,0.53,0.38,0.25,1.37,0.44,0.17,0.41,0.10,0.27,0.76,-0.20,-0.14,-1.12,-0.85,0.86,0.5]]
        if i_index > j_index:
            return contact_potential[i_index][j_index]
        else:
            return contact_potential[j_index][i_index]
    for i in range(len(seq)):
        for j in range(len(seq)):
            feat_file.write(str(round((braun_contact_potential(seq[i],seq[j]) + 1.77)/3.22,6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# Shannon entropy sum"+"\n")
    def remove_empty(l):
        while '' in l:
            l.remove('')
        return l
    for i in range(len(seq)):
        feat_file.write(str(round(float(remove_empty(linecache.getline(colstats_file_name, i+5).strip().split(" "))[-1]),6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# pstat_pots"+"\n")
    pstat_pots = np.zeros((len(seq),len(seq)))
    with open(pairstats_file_name,"r") as f:
        for line in f.readlines():
            items = remove_empty(line.strip().split(" "))
            if len(items) != 0:
                i = int(items[0]) - 1
                j = int(items[1]) - 1
                d = float(items[2])
                pstat_pots[i,j] = d
                pstat_pots[j,i] = d
    for i in range(len(seq)):
        for j in range(len(seq)):
            feat_file.write(str(round(pstat_pots[i,j],6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# pstat_mimt"+"\n")
    pstat_mimt = np.zeros((len(seq),len(seq)))
    with open(pairstats_file_name,"r") as f:
        for line in f.readlines():
            items = remove_empty(line.strip().split(" "))
            if len(items) != 0:
                i = int(items[0]) - 1
                j = int(items[1]) - 1
                d = float(items[3])
                pstat_mimt[i,j] = d
                pstat_mimt[j,i] = d
    for i in range(len(seq)):
        for j in range(len(seq)):
            feat_file.write(str(round(pstat_mimt[i,j],6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# pstat_mip"+"\n")
    pstat_mip = np.zeros((len(seq),len(seq)))
    with open(pairstats_file_name,"r") as f:
        for line in f.readlines():
            items = remove_empty(line.strip().split(" "))
            if len(items) != 0:
                i = int(items[0]) - 1
                j = int(items[1]) - 1
                d = float(items[4])
                pstat_mip[i,j] = d
                pstat_mip[j,i] = d
    for i in range(len(seq)):
        for j in range(len(seq)):
            feat_file.write(str(round(pstat_mip[i,j],6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.write("# guassdca"+"\n")
    gdca_dict = gaussdca.run(aln_file_name)
    x_i = gdca_dict['gdca_corr']
    for i in range(len(x_i)):
        for j in range(len(x_i)):
            feat_file.write(str(round(x_i[i][j],6))+' ')
    feat_file.write("\n")
    ####################################################################################################
    feat_file.close()
    return feat_file_name

def main():
    pass

if __name__ == '__main__':
    main()
