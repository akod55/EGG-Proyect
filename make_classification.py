import numpy as np
import re
import scipy

from read_files import extract_waves_stage, extract_waves_event

data_complete =np.load('data_complete_dict.npy').item()

dict_location = {'EOG': 0, 'EEG-F8-O2': 0, 'O2-A1': 0, 'EEG-F7-T3': 0, 'EOG-Left': 0, 'EEG-O2-A1': 0, 'EEG-P3-O1': 0, 'C4-A1': 0, 'EEG-P3-C4': 0, 'EEG-C4-A1': 0, 'EEG-P4-Fp1': 0, 'EEG-T3-T5': 0, 'EMG1-EMG2': 0, 'ECG1-ECG2': 0, 'C3-A2': 0, 'LOC-A2': 0, 'EEG-C4-P4': 0, 'EEG-F4-C4': 0, 'EEG-T4-Fp2': 0, 'EOG-Left-A2': 0, 'EEG-Fp2-F4': 0, 'EEG-C3-O1': 0, 'EOG-Right-A1': 0, 'EEG-T4-T6': 0, 'ROC-LOC': 0, 'EEG-C3-A2': 0, 'LOC-A1': 0, 'ROC-A2': 0, 'EEG-F1-F3': 0, 'EEG-F3-C3': 0, 'CHIN1': 0, 'EEG-F3-A2': 0, 'EEG-F2-F4': 0, 'EEG-C4-F8': 0, 'EEG-C3-P3': 0, 'EEG-P4-O2': 0, 'EEG-Fp1-F3': 0, 'EEG-F3-P3': 0, 'EKG-H-R': 0, 'EEG-Fp2-C3': 0, 'EEG-F8-T4': 0, 'EEG-Fp1-T6': 0}

dict_ROC_LOC = {}
dict_EEG_Fp2_F4 = {}

# labels = Narcolepsy, Insomnia, No pathology (controls), Sleep-disordered breathing, Nocturnal frontal lobe epilepsy, Periodic leg movements, REM behavior disorder
labels_EEG = {'narco':0, 'ins':1, 'n':2, 'nfle':3, 'plm':4, 'rbd':5, 'sdb':6}


for key in data_complete:
    if "ROC-LOC" in data_complete[key]['Location']:
        dict_ROC_LOC[key] = data_complete[key]
    if "EEG-Fp2-F4" in data_complete[key]['Location']:
        dict_EEG_Fp2_F4[key] = data_complete[key]

    match = re.match(r"([a-z]+)([0-9]+)", key, re.I)
    if match:
        items = match.groups()


################################################################
# Vamos a hacer la clasificacion por los eventos del sueno
################################################################

min_data_text = 'plm6.txt'
posible_roc = []
"""
TO FIND THE MIN_NUMBER OF TIME
tmp_waves = extract_waves_stage(key_ROC)["ROC-LOC"][0]
if i_add in tmp_waves:
    index_check+=1

tmp_min_waves = extract_waves_stage(min_data_text)["ROC-LOC"][0]

print len(set(tmp_waves).intersection(tmp_waves))
"""
min_posible_number_time = 931

"""
TO FIND THE INDEX THAT TAKE MORE INFORMATION
tmp__ = sorted(nsmallest(3, tmp_waves, key=lambda x: abs(x - min_posible_number_time)))[2]
tmp_waves.index(tmp__), key_ROC
"""
min_possible_number_index = 436


#from heapq import nsmallest

# min(myList, key=lambda x:abs(x-myNumber))


matrix_to_classify_ROC_data = []
matrix_to_classify_ROC_label = []

def Gaussian(x,z,sigma=1,axis=None):
    return np.exp((-(np.linalg.norm(x-z, axis=axis)**2))/(2*sigma**2))

def kernel_dot(x,y):
    return x*y


def kernel_dot_ex(x,y, exponent):
    return (x*y+1)**exponent

def kernel_triweight(x,z, axis=None):
    return (3.0/4.0)*(1.0-(np.linalg.norm(x - z, axis=axis) ** 2))

for key_ROC in dict_ROC_LOC:
    match = re.match(r"([a-z]+)([0-9]+)", key_ROC, re.I)
    label = -1
    if match:
        items = match.groups()
        label = labels_EEG[items[0]]
    waves = extract_waves_stage(key_ROC)["ROC-LOC"]
    time_waves = waves[0][:436]
    events_waves = waves[1][:436]

    tuples_data = []
    for (ind_i, indj) in zip(time_waves, events_waves):
        tuples_data.append(Gaussian(ind_i, indj))

    matrix_to_classify_ROC_data.append(tuples_data)
    matrix_to_classify_ROC_label.append(label)


from sklearn import svm, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier


scores = list()

y = np.array(matrix_to_classify_ROC_label, dtype=float)
X = np.array(matrix_to_classify_ROC_data, dtype=float)

print X
print y.shape

from sklearn import linear_model


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
alphas = np.arange(0.01, 2, 0.1)
#for k in alphas:
# clf = svm.SVC(C=0.001)
clf = AdaBoostClassifier(n_estimators=100)

    #for C in C_s:
    #    clf.C = C
this_scores = cross_validation.cross_val_score(clf, X, y, cv=10)

print np.mean(this_scores)
"""
import matplotlib.pyplot as plt

plt.figure(1, figsize=(10, 5))
plt.clf()
plt.plot(alphas, scores)
#locs, labels = plt.yticks()
#plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.show()



# Refactor data ROC_LOC
print len(dict_ROC_LOC.keys())
index_to_add = []
total_index = 0
for i_add in xrange(1100):
    index_check = 0
    for key_ROC in dict_ROC_LOC:
        tmp_waves = extract_waves_stage(key_ROC)["ROC-LOC"][0]
        if i_add in tmp_waves:
            index_check+=1

        tmp_min_waves = extract_waves_stage(min_data_text)["ROC-LOC"][0]

        print len(set(tmp_waves).intersection(tmp_waves))

    print i_add, len(dict_ROC_LOC.keys()), index_check
    if index_check > 50:
        total_index += 1

print total_index

"""

"""
matrix_to_classify_ROC = []

posible_roc = []

min_ = 5000
min_dat = ""
for key_ROC in dict_ROC_LOC:
    match = re.match(r"([a-z]+)([0-9]+)", key_ROC, re.I)
    label = -1
    if match:
        items = match.groups()
        label = labels_EEG[items[0]]
    if len(extract_waves_stage(key_ROC)["ROC-LOC"][0]) < min_:
        min_dat = key_ROC
        min_ = len(extract_waves_stage(key_ROC)["ROC-LOC"][0])

print min_dat
print extract_waves_stage(min_dat)["ROC-LOC"][0]




print extract_waves_stage('nfle24.txt')
#print extract_waves_event('nfle24.txt')
#print dict_EEG_Fp2_F4.keys()
#print labels


#plot_waves_comparison("nfle1.txt", "n10.txt", "rbd14.txt")
"""