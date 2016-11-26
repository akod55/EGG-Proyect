import numpy as np
import re
import scipy
from sklearn.ensemble import RandomForestClassifier

from read_files import extract_waves_stage, extract_waves_event
from transfotm_functions import fourier_transform

data_complete =np.load('data_complete_dict.npy').item()

dict_location = {'EOG': 0, 'EEG-F8-O2': 0, 'O2-A1': 0, 'EEG-F7-T3': 0, 'EOG-Left': 0, 'EEG-O2-A1': 0, 'EEG-P3-O1': 0, 'C4-A1': 0, 'EEG-P3-C4': 0, 'EEG-C4-A1': 0, 'EEG-P4-Fp1': 0, 'EEG-T3-T5': 0, 'EMG1-EMG2': 0, 'ECG1-ECG2': 0, 'C3-A2': 0, 'LOC-A2': 0, 'EEG-C4-P4': 0, 'EEG-F4-C4': 0, 'EEG-T4-Fp2': 0, 'EOG-Left-A2': 0, 'EEG-Fp2-F4': 0, 'EEG-C3-O1': 0, 'EOG-Right-A1': 0, 'EEG-T4-T6': 0, 'ROC-LOC': 0, 'EEG-C3-A2': 0, 'LOC-A1': 0, 'ROC-A2': 0, 'EEG-F1-F3': 0, 'EEG-F3-C3': 0, 'CHIN1': 0, 'EEG-F3-A2': 0, 'EEG-F2-F4': 0, 'EEG-C4-F8': 0, 'EEG-C3-P3': 0, 'EEG-P4-O2': 0, 'EEG-Fp1-F3': 0, 'EEG-F3-P3': 0, 'EKG-H-R': 0, 'EEG-Fp2-C3': 0, 'EEG-F8-T4': 0, 'EEG-Fp1-T6': 0}

dict_ROC_LOC = {}
dict_EEG_Fp2_F4 = {}


##################################################################
# Extract data of ROC-LOC and EEG-Fp2-F4 position
##################################################################
# labels = Narcolepsy, Insomnia, No pathology (controls), Sleep-disordered breathing, Nocturnal frontal lobe epilepsy, Periodic leg movements, REM behavior disorder
labels_EEG = {'narco':0, 'ins':1, 'n':2, 'nfle':3, 'plm':4, 'rbd':5, 'sdb':6}

for key in data_complete:
    if "ROC-LOC" in data_complete[key]['Location']:
        dict_ROC_LOC[key] = data_complete[key]
    if "EEG-Fp2-F4" in data_complete[key]['Location']:
        dict_EEG_Fp2_F4[key] = data_complete[key]


def clean_data():
    tmp_dict_ROC_LOC = {}
    min_wave = 2000
    for roc in dict_ROC_LOC:
        waves_ = extract_waves_stage(roc)["ROC-LOC"]
        sss = np.arange(len(waves_[0]))
        tmp_time = []
        tmp_stages = []
        for s in sss:
            tmp_time.append(s)
            if s in waves_[0]:
                tmp_stages.append(waves_[1][s]+1)
            else:
                tmp_stages.append(0)
        tmp_dict_ROC_LOC[roc] = [tmp_time, tmp_stages]
        if len(waves_[0]) < min_wave:
            min_wave = len(waves_[0])
    return tmp_dict_ROC_LOC, min_wave


from sklearn import preprocessing
def transform_data_fourier():
    matrix_to_classify_ROC_data = []
    matrix_to_classify_ROC_label = []

    dict_tmp_ROC_LOC, index_min = clean_data()

    for tmp_key_ROC in dict_tmp_ROC_LOC:
        match = re.match(r"([a-z]+)([0-9]+)", tmp_key_ROC, re.I)
        label = -1
        if match:
            items = match.groups()
            label = labels_EEG[items[0]]
        fourier_transform_data = fourier_transform(dict_tmp_ROC_LOC[tmp_key_ROC][1][:index_min], 11)[1]

        matrix_to_classify_ROC_data.append(fourier_transform_data)
        matrix_to_classify_ROC_label.append(label)

    # normalizer = preprocessing.Normalizer().fit(matrix_to_classify_ROC_data)

    return matrix_to_classify_ROC_data, matrix_to_classify_ROC_label # , normalizer


fourier_matrix = []
X, y = transform_data_fourier()

from sklearn import svm, cross_validation
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

clf = AdaBoostClassifier(n_estimators=20)
clf_1 = svm.SVC(kernel='linear', C=0.5)
clf_2 = RandomForestClassifier(n_estimators=20)

print np.mean(cross_validation.cross_val_score(clf, X, y, cv=10))
print np.mean(cross_validation.cross_val_score(clf_1, X, y, cv=10))
print np.mean(cross_validation.cross_val_score(clf_2, X, y, cv=10))


"""
C_s = np.logspace(-5, 5, 10)
scores = list()

for C in C_s:
    clf_1.C = C
    this_scores = cross_validation.cross_val_score(clf_1,  X, y)
    scores.append(np.mean(this_scores))

import matplotlib.pyplot as plt

plt.figure(1, figsize=(10, 5))
plt.clf()

plt.semilogx(C_s, scores)
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.show()


"""

"""
import matplotlib.pyplot as plt

plt.figure(1, figsize=(10, 5))
plt.clf()

##############################################
# Plot Real song and his Fourier transform
##############################################
frq = fourier_matrix[2][0]
Y = fourier_matrix[2][1]
ax = plt.subplot(2,2,1)
ax.set_title("Fourier Transformation for the Real Function")
ax.plot(frq, abs(Y))



##############################################
# Plot first Harr Level function and his Fourier transform
##############################################
frq = fourier_matrix[10][0]
Y = fourier_matrix[10][1]
ax = plt.subplot(2,2,2)
ax.set_title("Fourier Transformation for the Level 1")
plt.plot(frq, abs(Y))



##############################################
# Plot Second Harr Level function and his Fourier transform
##############################################
frq = fourier_matrix[30][0]
Y = fourier_matrix[30][1]
ax = plt.subplot(2,2,3)
ax.set_title("Fourier Transformation for the Level 2")
plt.plot(frq, abs(Y))


##############################################
# Plot Third Harr Level function and his Fourier transform
##############################################
frq = fourier_matrix[50][0]
Y = fourier_matrix[50][1]
ax = plt.subplot(2,2,4)
ax.set_title("Fourier Transformation for the Level 3")
plt.plot(frq, abs(Y))


# Show the graphs
plt.tight_layout(pad=0.2)
plt.show()

"""




################################################################
# Dream Classification
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


########################################
# Use kernel method to R^2 -> R
########################################
def Gaussian(x,z,sigma=1,axis=None):
    return np.exp((-(np.linalg.norm(x-z, axis=axis)**2))/(2*sigma**2))

def kernel_dot(x,y):
    return x*y

def kernel_dot_ex(x,y, exponent=2):
    return (x*y+1)**exponent

def kernel_triweight(x,z, axis=None):
    return (3.0/4.0)*(1.0-(np.linalg.norm(x - z, axis=axis) ** 2))

################################################################
# Transform data using kernels and normalization
################################################################
from sklearn import preprocessing
def transform_data(kernel):
    matrix_to_classify_ROC_data = []
    matrix_to_classify_ROC_label = []
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
            tuples_data.append(kernel(ind_i, indj))

        matrix_to_classify_ROC_data.append(tuples_data)
        matrix_to_classify_ROC_label.append(label)

    print matrix_to_classify_ROC_data[0]
    normalizer = preprocessing.Normalizer().fit(matrix_to_classify_ROC_data)

    return normalizer.transform(matrix_to_classify_ROC_data), matrix_to_classify_ROC_label, normalizer