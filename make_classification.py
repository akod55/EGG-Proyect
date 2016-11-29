import numpy as np
import re

from read_files import extract_waves_stage, extract_waves_event
from transfotm_functions import fourier_transform

data_complete =np.load('data_test_dict.npy').item()

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
def transform_data(kernel):

    """
    TO FIND THE INDEX THAT TAKE MORE INFORMATION
    tmp__ = sorted(nsmallest(3, tmp_waves, key=lambda x: abs(x - min_posible_number_time)))[2]
    tmp_waves.index(tmp__), key_ROC
    """
    min_possible_number_index = 436

    matrix_to_classify_ROC_data = []
    matrix_to_classify_ROC_label = []
    for key_ROC in dict_ROC_LOC:
        match = re.match(r"([a-z]+)([0-9]+)", key_ROC, re.I)
        label = -1
        if match:
            items = match.groups()
            label = labels_EEG[items[0]]
        waves = extract_waves_stage(key_ROC)["ROC-LOC"]
        time_waves = waves[0][:min_possible_number_index]
        events_waves = waves[1][:min_possible_number_index]

        tuples_data = []
        for (ind_i, indj) in zip(time_waves, events_waves):
            tuples_data.append(kernel(ind_i, indj))

        matrix_to_classify_ROC_data.append(tuples_data)
        matrix_to_classify_ROC_label.append(label)

    normalizer = preprocessing.Normalizer().fit(matrix_to_classify_ROC_data)

    return normalizer.transform(matrix_to_classify_ROC_data), matrix_to_classify_ROC_label, normalizer


################################################################
# Transform data using Fourier transform
################################################################
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
    return tmp_dict_ROC_LOC, 300


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

        matrix_to_classify_ROC_data.append(np.array(fourier_transform_data).real)
        matrix_to_classify_ROC_label.append(label)

    # normalizer = preprocessing.Normalizer().fit(matrix_to_classify_ROC_data)

    return matrix_to_classify_ROC_data, matrix_to_classify_ROC_label # , normalizer


#########################################################
# Load classifiers
#########################################################

import cPickle
with open('AdaBoost_kernel_classifier.pkl', 'rb') as fid:
    clf_kernel_AdaBoost = cPickle.load(fid)

with open('SVM_kernel_classifier.pkl', 'rb') as fid:
    clf_kernel_SVM = cPickle.load(fid)

with open('RandomForest_kernel_classifier.pkl', 'rb') as fid:
    clf_kernel_Forest = cPickle.load(fid)

with open('AdaBoost_fourier_classifier.pkl', 'rb') as fid:
    clf_fourier_AdaBoost = cPickle.load(fid)

with open('SVM_fourier_classifier.pkl', 'rb') as fid:
    clf_fourier_SVM = cPickle.load(fid)

with open('RandomForest_fourier_classifier.pkl', 'rb') as fid:
    clf_fourier_Forest = cPickle.load(fid)



#################################################################
# Make classification using kernels and normalization.
#################################################################
from sklearn import svm, cross_validation
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier

X_kernel, y_kernel, normalizer_function = transform_data(kernel_triweight)
X_fourier, y_fourier = transform_data_fourier()

# print X_kernel

print "Predicted using kernel: ", clf_kernel_Forest.predict(X_kernel)
print "Predicted using fourier: ", clf_fourier_Forest.predict(X_fourier)
print
print "Real class: ", y_fourier
#clf_kernel_AdaBoost = AdaBoostClassifier(n_estimators=50).fit(X_kernel, y_kernel)
#clf_kernel_SVM = svm.SVC(kernel='linear', C=1).fit(X_kernel, y_kernel)
#clf_kernel_Forest = RandomForestClassifier(n_estimators=50).fit(X_kernel, y_kernel)

# Score using crossvalidation
# print "Score using AdaBoost kernel: ", np.mean(cross_validation.cross_val_score(clf_kernel_AdaBoost, X, y, cv=10))
# print "Score using SVM kernel: ", np.mean(cross_validation.cross_val_score(clf_kernel_SVM, X, y, cv=10))
# print "Score using Random Forest kernel: ", np.mean(cross_validation.cross_val_score(clf_kernel_Forest, X, y, cv=10))
# print


#################################################################
# Make classification using kernels..
#################################################################
from sklearn import svm, cross_validation
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

#clf_fourier_AdaBoost = AdaBoostClassifier(n_estimators=50).fit(X_fourier, y_fourier)
#clf_fourier_SVM = svm.SVC(kernel='linear', C=0.5).fit(X_fourier, y_fourier)
#clf_fourier_Forest = RandomForestClassifier(n_estimators=50).fit(X_fourier, y_fourier)

# print "Score using AdaBoost fourier: ", np.mean(cross_validation.cross_val_score(clf_fourier_AdaBoost, X, y, cv=10))
# print "Score using SVM fourier: ", np.mean(cross_validation.cross_val_score(clf_fourier_SVM, X, y, cv=10))
# print "Score using Random Forest fourier: ",np.mean(cross_validation.cross_val_score(clf_fourier_Forest, X, y, cv=10))

#########################################################
# Save classifiers
#########################################################
"""
import cPickle
with open('AdaBoost_kernel_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf_kernel_AdaBoost, fid)

with open('SVM_kernel_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf_kernel_SVM, fid)

with open('RandomForest_kernel_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf_kernel_Forest, fid)

with open('AdaBoost_fourier_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf_fourier_AdaBoost, fid)

with open('SVM_fourier_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf_fourier_SVM, fid)

with open('RandomForest_fourier_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf_fourier_Forest, fid)
"""

