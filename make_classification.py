import numpy as np
import re
import scipy

from read_files import extract_waves_stage

data_complete =np.load('data_complete_dict.npy').item()

dict_location = {'EOG': 0, 'EEG-F8-O2': 0, 'O2-A1': 0, 'EEG-F7-T3': 0, 'EOG-Left': 0, 'EEG-O2-A1': 0, 'EEG-P3-O1': 0, 'C4-A1': 0, 'EEG-P3-C4': 0, 'EEG-C4-A1': 0, 'EEG-P4-Fp1': 0, 'EEG-T3-T5': 0, 'EMG1-EMG2': 0, 'ECG1-ECG2': 0, 'C3-A2': 0, 'LOC-A2': 0, 'EEG-C4-P4': 0, 'EEG-F4-C4': 0, 'EEG-T4-Fp2': 0, 'EOG-Left-A2': 0, 'EEG-Fp2-F4': 0, 'EEG-C3-O1': 0, 'EOG-Right-A1': 0, 'EEG-T4-T6': 0, 'ROC-LOC': 0, 'EEG-C3-A2': 0, 'LOC-A1': 0, 'ROC-A2': 0, 'EEG-F1-F3': 0, 'EEG-F3-C3': 0, 'CHIN1': 0, 'EEG-F3-A2': 0, 'EEG-F2-F4': 0, 'EEG-C4-F8': 0, 'EEG-C3-P3': 0, 'EEG-P4-O2': 0, 'EEG-Fp1-F3': 0, 'EEG-F3-P3': 0, 'EKG-H-R': 0, 'EEG-Fp2-C3': 0, 'EEG-F8-T4': 0, 'EEG-Fp1-T6': 0}

dict_ROC_LOC = {}
dict_EEG_Fp2_F4 = {}

# labels = Narcolepsy, Insomnia, No pathology (controls), Sleep-disordered breathing, Nocturnal frontal lobe epilepsy, Periodic leg movements, REM behavior disorder
labels_EEG = {'narco':0, 'ins':1, 'n':2, 'sdb':3, 'nfle':4, 'plm':5, 'rbd':6}


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

for key_ROC in dict_ROC_LOC:
    match = re.match(r"([a-z]+)([0-9]+)", key_ROC, re.I)
    label = -1
    if match:
        items = match.groups()
        label = labels_EEG[items[0]]
    tmp_time_ = extract_waves_stage(key_ROC)["ROC-LOC"][0][:437]
    tmp_waves = extract_waves_stage(key_ROC)["ROC-LOC"][1][:437]

    tuples_data = [(tmp_time_[ind], tmp_waves[ind]) for ind in xrange(len(tmp_waves))]
    matrix_to_classify_ROC_data.append(label)
    matrix_to_classify_ROC_label.append(label)


from sklearn import svm
X = matrix_to_classify_ROC_data
y = matrix_to_classify_ROC_label
clf = svm.SVC()
clf.fit(X, y)




for p in matrix_to_classify_ROC:
    print p


"""
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