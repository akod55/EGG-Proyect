# Data files in a directory (next(os.walk("path"))[2])
from read_files import plot_waves
import os

all_data_file_test = next(os.walk("/home/akosoriod/Documentos/Python/EGG-Proyect/TestBaseTXT"))[2]
#all_data_file = next(os.walk("C:\Users\Usuario\Documents\Python\EGG-Proyect\DataBaseTXT"))[2]
#globlas
heads = ['Sleep Stage', 'Time [hh:mm:ss]', 'Event', 'Duration[s]', 'Location']
# stages
stages = ['S3', 'S2', 'S1', 'S4', 'MT', 'W', 'REM']
# events
events = ['SLEEP-S3', 'SLEEP-S2', 'SLEEP-S1', 'SLEEP-S0', 'SLEEP-S4', 'SLEEP-REM', 'SLEEP-MT', 'MCAP-A1', 'MCAP-A2', 'MCAP-A3']

def get_test_data():
    all_data = {}
    for i in all_data_file_test:
        #tmp = open("C:\Users\Usuario\Documents\Python\EGG-Proyect\DataBaseTXT/" + i).readlines()
        tmp = open("/home/akosoriod/Documentos/Python/EGG-Proyect/TestBaseTXT/" + i).readlines()
        for line in xrange(len(tmp)):
            if "Sleep Stage" in tmp[line]:
                all_data[i] = tmp[line:]
                break
    return all_data

# Clear data, split the data by tabs
def clear_test_data_into_list(all_data):
    clear_data = {}
    for dat in all_data:
        data_list = []
        for o in all_data[dat]:
            # Split the string by tabs.
            tmp_dat = "".join(o).strip().split("\t")

            # Add the data into a list
            if len(tmp_dat) == 6:
                data_list.append([tmp_dat[0]] + tmp_dat[2:])
            else:
                data_list.append(tmp_dat)
        data_list.pop(0)
        clear_data[dat] = data_list

    return clear_data
