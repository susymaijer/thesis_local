import os 
import zipfile
import matplotlib.pyplot as plt
import numpy as np

'''
    Various functions for general 
'''

def get_patient_id_from_rel_id(img_path, rel_patient_id):
    ''' retrieve scan id, given a folder and the relative patient id in that folder '''
    return os.listdir(img_path)[rel_patient_id].split(".")[0].split("_")[1]

def create_maybe_dir(path):
    ''' create a folder if it does not exist'''
    if not(os.path.exists(path)):
        print("Created new dir" + path)
        os.mkdir(path)

def unzip_all_folders(path):
    ''' unzip all zip files in a given folder '''
    for zip in os.listdir(path):
        if zip.endswith(".zip") and not(zip[:-4] in path):
                with zipfile.ZipFile(os.path.join(path, zip), 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(path, zip)[:-4])
        else:
            print("Not unzipped: " + zip)

def get_plot_layout(doubleAxis=False):
    ''' Make the generic plot layout we want for each plot. '''

    # create figure
    fig = plt.figure(facecolor='w', figsize=(30, 24))

    # set colors
    ax = fig.add_subplot(111, facecolor='#f1f1f1', axisbelow=True)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

    # make grid
    ax.grid(b=True, which='major', c='w', lw=1, ls='-')

    # remove border
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    # two y-axises 
    if doubleAxis:
        ax2 = ax.twinx()
        return fig, ax, ax2
    else:
        return fig, ax

def convert_string_array_to_list(string_array, makeInt=True):
    ''' create a real array from a string imitating an array '''
    array = string_array[1:-1].split(",")
    if makeInt:
        return [float(x) for x in array]
    else:
        return array

def create_filename(file_name, useModality=False, modality=None):
    ''' create filename similar to nih and following nnUnet dataformat (i.e., "panc_XXXX")'''
    file_name = "panc_" + file_name.split("_")[1]
    
    # also add modality to filename (necessary for imnages)
    if useModality:
        file_name = file_name.split(".")
        file_name = f"{file_name[0]}_{modality}.{file_name[1]}.{file_name[2]}"
    return file_name