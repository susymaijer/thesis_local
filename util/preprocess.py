import os
import shutil
import random
import numpy as np 
import nibabel as nib
from collections import OrderedDict
import json
from util import management as mana

''' Various functions used by the notebooks that create the data-sets in nnu-net format '''

def create_all_folders(tr_data, tr_lab, ts_data, ts_lab):
    ''' creates following folders: imagesTr, labelstr, imagesTs, labelsTs '''
    os.makedirs(tr_data, exist_ok=True)
    os.makedirs(tr_lab, exist_ok=True)
    os.makedirs(ts_data, exist_ok=True)
    os.makedirs(ts_lab, exist_ok=True)

def get_all_files(path):
    ''' get the paths to all the files, given the folder '''
    return [os.path.join(path, i) for i in os.listdir(path)]

def get_filename_modality(x):
    ''' retrieve filename modality. is standard 0000, but can be overwritten in the notebooks themselves '''
    return "0000"

def create_images(tr, out_tr_dir, out_ts_dir, test_perc, get_all_files=get_all_files, get_filename_modality=get_filename_modality, create_filename=mana.create_filename):
    ''' populate both imagesTr and imagesTs '''
    images = get_all_files(tr)
    
    # get files which go in the test set
    if isinstance(test_perc, list):
        test_filenames = test_perc
    else: 
        test_size = int(test_perc * len(images))
        test_filenames = [i.split("\\")[-1] for i in random.sample(images, k=test_size)]
    print(f"Create test set of {len(test_filenames)} images (total: {len(images)})")

    # move all the files to their correct locations
    for i in images:
        file_name = i.split("\\")[-1]

        # move files
        if file_name in test_filenames:    
            # test
            modality = get_filename_modality(file_name)
            shutil.copy(i, os.path.join(out_ts_dir, create_filename(file_name, True, modality)))
        else:
            # training
            modality = get_filename_modality(file_name)
            shutil.copy(i, os.path.join(out_tr_dir, create_filename(file_name, True, modality)))

    return test_filenames

def is_test_file(filenames, filename):
    ''' default function for checking whether a certain scan is in the test set. can be overwritten in the notebooks'''
    return filename in filenames
    
def create_labels(test_filenames, tr, out_tr_dir, out_ts_dir, labels, get_all_files=get_all_files, create_filename=mana.create_filename, is_test_file=is_test_file, panc_label=None):
    ''' populate both labelsTr and labelsTs  '''

    print("Creating mask for organ label values: " + str(labels))
    # get the paths to all the files
    label_paths = get_all_files(tr)

    # create label
    for label_path in label_paths:
        file_name = label_path.split("\\")[-1]

        # load label niifti and get the data
        label_nif = nib.load(label_path)
        label = np.array(label_nif.dataobj)

        # filter labels, if necessary
        if labels != None:
            # combine all masks so we only keep the relevant labels
            masks = []
            for ol in labels:
                mask = label != int(ol)
                masks.append(mask)
            final_mask = np.logical_and.reduce(masks)
            lab_masked = np.ma.masked_array(label, mask=final_mask, fill_value=0)
            label = np.ma.filled(lab_masked)

            # replace by dummy value because otherwise the original '1' label gets lost if we translate
            # panc_label to '1' immediately
            DUMMY = 255
            if panc_label is not None: 
                print(f'Replace {panc_label} by {DUMMY}')
                print(sum(sum(sum(label == panc_label))))
                np.place(label, label == panc_label, DUMMY)
                print(sum(sum(sum(label == panc_label))))

            # replace all non-pancreas label values so they're in incremental order
            for i, ol in enumerate(labels[2:][::-1]):  # we know 0=background and 1=pancreas
                ol = int(ol)
                print(f'Replace ol by {i+2}')
                np.place(label, label == ol, len(labels)-i-1)

            # make pancreas the '1' label 
            if panc_label is not None: 
                print(f'Replace {DUMMY} by 1')
                np.place(label, label == DUMMY, 1)
                print(sum(sum(sum(label == 1))))

        new_label = nib.Nifti1Image(label, label_nif.affine, label_nif.header)

        # write label
        if is_test_file(test_filenames, file_name):    
            # test
            nib.save(new_label, os.path.join(out_ts_dir, create_filename(file_name)))
        else:
            # training
            nib.save(new_label, os.path.join(out_tr_dir, create_filename(file_name)))

def generate_dataset_json(overwrite_json_file, reference, task_dir, task, modality, labels,
                            tr_label_dir, ts_label_dir):
    ''' make the dataset.json '''
    json_file_exist = False
    json_path = os.path.join(task_dir, 'dataset.json')

    if os.path.exists(json_path):
        print(f'dataset.json already exist! {json_path}')
        json_file_exist = True

    if json_file_exist==False or overwrite_json_file:

        json_dict = OrderedDict()
        json_dict['name'] = f"Task{task}"
        json_dict['description'] = task
        json_dict['tensorImageSize'] = "3D"
        json_dict['reference'] = reference
        json_dict['licence'] = reference
        json_dict['release'] = "0.0"
        json_dict['modality'] = modality
        json_dict['labels'] = labels
        
        train_ids = os.listdir(tr_label_dir)
        test_ids = os.listdir(ts_label_dir)
        json_dict['numTraining'] = len(train_ids)

        #no modality in train image and labels in dataset.json 
        json_dict['training'] = [{'image': f"./imagesTr/{i}", "label": f"./labelsTr/{i}"} for i in train_ids]
        json_dict['test'] = [f"./imagesTs/{i}" for i in test_ids]
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)