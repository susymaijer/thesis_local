import numpy as np
import pandas as pd
import os
import json
from math import *
from functools import reduce
import math

# ipywidgets for some interactive plots
from ipywidgets.widgets import * 

# plotly 3D interactive graphs 
from plotly.graph_objs import *

# medical file imaging 
import nibabel as nib

# scientici imaging
from skimage.measure import *

# custom
from util import management as mana

'''
    Various functions for retrieving, evaluating
'''

def get_filename(patient_id, modality= None):
    ''' create a filename based on patient id '''
    if modality:
        filename = f'/panc_{patient_id}_{modality}.nii.gz'
    else:
        filename = f'/panc_{patient_id}.nii.gz'
    return filename

def get_results_validation_and_test_paths(results_dir, task, config, trainer):
    ''' given nnU-Net model specifations, retrieve the paths to the validation cases results and the held-out test set results'''
    validation_path = f"{results_dir}\\{task}\\{config}\\{trainer}\\cv_niftis_postprocessed"
    test_path = f"{results_dir}\\{task}\\{config}\\{trainer}\\inference\\{task}\\imagesTs"
    return validation_path, test_path

def get_file(img_path, patient_id, modality = None):
    ''' given a scan path, load nibabel object '''
    filename = get_filename(patient_id, modality)
    return nib.load(img_path + filename)

def get_3d_img(img_path, patient_id, modality = None):
    ''' given a scan path, load the voxel data'''
    return np.array(get_file(img_path, patient_id, modality).dataobj)

def get_all_data(patient_id, img_path, label_path, modality, seg_path=None):
    ''' 
        given a scan path, load the voxels of
        1) the scan
        2) the label
        3) the segmentation (optional)
    '''
    img = get_3d_img(img_path, patient_id, modality)
    label = get_3d_img(label_path, patient_id)
    if seg_path is not None:
        seg = get_3d_img(seg_path, patient_id)
        return img, label, seg
    return img, label
    
def get_dice(gt, seg, l=1):
    ''' given a ground truth and segmentation, get the DICE score of the segmentation'''
    return np.sum(seg[gt==l])*2.0 / (np.sum(seg) + np.sum(gt))

def get_patient_dices(img, label, segmentation):
    ''' given scan information, evaluate each 2D slice with dice metric '''
    dices = {}
    for sl_id in range(0, img.shape[2]):
        # create GT mask and plot over dicom
        gt = label[:,:,sl_id] # GT
        seg = segmentation[:,:,sl_id]
        # no pancreas 
        if not(has_slice_pancreas(gt)) and not(has_slice_pancreas(seg)):
            dices[sl_id] = np.nan
        # either GT or segmentation contains pancreas
        else:
            dice = get_dice(gt, seg)
            dices[sl_id] = dice

    return dices

def get_gt_img_slice(img, label, idx):
    ''' shorthand function for retrieving scan and label slices from 3D voxel arrays'''
    gt = label[:,:,idx] # GT
    img_slice = img[:,:,idx] # dicom
    return gt, img_slice

def get_gt_img_slice_seg_dice(img, label, segmentation, idx):
    ''' shorthand function for retrieving scan, label, segmentation slices from 3D voxel arrays AND the dice score '''
    gt, img_slice = get_gt_img_slice(img, label, idx)
    seg = segmentation[:,:,idx]
    dice = get_dice(gt, seg)
    return gt, img_slice, seg, dice

def get_worst_best(dices):
    ''' given an array of dices, get the worst and best dice '''
    worst = np.nanmin(np.array(list(dices.values())))
    best = np.nanmax(np.array(list(dices.values())))
    worst_idx = [k for k, v in dices.items() if v == worst]
    best_idx = [k for k, v in dices.items() if v == best]
    return worst, best, worst_idx, best_idx
    
def has_slice_pancreas(mask):
    ''' check whether the slice contains a label '''
    return (mask != 0).any()

def load_summary_data(path):
    ''' load the summary.json from a given folder'''
    summary = open(os.path.join(path, "summary.json"))
    return json.load(summary)["results"]["all"]

def load_summary_dices_data_array(path):
    ''' get all the dices from the summary.json from a given folder'''
    summary = load_summary_data(path)

    # Load data
    dices = []
    for val in summary:
        panc = val["1"] # pancreas label == 1
        dices.append(panc["Dice"])
    return np.array(dices)

#################### CROPPING FUNCTIONS

def get_all_data_dictionary(img_path, label_path, modality, seg_path=None, doPrint=False):
    filenames = os.listdir(img_path)
    labels, imgs, labels_niffs, img_niffs = {}, {}, {}, {}
    if seg_path is not None:
        segs = {}

    for i in range(len(filenames)):

        # Get filename info        
        filename = filenames[i]
        patient_id = mana.get_patient_id_from_rel_id(img_path, i)
        if doPrint:
            print(f"Loading data for patient {patient_id}")

        # get numpy data and save in dicts
        data = get_all_data(patient_id, img_path, label_path, modality, seg_path)
        imgs[filename] = data[0]
        labels[filename] = data[1]
        if seg_path is not None:
            segs[filename] = data[2]

        # get affine and header info
        img_niff = get_file(img_path, patient_id, modality)
        label_nif = get_file(label_path, patient_id)
        img_niffs[filename] = img_niff
        labels_niffs[filename] = label_nif    

    if seg_path is not None:
        return filenames, labels, segs, imgs, labels_niffs, img_niffs 
    return filenames, labels, imgs, labels_niffs, img_niffs 
    
def get_bounding_box(data, minX, minY, minZ, maxX, maxY, maxZ):
    return data[minX: maxX, minY : maxY, minZ : maxZ]
    
def determine_minimal_bounding_box(lab):
    lab_bboxes = regionprops(lab)
    minX, minY, minZ = 99999, 99999, 99999
    maxX, maxY, maxZ = -9999, -9999, -9999
    for bbox in lab_bboxes:
        minX0, minY0, minZ0, maxX0, maxY0, maxZ0 = bbox.bbox
        if minX0 < minX:
            minX = minX0
        if minY0 < minY:
            minY = minY0
        if minZ0 < minZ:
            minZ = minZ0
        if maxX0 > maxX:
            maxX = maxX0
        if maxY0 > maxY:
            maxY = maxY0
        if maxZ0 > maxZ:
            maxZ = maxZ0
    return minX, minY, minZ, maxX, maxY, maxZ