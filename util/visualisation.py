import matplotlib.pyplot as plt
import numpy as np 

# skimage image processing packages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from util import data

def show_slice(slice):
    # show label single slice
    f = plt.figure(figsize=(30,30))
    a1 = f.add_subplot(2, 2, 1)
    plt.imshow(slice, cmap=plt.cm.gray)
    plt.show()

def show_2d_slices_all_views_from_img3d(img3d):
    plt.figure(figsize = (15,15))
    img_shape = img3d.shape

    # plot 3 orthogonal slices from the middle
    a1 = plt.subplot(1, 3, 1)
    plt.imshow(img3d[:, :, img_shape[2]//2], cmap=plt.cm.gray)

    a2 = plt.subplot(1, 3, 2)
    plt.imshow(img3d[:, img_shape[1]//2, :], cmap=plt.cm.gray)


    a3 = plt.subplot(1, 3, 3)
    plt.imshow(img3d[img_shape[0]//2, :, :], cmap=plt.cm.gray)

    plt.show()

def plot_3d(image):

    # copy so we don't change the original object
    p = image.copy()
    p = np.array(image)
    
    # marching cubes algorithm
    verts, faces, _, _ = measure.marching_cubes(p)

    # plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.70) # verts[faces] creates a collection of triangles
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()

def visualise_worst_best(img, label, segmentation):
    import pandas as pd

    # get dices and dice data
    dicesDict = data.get_patient_dices(img, label, segmentation)
    dices = np.array(list(dicesDict.values()))
    dices_desc = pd.DataFrame(dices).describe()
    worst, best, worst_idx, best_idx = data.get_worst_best(dicesDict)

    print(dices_desc)

    print("\nAggregated DICE info (for all slices)")
    print(f'Mean dice : {data.get_dice(label, segmentation)}')
    print(f'Best : {best}, {len(best_idx)} times')
    print(f'\nWorst : {worst}, {len(worst_idx)} times')
    
    # Visualise the best 
    amt_best = min(len(best_idx), 6)
    f = plt.figure(figsize=(30, 30 * amt_best))
    for i in range(amt_best):
        # get data
        best = best_idx[i]
        gtb, imgb, segb, diceb = data.get_gt_img_slice_seg_dice(img, label, segmentation, best)
        print(f"\nDICE best case: {diceb}, with slice index {best}")

        # show GT + segmentation 
        a1 = f.add_subplot(amt_best, 2, (i*2)+1)
        plt.imshow(gtb, cmap=plt.cm.gray)
        plt.imshow(segb, cmap=plt.cm.jet_r, alpha=0.5)

        # show segmentation over thing
        a1 = f.add_subplot(amt_best, 2, (i*2)+2)
        plt.imshow(imgb, cmap=plt.cm.gray)
        plt.imshow(segb, cmap=plt.cm.jet_r, alpha=0.1)
    plt.show()

    # Visualise the worst
    amt_worst = min(len(worst_idx), 20)
    f = plt.figure(figsize=(30, 15 * amt_worst))
    for i in range(amt_worst):
        # get data
        worst = worst_idx[i]
        gtw, imgw, segw, dicew = data.get_gt_img_slice_seg_dice(img, label, segmentation, worst)
        print(f"\nDICE worst case: {dicew}, with slice index {worst}")

        # show worst slice
        a1 = f.add_subplot(amt_worst, 2, (i*2)+1)
        plt.imshow(gtw, cmap=plt.cm.gray)
        plt.imshow(segw, cmap=plt.cm.jet_r, alpha=0.5)

        # show worst slice over dicom
        a1 = f.add_subplot(amt_worst, 2, (i*2)+2)
        plt.imshow(imgw, cmap=plt.cm.gray)
        plt.imshow(segw, cmap=plt.cm.jet_r, alpha=0.1)
    plt.show()


def visualise_patient_worst_best(patient_id, img_path, label_path, seg_path, modality):
    print(f"Visualising worst and best for {patient_id}\n")
    img, label, segmentation = data.get_all_data(patient_id, img_path, label_path, modality, seg_path)
    visualise_worst_best(img, label, segmentation)