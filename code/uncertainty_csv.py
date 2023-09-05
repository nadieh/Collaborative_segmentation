import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from wholeslidedata.image.wholeslideimage import WholeSlideImage
import os
import scipy.spatial
import scipy.ndimage
import scipy.spatial.distance
import glob
import pandas as pd


current_os = "w" if os.name == "nt" else "l"
other_os = "l" if current_os == "w" else "w"
def dice(pred, true, k):
    pred_bool = pred == k
    true_bool = true == k
    
    assert pred.shape == true.shape, "Both matrices should be of the same shape"    
    intersection = np.sum(np.logical_and(pred_bool,true_bool) * 2.0)
    dice = (intersection + 1) / (np.sum(pred_bool) + np.sum(true_bool)+1)
    return dice



def convert_path(path, to=current_os):
    """
    This function converts paths to the format of the desired platform.
    By default it changes paths to the platform you are cyurerntly using.
    This way you do not need to change your paths if you are executing your code on a different platform.
    It may however be that you mounted the drives differently.
    In that case you may need to change that in the code below.
    """
    if to in ["w", "win", "windows"]:

        path = path.replace("/mnt/pa_cpg", "Y:")
        path = path.replace("/data/pathology", "B:")
        path = path.replace("/mnt/pa_cpgarchive1", "W:")
        path = path.replace("/mnt/pa_cpgarchive2", "X:")
    if to in ["u", "unix", "l", "linux"]:
        path = path.replace("Y:", "/mnt/pa_cpg")
        path = path.replace("B:", "/data/pathology")
        path = path.replace("W:", "/mnt/pa_cpgarchive1")
        path = path.replace("X:", "/mnt/pa_cpgarchive2")
        path = path.replace("\\", "/")
    return path

def remove_background(image):
    # Load the image

    # Create a mask initialized with zeros
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define the background and foreground models
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # Define the rectangular region of interest (ROI) for GrabCut
    # Set the ROI to include the entire image
    width, height = image.shape[:2]
    rectangle = (1, 1, width - 1, height - 1)

    # Perform GrabCut
    cv2.grabCut(image, mask, rectangle, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create a mask where the background is 0 and the foreground is 1 or 3
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the input image to remove the background
    #image_no_bg = image * mask2[:, :, np.newaxis]

    return mask2


def remove_white_background(image):
    # Load the image

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper thresholds for the purple color range
    lower_purple = np.array([0, 0, 230])
    upper_purple = np.array([180, 30, 255])

    # Create a mask for the purple pixels in the image
    mask_black = cv2.inRange(hsv, lower_purple, upper_purple)

    # Define the lower and upper thresholds for the white color range
    lower_white = np.array([0, 0, 0])
    upper_white = np.array([180, 255, 30])

    # Create a mask for the white pixels in the image
    mask_white = cv2.inRange(hsv, lower_white, upper_white)


    return mask_black, mask_white



csv_file = 'metrics.csv'  # results will be saved to this file and prented on terminal as well. If not set, results 
trainer = 'nnUNetTrainerV2_BN__nnUNet_RGB_scaleTo_0_1_bs8_ps512_TEST2'
task = 'task_name'

patch_size =[1024, 1024]
spacing = 1
plots = 6

for center in ['umcu','rumc',  'cwz', 'lpon','rst' ]:
    data = []
    directory_uncertain = sorted(glob.glob(convert_path(rf'dir_to_uncertainty/*_uncertainty.tif')))
    directory_seg = sorted(glob.glob(convert_path(rf'directory_to_image/*_nnunet.tif')))

    for seg_path, uncertain_path in zip(directory_seg, directory_uncertain):
        filename = os.path.basename(uncertain_path)

        ref_path = convert_path(rf'/CAMELYON17//masks//{filename[:-16]}_mask.tif', to =current_os)
        im_path = convert_path(rf'/CAMELYON17//images//{filename[:-16]}.tif', to =current_os)

        wsi_un = WholeSlideImage(uncertain_path, backend='asap')
        ref = WholeSlideImage(ref_path, backend='asap')
        im = WholeSlideImage(im_path, backend='asap')
        wsi_seg = WholeSlideImage(seg_path, backend='asap')
        for x in range(0, im.shapes[0][0], patch_size[0]):
            for y in range(0, im.shapes[0][1], patch_size[1]):
                patch_unc = wsi_un.get_patch(x,y, patch_size[0], patch_size[1], relative= True, spacing = spacing)
                patch_seg = wsi_seg.get_patch(x,y, patch_size[0], patch_size[1],  relative= True, spacing = spacing)
                patch_ref = ref.get_patch(x,y, patch_size[0], patch_size[1],  relative= True,spacing = spacing)
                patch_im = im.get_patch(x,y, patch_size[0], patch_size[1], relative= True,spacing = spacing)
                if np.all(patch_unc==0):
                    continue
                #if (np.sum(patch_im==0)/3)>0.20*patch_size[0]*patch_size[1]:
                #    continue
                mask_w, mask_b = remove_white_background(patch_im)
                maximum_uncertainty = np.max(patch_unc)
                max_uncertain = np.sum(patch_unc>(maximum_uncertainty*0.8))
                uncertainty = np.sum(patch_unc)
                dice_array = []
                for i in range(1,3):    
                    d = dice(patch_seg, patch_ref,i)
                    dice_array.append(d)
                cancer_ref = np.any(patch_ref==2)
                cancer_seg = np.any(patch_seg==2)

                if (np.sum(mask_b==0)<0.60*patch_size[0]*patch_size[0]):
                    continue 
                #data.append({'filename':filename, 'image': patch_im, 'uncertain': patch_unc,'segmentation':patch_seg, 'Ref': patch_ref,'abs_uncertain':abs_uncertain, 'uncertainty_score':uncertainty,'center': center,'x':x, 'y':y, 'dice': dice_array, 'cancer_ref':cancer_ref
                #             , 'cancer_seg':cencer_seg})
                
                if (not(cancer_seg)) and (np.sum(mask_w==0)<0.7*patch_size[0]*patch_size[1]):

                    continue

                data.append({'filename':filename, 'max_uncertain':max_uncertain, 'uncertainty_score':uncertainty,'center': center,'x':x, 'y':y, 'dice': dice_array, 'cancer_ref':cancer_ref, 'cancer_seg':cancer_seg})
    data = pd.DataFrame(data)
    data.to_csv(convert_path(rf'root/{task}/{trainer}/{center}.csv'), index=False)
    print('saved')
