from savenifti import convert_2d_image_to_nifti
from polygon import convert_polygons_to_annotations, get_tissue_union_from_mask
from filepath import convert_path, GenerateJson
import os
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from wholeslidedata.image.wholeslideimage import WholeSlideImage
#from wholeslidedata.accessories.asap.parser import AsapAnnotationParser
from matplotlib import pyplot as plt
from wholeslidedata.samplers.patchlabelsampler import SegmentationPatchLabelSampler
from shapely.geometry import Point
import numpy as np
import pandas as pd
import SimpleITK as sitk
from wholeslidedata.annotation.utils import cv2_polygonize
from shapely import geometry
from wholeslidedata.labels import Label
from wholeslidedata.annotation.structures import Annotation
import yaml
import random
current_os = "w" if os.name == "nt" else "l"
other_os = "l" if current_os == "w" else "w"
camelyon_path = convert_path('/data/pathology/projects/pathology-mrifuse/yaml/data_camelyon17.yaml', current_os)
with open(camelyon_path, 'r') as f:
    yaml = yaml.safe_load(f)
root = yaml['path']['root']
camelyon = yaml['path']['camelyon17']     
train_data = yaml['data']['training']
labels={'Background':0,
        'Benign': 1, 
        'Tumor': 2}

current_os = "w" if os.name == "nt" else "l"
other_os = "l" if current_os == "w" else "w"
nnUNet_base = r"root_base"
nnUNet_data_root = convert_path(os.path.join(nnUNet_base, "nnUNet_raw_data"), current_os)
sorted(os.listdir(nnUNet_data_root))
nnUNet_base_linux = convert_path(nnUNet_base, to=current_os)
print("nnUNet ROOT:\t", nnUNet_base_linux)
from shapely.geometry import Point, Polygon
patch_size = (1024,1024)
#sample every 100 pixel
sample_rate = patch_size[0]*patch_size[1]
plot = False
contrast_value = 20000 #remove white patch--- masks are bad quality
save = True
# 'rst', 'umcu','rumc', 'lpon','cwz'
center_list = ['rst', 'umcu','rumc', 'lpon','cwz']
cancer_list = [ 'negative','itc', 'micro','macro']
# counts_wsi = np.zeros(n_labels)

import pandas as pd
trainer = 'nnUNetTrainerV2_BN__nnUNet_RGB_scaleTo_0_1_bs8_ps512_TEST2'
# Specify the path to the CSV file
patchsize = 1024
plots = 2
spacing = 1
df_total = []

for center in ['umcu','rumc',  'cwz','lpon','rst' ]:

    csv_file_path = convert_path(rf'/{center}.csv',to = current_os )
    df = pd.read_csv(csv_file_path)
        
    df_total.append(df)

df_total = pd.concat(df_total)
dict_data = df_total.to_dict(orient='records')
sorted_data = sorted(dict_data, key=lambda x: 100*x['max_uncertain']+x['uncertainty_score'], reverse= True)



task_name = f'Task0{i}_name_{sample_per_slide}'
task_root = os.path.join(nnUNet_data_root, task_name)
task_root = convert_path(task_root, current_os)
print("TASK:\t\t", task_name)
nnUNet_base_linux = convert_path(nnUNet_base, to=current_os)
print("nnUNet ROOT:\t", nnUNet_base_linux)
image_folder = os.path.join(task_root, "imagesTr")
image_folder = convert_path(image_folder, current_os)
label_folder = os.path.join(task_root, "labelsTr")
label_folder = convert_path(label_folder, current_os)
if not(os.path.isdir(image_folder)):
    os.makedirs(image_folder)
if not(os.path.isdir(label_folder)):
    os.makedirs(label_folder)

sample_num = sample_per_slide*15*5
for i,sample in enumerate(range(sample_num)):
    if sample> len(sorted_data):
        sample = random.randint(0,len(sorted_data))
    data = sorted_data[sample]
    filename = data['filename'][:-16]
    x = random.randint(data['x']-2*patch_size[0]/2,data['x']+2*patch_size[0]/2)
    y = random.randint(data['y']-2*patch_size[0]/2,data['y']+2*patch_size[0]/2)
    image = convert_path(f'CAMELYON17\images\{filename}.tif',  to=current_os)
    mask = convert_path(f'CAMELYON17\masks\{filename}_mask.tif',  to=current_os)
    s = os.path.split(image)[-1].split(".")[-2]

    wsi = WholeSlideImage(image, backend="asap")
    wsi_mask = WholeSlideImage(mask, backend="asap")

    image_patch = wsi.get_patch(x,y, patch_size[0], patch_size[1],  spacing,relative=True)
    mask_patch = wsi_mask.get_patch(x,y, patch_size[0], patch_size[1], spacing, relative=True)
    output_filename = filename+ '_{}'.format(i)
    if save:
        convert_2d_image_to_nifti(image_patch, os.path.join(image_folder, output_filename))
        convert_2d_image_to_nifti(mask_patch, os.path.join(label_folder, output_filename), is_seg=True)  
for i,sample_per_slide in zip([task_name],[sample_per_slide]):
    task_name = f'Task{i}_camelyon_uncertainty_{sample_per_slide}'
    task_root = os.path.join(nnUNet_data_root, task_name)
    task_root = convert_path(task_root, current_os)
    print("TASK:\t\t", task_name)
    nnUNet_base_linux = convert_path(nnUNet_base, to=current_os)
    print("nnUNet ROOT:\t", nnUNet_base_linux)
    image_folder = os.path.join(task_root, "imagesTr")
    image_folder = convert_path(image_folder, current_os)
    label_folder = os.path.join(task_root, "labelsTr")
    label_folder = convert_path(label_folder, current_os)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    dataset_path = task_root
    GenerateJson(dataset_path)