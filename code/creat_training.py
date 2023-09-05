from savenifti import convert_2d_image_to_nifti
from polygon import convert_polygons_to_annotations, get_tissue_union_from_mask
from filepath import convert_path, GenerateJson
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from matplotlib import pyplot as plt
from shapely.geometry import Point
import numpy as np
import yaml
import random
import os
import math
random.seed(10)
patch_size = (1024, 1024)


current_os = "w" if os.name == "nt" else "l"
other_os = "l" if current_os == "w" else "w"
camelyon_path = convert_path('/data/pathology/projects/pathology-mrifuse/yaml/data_camelyon16.yaml', current_os)
with open(camelyon_path, 'r') as f:
    yaml = yaml.safe_load(f)
root = yaml['path']['root']
camelyon = yaml['path']['camelyon16']
train_data = yaml['data']['training']
from matplotlib.colors import LinearSegmentedColormap

label_names = ['Background','Benign','Tumor']
label_index = list(range(len(label_names)))
n_labels = len(label_names)
label_plot_args = {"cmap":cmap, "vmin":0, "vmax":255, "interpolation":"none"}
# easy to copy paste in figures this way
fig, axs = plt.subplots(1, 2, figsize=(5, 5))
axs[-1].imshow([[i] for i in list(range(3))], **label_plot_args)
axs[-1].set_yticks(label_index)
axs[-1].set_yticklabels(label_names)
axs[-1].yaxis.tick_right()
axs[-1].get_xaxis().set_visible(False)
axs[-1].set_title("Labels")
plt.show()

labels={'Background':0,
        'Benign': 1,
        'Tumor': 2}

nnUNet_base = r"B:\projects\pathology-mrifuse"
nnUNet_data_root = convert_path(os.path.join(nnUNet_base, "nnUNet_raw_data"), current_os)
sorted(os.listdir(nnUNet_data_root))
task_name = 'camelyon16'
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



# sample every 100 pixel
sample_rate = patch_size[0] * patch_size[1]
spacing = 1
contrast_value = 20000  # remove white patch--- masks are bad quality
plot = False
# 'rst', 'umcu',,'rumc', 'lpon'
center_list = ['rst', 'umcu',,'rumc', 'lpon]
cancer_list = [ 'negative','micro','macro', 'itc']
save = True
# counts_wsi = np.zeros(n_labels)
for center in center_list:
    dic_sample = {}
    for cancer in cancer_list:
        path = train_data['{}/breast/{}'.format(cancer, center)]
        for i in path:
            image = i['image'].format(root='/data/pathology', camelyon16=camelyon)
            mask = i['mask'].format(root='/data/pathology', camelyon16=camelyon)
            annotation_path = '/data/pathology/archives/breast/camelyon/CAMELYON16/annotations/{}'
            filename = os.path.basename(image)[:-4]
            a = annotation_path.format(filename + '.xml')
            s = os.path.split(a)[-1].split(".")[-2]
            print(filename)
            wsi = WholeSlideImage(image, backend="asap")
            wsi_mask = WholeSlideImage(mask, backend="asap")
            tissues = get_tissue_union_from_mask(wsi_mask, wsi, spacing, value=2)

            sample_num = 10
            wsa = WholeSlideAnnotation(a)
            for idx, tissue in enumerate(tissues):
                tumore_sample_size = tissue.area / (sample_rate / 10)
                print('sample size', cancer, tumore_sample_size)

                if tumore_sample_size < 1:
                    tumore_sample_size = 5
                if tumore_sample_size > 25:
                    tumore_sample_size = 25
                print('sample size', cancer, tumore_sample_size)
                for sample in range(math.ceil(tumore_sample_size)):
                    point = Point(0, 0)
                    while not (tissue.contains(point)):
                        random_center_x = random.randint(int(np.min(tissue.coordinates[:, 0])) - patch_size[0],
                                                         int(np.max(tissue.coordinates[:, 0])) + patch_size[0])
                        random_center_y = random.randint(int(np.min(tissue.coordinates[:, 1])) - patch_size[0],
                                                         int(np.max(tissue.coordinates[:, 1])) + patch_size[0])
                        point = Point(random_center_x, random_center_y)


                    image_patch = wsi.get_patch(random_center_x, random_center_y,
                                                patch_size[0], patch_size[1], spacing)

                    mask_patch = wsi_mask.get_patch(random_center_x, random_center_y,
                                                    patch_size[0], patch_size[1], spacing)
                    counts, bins = np.histogram(image_patch, range(255))
                    contrast = np.sum(counts)
                    print(contrast)
                    if contrast < contrast_value:
                        continue
                    output_filename = filename + '_{}'.format(idx) + '_{}'.format(sample)
                    if save:
                        convert_2d_image_to_nifti(image_patch, os.path.join(image_folder, output_filename))
                        convert_2d_image_to_nifti(mask_patch, os.path.join(label_folder, output_filename), is_seg=True)


