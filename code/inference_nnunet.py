#################################################################
### Imports
#################################################################
from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.accessories.asap.imagewriter import WholeSlideMaskWriter
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np
from nnunet.training.model_restore import load_model_and_checkpoint_files
import os
import torch
from wholeslidedata.samplers.utils import fit_data
import yaml
from wholeslidedata.image.wholeslideimage import WholeSlideImage
import time
import matplotlib.pyplot as plt

#################################################################
### Functions and utilities
#################################################################
current_os = "w" if os.name == "nt" else "l"
other_os = "l" if current_os == "w" else "w"


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


def norm_01(x_batch):
    x_batch = x_batch / 255
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def z_norm(x_batch):
    mean = x_batch.mean(axis=(-2,-1), keepdims=True)
    std = x_batch.std(axis=(-2,-1), keepdims=True)
    x_batch = ((x_batch - mean) / (std + 1e-8))
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch


def ensemble_softmax_list(trainer, params, x_batch):
    softmax_list = []
    for p in params:
        trainer.load_checkpoint_ram(p, False)
        softmax_list.append(
            trainer.predict_preprocessed_data_return_seg_and_softmax(x_batch.astype(np.float32), verbose=False,
                                                                     do_mirroring=False, mirror_axes=[])[
                -1].transpose(1, 2, 3, 0).squeeze())
    return softmax_list


def array_to_formatted_tensor(array):
    array = np.expand_dims(array.transpose(2, 0, 1), 0)
    return torch.tensor(array)


def softmax_list_and_mean_to_uncertainty(softmax_list, softmax_mean):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    uncertainty_loss_per_pixel_list = []
    for softmax in softmax_list:
        log_softmax = np.log(softmax + 0.00000001)
        uncertainty_loss_per_pixel = loss(array_to_formatted_tensor(log_softmax),
                                          array_to_formatted_tensor(softmax_mean))
        uncertainty_loss_per_pixel_list.append(uncertainty_loss_per_pixel)
    uncertainty = torch.cat(uncertainty_loss_per_pixel_list).mean(dim=0)
    return uncertainty

def get_trim_indexes(y_batch):
    """
    Using the y_mask / tissue-background mask we can check if there are
    full empty rows and columns with a width of half the model patch size.
    We check this in half model patch size increments because otherwise
    we screw up the half overlap approach from nnunet (resulting in inconsistent
    overlap thoughout the WSI).
    We will still need 1 row or column that is empty to make sure the parts that
    do have tissue have 4x overlap
    """
    y = y_batch[0]
    r_is_empty = [not y[start:end].any() for start, end in zip(half_patch_size_start_idxs, half_patch_size_end_idxs)]
    c_is_empty = [not y[:, start:end].any() for start, end in zip(half_patch_size_start_idxs, half_patch_size_end_idxs)]
    empty_rs_top = 0
    for r in r_is_empty:
        if r == True:
            empty_rs_top += 1  # count empty rows
        else:
            trim_top_half_idx = empty_rs_top - 1  # should always include a single empty row, since we need the overlap
            trim_top_half_idx = np.clip(trim_top_half_idx, 0, None)  # cannot select regiouns outside sampled patch
            trim_top_idx = half_patch_size_start_idxs[trim_top_half_idx]
            break

    empty_rs_bottom = 0
    for r in r_is_empty[::-1]:
        if r == True:
            empty_rs_bottom += 1
        else:
            trim_bottom_half_idx = empty_rs_bottom - 1
            trim_bottom_half_idx = np.clip(trim_bottom_half_idx, 0, None)
            trim_bottom_idx = half_patch_size_end_idxs[::-1][trim_bottom_half_idx]  # reverse index
            break

    empty_cs_left = 0
    for c in c_is_empty:
        if c == True:
            empty_cs_left += 1
        else:
            trim_left_half_idx = empty_cs_left - 1
            trim_left_half_idx = np.clip(trim_left_half_idx, 0, None)
            trim_left_idx = half_patch_size_start_idxs[trim_left_half_idx]
            break

    empty_cs_right = 0
    for c in c_is_empty[::-1]:
        if c == True:
            empty_cs_right += 1
        else:
            trim_right_half_idx = empty_cs_right - 1
            trim_right_half_idx = np.clip(trim_right_half_idx, 0, None)
            trim_right_idx = half_patch_size_end_idxs[::-1][trim_right_half_idx]
            break

    # print(trim_top_half_idx, trim_bottom_half_idx, trim_left_half_idx, trim_right_half_idx)
    return trim_top_idx, trim_bottom_idx, trim_left_idx, trim_right_idx






def get_closest_spacing(spacing_value):
    possible_spacings = [0.25, 0.5, 1, 2, 4, 8, 16]
    closest = min(possible_spacings, key=lambda x:abs(x-spacing_value))
    return closest

#################################################################
### LOAD MODEL, change your model here if needed
#################################################################

import sys

# Retrieve the command line arguments
task_name = sys.argv[1:][0]

# Check if any arguments are provided
if len(task_name) > 0:
    # Concatenate the arguments into a single string
    input_string = ' '.join(task_name)

    # Use the input string in your code
    print("Input string:", input_string)
else:
    print("No input string provided.")

#trainer_name = 'nnUNetTrainerV2_BN_pathology_DA_hed005__nnUNet_RGB_scaleTo_0_1_bs8_ps512'
trainer_name = 'nnUNetTrainerV2_BN__nnUNet_RGB_scaleTo_0_1_bs8_ps512'

model_base_path = rf'/data/pathology/projects/pathology-mrifuse/results/nnUNet/2d/{task_name}/{trainer_name}'
print(model_base_path, '\n')

folds = (0)
mixed_precision = None
checkpoint_name = "model_final_checkpoint"

trainer, params = load_model_and_checkpoint_files(model_base_path, folds, mixed_precision=mixed_precision,checkpoint_name=checkpoint_name)
norm = z_norm if (('score' in trainer_name) or ('default' in trainer_name)) else norm_01

print('TEMP hardcoded')
train=False
if train:
    dataset_name = 'batch1_3' 
    load_data = 'training'
else:
    dataset_name = 'batch1_3' 
    load_data = 'validation'
    
    
config_yml_output_filename = f"wsi_borderless_inference_config_{dataset_name}.yml"
config_yml_output_path = convert_path(rf'B:\projects\pathology-mrifuse\code\config_files\{config_yml_output_filename}')
yaml_path = convert_path ('B:\projects\pathology-mrifuse\yaml\data_camelyon17.yaml') 

with open(yaml_path, 'r') as file:
    yaml_data = yaml.safe_load(file)
    
validation_data = yaml_data['data'][load_data]
camelyon = yaml_data['path']['camelyon17']       
save = True
# umcu 
# done rst cwz
center_list = [  'rumc','umcu','rst','cwz','lpon']
cancer_list = [ 'negative', 'itc','micro', 'macro' ]

for center in center_list:
    
    output_folder = Path(rf'/data/pathology/projects/pathology-mrifuse/nnUNet_raw_data_base/inference_results/{task_name}_WSI_inference/{trainer_name}_TEST2/{center}')
    if train: 
        files_yml_output_filename = f"wsi_borderless_inference_files_{dataset_name}_{center}_train.yml"
    else:
        files_yml_output_filename = f"wsi_borderless_inference_files_{dataset_name}_{center}.yml"
    os.makedirs(output_folder, exist_ok=True)
    files_yml_output_path = convert_path(rf'B:\projects\pathology-mrifuse\code\config_files\{files_yml_output_filename}')
    yaml_file = {"validation": []}
    dic_sample = {}
    for cancer in cancer_list:
        path = validation_data['{}/{}'.format(cancer, center)]
        #here group_number
        image_anno_batch = [(i['image'].format(root='/data/pathology',camelyon17=camelyon), i['mask'].format(root='/data/pathology',camelyon17=camelyon)) for i in path ]        
        for wsi, wsa in image_anno_batch:
            yaml_file["validation"].append({"wsi": {"path": convert_path(str(wsi), to='linux')},
                                            "wsa": {"path": convert_path(str(wsa), to='linux')}})
    with open(convert_path(files_yml_output_path), 'w') as f:
        yaml.dump(yaml_file, f)
        print('CREATED FILES YAML:', files_yml_output_path)


    spacing = 1
    model_patch_size = 512 # input size of model (should be square)
    half_model_patch_size=int(model_patch_size/2)
    sampler_patch_size = 4096 
    assert sampler_patch_size % (model_patch_size/2) == 0 # needed for correct half overlap
    output_patch_size = sampler_patch_size - 2 * half_model_patch_size # use this as your annotation_parser shape
    template = convert_path(r"B:\projects\pathology-mrifuse\code\config_files\template\borderless_config.yml")

    with open(template) as f:
        config_yml_str = str(yaml.safe_load(f))


    replace_dict = {
        "'auto_files_yml'" : files_yml_output_path,
        "'auto_sampler_patch_size'" : sampler_patch_size,
        "'auto_spacing'" : 1,
        "'auto_tissue_mask_spacing'" :1,
        "'auto_tissue_mask_ratio'" : 1,
        "'auto_output_patch_size'" : output_patch_size

    }

    print('\nAuto configuring CONFIG YAML. Replacing template placeholders:')
    for k, v in replace_dict.items():
        print('\t', k, v)
        config_yml_str = config_yml_str.replace(k, str(v))

    config_yml = yaml.safe_load(config_yml_str)

    with open(convert_path(config_yml_output_path), 'w') as f:
        yaml.dump(config_yml, f, )
        print('CREATED CONFIG YAML:', config_yml_output_filename)
    #################################################################

    # Some variable settings you can ignore
    mode = "validation"
    image_path = None
    previous_file_key = None
    files_exist_already = None  # not sure if needed
    plot = False
    # following is later used to check if we can remove big empty parts of the sampled patch before inference
    sampler_patch_size_range = list(range(sampler_patch_size))
    half_patch_size_start_idxs = sampler_patch_size_range[0::half_model_patch_size]
    half_patch_size_end_idxs = [idx + half_model_patch_size for idx in half_patch_size_start_idxs]
    training_iterator = create_batch_iterator(mode=mode,
                                              user_config=config_yml_output_path,
                                              presets=('slidingwindow',),
                                              cpus=4,
                                              number_of_batches=-1,
                                              return_info=True)
    previous_file_key = None
    for x_batch, y_batch, info in tqdm(training_iterator):

        ### Get image data, check if new image, save previous image if there was one, if new image create new image file
        if not np.any(y_batch):
            continue
        y_batch[np.where(y_batch==2)]=1

        sample_reference = info['sample_references'][0]['reference']
        current_file_key = sample_reference.file_key
        if current_file_key != previous_file_key:  # if starting a new image
            if previous_file_key != None and files_exist_already != True:  # if there was a previous image, and the previous image did not exist already (can also be None)
                wsm_writer.save()  # save previous mask
                wsu_writer.save()  # save previous uncertainty
                # Save runtime
                end_time = time.time()
                run_time = end_time - start_time
                text_file = open(output_folder / (image_path.stem + '_runtime.txt'), "w")
                text_file.write(str(run_time))
                text_file.close()
            # Getting file settings and path, doing check if exists already
            with training_iterator.dataset.get_wsi_from_reference(sample_reference) as wsi:
                image_path = wsi.path
                shape = wsi.shapes[wsi.get_level_from_spacing(spacing)]
                real_spacing = wsi.get_real_spacing(spacing)
            wsm_path = output_folder / (image_path.stem + '_nnunet.tif')
            wsu_path = output_folder / (image_path.stem + '_uncertainty.tif')
            textfile = output_folder / (image_path.stem + '_runtime.txt')

            if os.path.isfile(wsm_path) and os.path.isfile(wsu_path) and os.path.isfile(textfile):
                files_exist_already = True  # this means we can skip this whole loop for this file key, checked above '### Prep, predict and uncertainty'
                previous_file_key = current_file_key
                print(f'[SKIPPING] files for {image_path.stem} exist already')
                continue  # continue to next batch
            else:
                files_exist_already = False
            # Create new writer and file
            start_time = time.time()
            wsm_writer = WholeSlideMaskWriter()  # whole slide mask
            wsu_writer = WholeSlideMaskWriter()  # whole slide uncertainty
                # Create files
            wsm_writer.write(path=wsm_path, spacing=real_spacing, dimensions=shape,
                                 tile_shape=(output_patch_size, output_patch_size))
            wsu_writer.write(path=wsu_path, spacing=real_spacing,
                                 dimensions=shape, tile_shape=(output_patch_size, output_patch_size))


        if files_exist_already:
            continue
        ### Trim check
        trim_top_idx, trim_bottom_idx, trim_left_idx, trim_right_idx = get_trim_indexes(y_batch)
        x_batch_maybe_trimmed = x_batch[:, trim_top_idx : trim_bottom_idx, trim_left_idx: trim_right_idx, :]

        ### Prep, predict and uncertainty
        prep = norm(x_batch_maybe_trimmed)

        softmax_list = ensemble_softmax_list(trainer, params, prep)
        softmax_mean = np.array(softmax_list).mean(0)
        pred_output_maybe_trimmed = softmax_mean.argmax(axis=-1)

        uncertainty = softmax_list_and_mean_to_uncertainty(softmax_list, softmax_mean)
        uncertainty_output_maybe_trimmed = np.array((uncertainty.clip(0, 4) / 4 * 255).int()) 

        ### Reconstruct possible trim
        pred_output = np.zeros((sampler_patch_size, sampler_patch_size))
        pred_output[trim_top_idx : trim_bottom_idx, trim_left_idx: trim_right_idx] = pred_output_maybe_trimmed

        uncertainty_output = np.zeros((sampler_patch_size, sampler_patch_size))
        uncertainty_output[trim_top_idx: trim_bottom_idx, trim_left_idx: trim_right_idx] = uncertainty_output_maybe_trimmed

        # Only write inner part
        pred_output_inner = fit_data(pred_output, [output_patch_size, output_patch_size])
        uncertainty_output_inner = fit_data(uncertainty_output, [output_patch_size, output_patch_size])
        y_batch_inner = fit_data(y_batch[0], [output_patch_size, output_patch_size]).astype('int64')

        ### Get patch point
        point = info['sample_references'][0]['point']
        c, r = point.x - output_patch_size/2, point.y - output_patch_size/2 # from middle point to upper left point of tile to write

        ### Write tile and set previous file key for next loop check
        wsm_writer.write_tile(tile=pred_output_inner, coordinates=(int(c), int(r)), mask=y_batch_inner)
        wsu_writer.write_tile(tile=uncertainty_output_inner, coordinates=(int(c), int(r)), mask=y_batch_inner)
        previous_file_key = current_file_key
    if not files_exist_already:

        wsm_writer.save()  # if done save last image
        wsu_writer.save()  # if done save last image

    # Save runtime
        end_time = time.time()
        run_time = end_time - start_time
        text_file = open(output_folder / (image_path.stem + '_runtime.txt'), "w")
        text_file.write(str(run_time))
        text_file.close()

training_iterator.stop()
print("DONE")
# ~/c-submit --require-cpus=8 --require-mem=32g --gpu-count=1 --require-gpu-mem=10G --priority=high 'nadiehkhalili' 5742 48 doduo1.umcn.nl/pathology_lung_til/nnunet:9.4-midl2023 python3 /data/pathology/projects/pathology-mrifuse/nnunet/inference_nnun