import json
import os
import random
import shlex
from subprocess import Popen, PIPE
from typing import Union

import numpy as np
import torch
import torchvision
from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize as imresize

from experiment_utils.constants import MAYBE_INT, MAYBE_FLOAT, MAYBE_LIST_FLOAT, MAYBE_LIST_INT, \
    NESTED_MAYBE_LIST_FLOAT, \
    NESTED_MAYBE_LIST_INT


def initialize_paths(output_image_dir: str, output_scene_dir: str, up_to_here: str) -> None:
    if not os.path.exists(output_image_dir):
        try:
            os.mkdir(output_image_dir)
        except FileNotFoundError:
            os.mkdir('/'.join(output_image_dir.split('/')[:-1]))
        finally:
            try:
                os.mkdir(output_image_dir)
            except:
                pass

    if not os.path.exists(output_scene_dir):
        os.mkdir(output_scene_dir)

    if not os.path.exists(up_to_here + '/questions/'):
        os.mkdir(up_to_here + '/questions/')
    return


def item_creation(shape: MAYBE_INT = None,
                  color: MAYBE_INT = None,
                  material: MAYBE_INT = None,
                  size: MAYBE_INT = None,
                  x: MAYBE_FLOAT = None,
                  y: MAYBE_FLOAT = None,
                  theta: MAYBE_FLOAT = None) -> str:
    """
      Creates the dictionary argparse equivalent of an item.
      Leaving something None, takes a random guess.
    """

    shapes = {
        0: "\"SmoothCube_v2\"",
        1: "\"Sphere\"",
        2: "\"SmoothCylinder\""
    }

    if shape is not None:
        ret_shape = shapes[shape]
    else:
        ret_shape = shapes[random.randint(0, 2)]

    colors = {
        0: "\"gray\"",  # [87, 87, 87],
        1: "\"red\"",  # [173, 35, 35],
        2: "\"blue\"",  # [42, 75, 215],
        3: "\"green\"",  # [29, 105, 20],
        4: "\"brown\"",  # [129, 74, 25],
        5: "\"purple\"",  # [129, 38, 192],
        6: "\"cyan\"",  # [41, 208, 208],
        7: "\"yellow\""  # [255, 238, 51]
    }
    if color is not None:
        ret_color = colors[color]
    else:
        ret_color = colors[random.randint(0, 7)]

    materials = {
        0: "\"Rubber\"",
        1: "\"MyMetal\""
    }

    if material is not None:
        ret_material = materials[material]
    else:
        ret_material = materials[random.randint(0, 1)]

    sizes = {
        0: "\"large\"",  # 0.7,
        1: "\"small\"",  # 0.35
    }
    if size is not None:
        ret_size = sizes[size]
    else:
        ret_size = sizes[random.randint(0, 1)]

    if x is not None:
        ret_x = x
    else:
        ret_x = 6 * (random.random() - 0.5)

    if y is not None:
        ret_y = y
    else:
        ret_y = 6 * (random.random() - 0.5)

    if theta is not None:
        ret_theta = theta
    else:
        ret_theta = random.random()

    constructor = '{"object":' + str(ret_shape) + ',"color":' + str(ret_color) + ',"material":' + str(
        ret_material) + ',"size":' + str(ret_size) + ',"theta":' + str(ret_theta) + ',"x":' + str(
        ret_x) + ',"y":' + str(ret_y) + '}'
    return constructor


def scene_creation(shape_list: MAYBE_LIST_INT = [None],
                   color_list: MAYBE_LIST_INT = [None],
                   material_list: MAYBE_LIST_INT = [None],
                   size_list: MAYBE_LIST_INT = [None],
                   x_list: MAYBE_LIST_FLOAT = [None],
                   y_list: MAYBE_LIST_FLOAT = [None],
                   theta_list: MAYBE_LIST_FLOAT = [None],
                   multiply: int = 1) -> str:
    """
      Creates the dictionary argparse equivalent of a scene.
      Leaving something None, takes a random guess.
    """
    scene_list = '['
    for sh, c, m, si, x, y, t in zip(shape_list * multiply, color_list * multiply, material_list * multiply,
                                     size_list * multiply, x_list * multiply, y_list * multiply, theta_list * multiply):
        scene_list += item_creation(sh, c, m, si, x, y, t)
        scene_list += ','
    scene_list = ''.join([f for f in scene_list[:-1]]) + ']'
    return scene_list


def multi_scene_creation(list_of_shape_list: NESTED_MAYBE_LIST_INT = [[None], [None]],
                         list_of_color_list: NESTED_MAYBE_LIST_INT = [[None], [None]],
                         list_of_material_list: NESTED_MAYBE_LIST_INT = [[None], [None]],
                         list_of_size_list: NESTED_MAYBE_LIST_INT = [[None], [None]],
                         list_of_x_list: NESTED_MAYBE_LIST_FLOAT = [[None], [None]],
                         list_of_y_list: NESTED_MAYBE_LIST_FLOAT = [[None], [None]],
                         list_of_theta_list: NESTED_MAYBE_LIST_FLOAT = [[None], [None]]) -> str:
    """
      Creates the dictionary argparse equivalent of multiple scenes.
      Leaving something None, takes a random guess.
    """
    multi_scene_dict = '{'
    counter = 0
    for sh_list, c_list, m_list, si_list, x_list, y_list, t_list in zip(list_of_shape_list, list_of_color_list,
                                                                        list_of_material_list, list_of_size_list,
                                                                        list_of_x_list, list_of_y_list,
                                                                        list_of_theta_list):
        multi_scene_dict += f'"{counter}":'
        multi_scene_dict += scene_creation(sh_list, c_list, m_list, si_list, x_list, y_list, t_list)
        multi_scene_dict += ','
        counter += 1
    multi_scene_dict = ''.join([f for f in multi_scene_dict[:-1]]) + '}'
    return multi_scene_dict


def distribute(workers: int, num_images: int, start_idx: int, split: str, **kwargs) -> dict:
    ### Let's see if we can multi-process this thing ###
    ### Well if that is the case that must happen on Image Level ###
    ### So first check if we can split the arguments per-image-per-worker ###
    collect_locals = kwargs
    effective_workers = workers
    if num_images <= workers:
        images_per_worker = [1] * num_images
        effective_workers = list(range(num_images))
    elif num_images > workers:
        images_per_worker = [0] * workers
        for i in range(0, num_images):
            images_per_worker[i % workers] += 1
        ### Check if the processed can be broken down in a "fair" way ###
        effective_workers = list(range(workers))
    else:
        images_per_worker = -1
    start_indexes = [start_idx] * len(effective_workers)
    for i in range(1, len(effective_workers)):
        start_indexes[i] = start_indexes[i - 1] + images_per_worker[i - 1]

    effective_args: dict = {}
    for i in set(effective_workers):
        effective_args.update({f'worker_{i}': {
            'key_light_jitter': [],
            'fill_light_jitter': [],
            'back_light_jitter': [],
            'camera_jitter': [],
            'num_objects': [],
            'per_image_shapes': [],
            'per_image_colors': [],
            'per_image_materials': [],
            'per_image_sizes': [],
            'per_image_x': [],
            'per_image_y': [],
            'per_image_theta': [],
            'object_properties': None,
            'num_images': None,
            'split': None,
            'start_idx': None,
        }})

    for argument, passed_value in collect_locals.items():
        if argument in ['clean_before', 'assemble_after', 'output_scene_file', 'output_image_dir', 'output_scene_dir',
                        'output_scene_file']:
            continue
            #################################################################################
        else:
            for i in effective_workers:
                for j in range(images_per_worker[i]):
                    effective_args[f'worker_{i}'][argument].append(
                        passed_value[start_indexes[i] - min(start_indexes) + j])

    for argument in ['num_images', 'split', 'start_idx', 'workers']:
        if argument == 'num_images':
            for i in effective_workers:
                effective_args[f'worker_{i}'].update({'num_images': images_per_worker[i]})
        elif argument == 'split':
            for i in effective_workers:
                effective_args[f'worker_{i}'].update({'split': split})
        elif argument == 'start_idx':
            for i in effective_workers:
                effective_args[f'worker_{i}'].update({'start_idx': start_indexes[i]})
        elif argument == 'workers':
            pass
        else:
            raise ValueError("You passed an argument that made it up to here?")

    for i in effective_workers:
        effective_args[f'worker_{i}']['object_properties'] = multi_scene_creation(
            list_of_shape_list=effective_args[f'worker_{i}']['per_image_shapes'],
            list_of_color_list=effective_args[f'worker_{i}']['per_image_colors'],
            list_of_material_list=effective_args[f'worker_{i}']['per_image_materials'],
            list_of_size_list=effective_args[f'worker_{i}']['per_image_sizes'],
            list_of_x_list=effective_args[f'worker_{i}']['per_image_x'],
            list_of_y_list=effective_args[f'worker_{i}']['per_image_y'],
            list_of_theta_list=effective_args[f'worker_{i}']['per_image_theta'],
        )

    return effective_args


def dict_to_binary(the_dict: dict):
    string = json.dumps(the_dict)
    binary = '__'.join(format(ord(letter), 'b') for letter in string)
    return binary


def binary_to_dict(the_binary: str):
    jsn = ''.join(chr(int(x, 2)) for x in the_binary.split('__'))
    d = json.loads(jsn)
    return d


def command_template(num_images: MAYBE_LIST_INT,
                     key_light_jitter: MAYBE_LIST_INT,
                     fill_light_jitter: MAYBE_LIST_INT,
                     back_light_jitter: MAYBE_LIST_INT,
                     camera_jitter: MAYBE_LIST_INT,
                     num_objects: MAYBE_LIST_INT,
                     object_properties: dict,
                     split: str,
                     start_idx: int,
                     up_to_here: str,
                     output_image_dir: str,
                     output_scene_dir: str,
                     output_scene_file: str,
                     **kwargs
                     ) -> str:
    key_light_jitter = [str(f) for f in key_light_jitter]
    fill_light_jitter = [str(f) for f in fill_light_jitter]
    back_light_jitter = [str(f) for f in back_light_jitter]
    camera_jitter = [str(f) for f in camera_jitter]
    num_objects = [str(f) for f in num_objects]
    # TODO(Spyros) Enable  this in final  push
    cmd_template = f'blender -noaudio --background --python {up_to_here}/generation/det_render_images.py -- --num_images={num_images} \
      --key_light_jitter={",".join(key_light_jitter)} \
      --fill_light_jitter={",".join(fill_light_jitter)} \
      --back_light_jitter={",".join(back_light_jitter)} \
      --camera_jitter={",".join(camera_jitter)} \
      --num_objects={",".join(num_objects)} \
      --split={split} \
      --object_properties={dict_to_binary(object_properties)}\
      --output_image_dir={output_image_dir} \
      --output_scene_dir={output_scene_dir} \
      --output_scene_file={output_scene_file} \
      --use_gpu=1 --render_tile_size=2048 --render_num_samples=256 --width=480 --height=320 --start_idx={start_idx}'
    return cmd_template


def question_template(input_scene_file: str,
                      output_questions_file: str,
                      templates_per_image: int,
                      instances_per_template: int,
                      start_idx: int) -> str:
    cmd_template = f'python ./generation/generate_questions.py \
                  --input_scene_file={input_scene_file} \
                  --output_questions_file={output_questions_file} \
                  --templates_per_image={templates_per_image} \
                  --instances_per_template={instances_per_template} \
                  --scene_start_idx={start_idx}'
    return cmd_template


def restore_digits(number: int, digits: int = 6) -> str:
    number_as_str = str(number)
    n_chars = len(number_as_str)
    n_pad = max(0, digits - n_chars)
    padded_number = ['0'] * n_pad + [f for f in number_as_str]
    return ''.join(padded_number)


def render_image(
        output_image_dir: str,
        output_scene_dir: str,
        output_scene_file: str,
        up_to_here: str,
        clean_before: bool = False,
        **kwargs
):
    collected_locals = kwargs
    start_idx = kwargs['start_idx']
    num_images = kwargs['num_images']
    split = kwargs['split']
    if clean_before:
        targets = os.listdir(output_image_dir)
        for target in targets:
            if split in target:
                try:
                    os.remove(output_image_dir + '/' + target)
                except:
                    pass

        targets = os.listdir(output_scene_dir)
        for target in targets:
            if split in target:
                try:
                    os.remove(output_scene_dir + '/' + target)
                except:
                    pass
    effective_args = distribute(**collected_locals)
    cmds = [
        command_template(up_to_here=up_to_here, output_image_dir=output_image_dir, output_scene_dir=output_scene_dir,
                         output_scene_file=output_scene_file, **effective_args[f]) for f in effective_args.keys()]

    args = [shlex.split(cmd) for cmd in cmds]
    procs = [Popen(arg) for arg in args]
    for i, proc in enumerate(procs):
        proc.wait()

    ### Assemble Images
    assembled_image_paths = [f"{output_image_dir}/CLEVR_{split}_{restore_digits(start_idx + f)}.png" for f in
                             range(num_images)]
    ### Check Validity
    assembled_images = [(f, 1) if os.path.exists(f) else (f, 0) for f in assembled_image_paths]

    return assembled_images


def load_resnet_backbone(dtype):
    whole_cnn = getattr(torchvision.models, 'resnet101')(pretrained=True)
    layers = [
        whole_cnn.conv1,
        whole_cnn.bn1,
        whole_cnn.relu,
        whole_cnn.maxpool,
    ]
    for i in range(3):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(whole_cnn, name))
    cnn = torch.nn.Sequential(*layers)
    cnn.type(dtype)
    cnn.eval()
    return cnn


# cnn = load_resnet_backbone(dtype = torch.cuda.FloatTensor)


def extract_features(image, cnn, dtype):
    img_size = (224, 224)
    img = imread(image)
    img = rgba2rgb(img)
    img = imresize(img, img_size)
    img = img.astype('float32')
    img = img.transpose(2, 0, 1)[None]
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
    img = (img - mean) / std
    img_var = torch.FloatTensor(img).type(dtype)
    with torch.no_grad():
        feats_var = cnn(img_var)
    return feats_var.cpu()


def find_out_how_many(split: str, output_image_dir: str) -> None:
    files = os.listdir(output_image_dir)
    split_files = [f for f in files if split in f]
    split_files = [f for f in split_files if '.png' in f]
    indexes = [int(f.split('.png')[0].split('_')[-1]) for f in split_files]
    print(f"Found {len(split_files)} items of split {split}, with max id {max(indexes)}")
    return


def merge_scenes(output_scene_dir: str, split: str) -> None:
    all_scenes = []
    for scene_path in os.listdir(output_scene_dir):
        if not scene_path.endswith('.json') or 'scenes' in scene_path or split not in scene_path:
            continue
        else:
            with open(output_scene_dir + '/' + scene_path, 'r') as f:
                all_scenes.append(json.load(f))

    output = {
        'info': {
            'split': split,
        },
        'scenes': all_scenes
    }
    with open(f'{output_scene_dir}/CLEVR_{split}_scenes.json', 'w') as f:
        json.dump(output, f)
    return


def make_questions(output_questions_file: str,
                   output_image_dir: str,
                   word_replace_dict: Union[dict, None] = None,
                   mock: bool = True,
                   mock_name: str = None,
                   mock_image_dir: Union[str, None] = None,
                   **kwargs
                   ) -> object:
    if not mock:
        cmd = question_template(**kwargs)
        arg = shlex.split(cmd)
        proc = Popen(arg, stderr=PIPE)
        out, err = proc.communicate()
        if err == bytes(('').encode('utf-8')):
            # print("Questions were generated succesfully!")
            pass
        else:
            print(err)

    with open(output_questions_file, 'r') as fin:
        data = json.load(fin)

    pairs_to_yield = {}

    for idx in range(0, len(data['questions'])):
        image = output_image_dir + '/' + data['questions'][idx]['image_filename']
        if mock_name is not None:
            id = data['questions'][idx]['image_filename'].split('_')[-1]
            image = output_image_dir + '/' + 'CLEVR_' + mock_name + '_' + id
        if mock_image_dir is not None:
            id = data['questions'][idx]['image_filename'].split('_')[-1]
            if mock_name is not None:
                image = mock_image_dir + '/' + 'CLEVR_' + mock_name + '_' + id
            else:
                image = mock_image_dir + '/' + data['questions'][idx]['image_filename']
        question = str(data['questions'][idx]['question'])
        answer = str(data['questions'][idx]['answer'])
        if word_replace_dict is None:
            pass
        else:
            for word, replacement in word_replace_dict.items():
                question = question.replace(word, replacement)
                answer = answer.replace(word, replacement)
        if image in pairs_to_yield:
            pairs_to_yield[image]['questions'].append(question)
            pairs_to_yield[image]['answers'].append(answer)
        else:
            pairs_to_yield.update({image: {'questions': [question], 'answers': [answer]}})

    def one_shot_gen():
        for image in pairs_to_yield:
            questions = pairs_to_yield[image]['questions']
            answers = pairs_to_yield[image]['answers']
            yield image, questions, answers

    return one_shot_gen
