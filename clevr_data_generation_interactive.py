import os
import sys
import torch
import numpy as np
import json
import random
import types
import time
import matplotlib
import matplotlib.pyplot as plt
import shlex
from subprocess import Popen, PIPE
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize as imresize
from generation.my_run_model import inference_with_cnn_sa, inference_with_iep, inference_with_film
"""### Set Randomness
***
"""

random.seed(666)
np.random.seed(666)
torch.manual_seed(666)
    
def main(use_gpu=1, split='rendered', img_offset=0, max_episodes=1, batch_size=1):

    USE_GPU = use_gpu
    SPLIT = split

    """### Path Setting
    ***
    """
    up_to_here = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    blender_version = 'blender2.79'
    output_image_dir = up_to_here + '/images'
    output_scene_dir = up_to_here + '/scenes'
    output_scene_file = up_to_here + '/scenes/CLEVR_{SPLIT}_scenes.json'
    output_question_file = up_to_here + '/questions/CLEVR_{SPLIT}_questions.json'
    """### Create Folders if they do not exist
    ***
    """

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


    """### Use this as a translation template
    ***
    """

    translation = {
        "cube": 0,
        "sphere": 1,
        "cylinder": 2,
        "gray": 0,
        "red": 1,
        "blue": 2,
        "green": 3,
        "brown": 4,
        "purple": 5,
        "cyan": 6,
        "yellow": 7,
        "rubber": 0,
        "metal": 1,
        "large": 0,
        "small": 1
        }

    """### Item Constructor
    ***
    """

    def item_creation(shape=None, color=None, material=None, size=None, x=None, y=None, theta=None):
      """
        Creates the dictionary argparse equivalent of an item.
        Leaving something None, takes a random guess.
      """

      shapes = {
        0 : "\"SmoothCube_v2\"",
        1 : "\"Sphere\"",
        2 : "\"SmoothCylinder\""
      }

      if shape is not None:
        ret_shape = shapes[shape]
      else:
        ret_shape = shapes[random.randint(0,2)]

      colors = {
        0: "\"gray\"",   #[87, 87, 87],   
        1: "\"red\"",    #[173, 35, 35],  
        2: "\"blue\"",   #[42, 75, 215],  
        3: "\"green\"",  #[29, 105, 20],  
        4: "\"brown\"",  #[129, 74, 25],  
        5: "\"purple\"", #[129, 38, 192], 
        6: "\"cyan\"",     #[41, 208, 208], 
        7: "\"yellow\""  #[255, 238, 51]
      }
      if color is not None:
        ret_color = colors[color]
      else:
        ret_color = colors[random.randint(0,7)]

      materials = {
        0: "\"Rubber\"",
        1: "\"MyMetal\""
      }

      if material is not None:
        ret_material = materials[material]
      else:
        ret_material = materials[random.randint(0,1)]

      sizes = {
        0: "\"large\"", #0.7,
        1: "\"small\"", #0.35
      }
      if size is not None:
        ret_size = sizes[size]
      else:
        ret_size = sizes[random.randint(0,1)]

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


      constructor = '{"object":'+ str(ret_shape) + ',"color":'+str(ret_color)+',"material":'+str(ret_material)+',"size":'+str(ret_size)+',"theta":'+str(ret_theta)+',"x":'+str(ret_x)+',"y":'+str(ret_y)+'}'
      return constructor

    ### Single Scene Constructor
    def scene_creation(shape_list=[None], color_list=[None], material_list=[None], size_list=[None], x_list=[None], y_list=[None], theta_list=[None], multiply=1):
      """
        Creates the dictionary argparse equivalent of a scene.
        Leaving something None, takes a random guess.
      """
      scene_list = '['
      for sh,c,m,si,x,y,t in zip(shape_list*multiply, color_list*multiply, material_list*multiply, size_list*multiply, x_list*multiply, y_list*multiply, theta_list*multiply):
        scene_list += item_creation(sh,c,m,si,x,y,t)
        scene_list += ','
      scene_list = ''.join([f for f in scene_list[:-1]]) + ']'
      return scene_list

    ### Multi Scene Constructor
    def multi_scene_creation(list_of_shape_list=[[None],[None]],
                            list_of_color_list=[[None],[None]],
                            list_of_material_list=[[None],[None]],
                            list_of_size_list=[[None],[None]], 
                            list_of_x_list=[[None],[None]],
                            list_of_y_list=[[None],[None]],
                            list_of_theta_list=[[None],[None]]):
      """
        Creates the dictionary argparse equivalent of multiple scenes.
        Leaving something None, takes a random guess.
      """
      multi_scene_dict = '{'
      counter = 0
      for sh_list,c_list,m_list,si_list,x_list,y_list,t_list in zip(list_of_shape_list, list_of_color_list, list_of_material_list, list_of_size_list, list_of_x_list, list_of_y_list, list_of_theta_list):
        multi_scene_dict += f'"{counter}":'
        multi_scene_dict += scene_creation(sh_list,c_list,m_list,si_list,x_list,y_list,t_list)
        multi_scene_dict += ','
        counter += 1
      multi_scene_dict = ''.join([f for f in multi_scene_dict[:-1]]) + '}'
      return multi_scene_dict

    """### Create a function that can distribute the image generation over many processes
    ***
    """

    def distribute(  key_light_jitter=[1,2],
                    fill_light_jitter=[1,2],
                    back_light_jitter=[1,2],
                    camera_jitter=[1,2],
                    num_objects=[2,2],
                    per_image_shapes = [[None,None],[None,None]],
                    per_image_colors = [[None,None],[None,None]],
                    per_image_materials = [[None,None],[None,None]],
                    per_image_sizes = [[None,None],[None,None]],
                    per_image_x = [[2.0,-3.0],[1.0,-1.5]],
                    per_image_y = [[2.5,1.5], [2.5,1.5]],
                    per_image_theta = [[None,None],[None,None]],
                    num_images=2,
                    split='test',
                    start_idx=1,
                    workers=1,
                    clean_before=False,
                    assemble_after=False,
                    ):
      ### Let's see if we can multi-process this thing ###
      ### Well if that is the case that must happen on Image Level ###
      ### So first check if we can split the arguments per-image-per-worker ###
      collect_locals = locals().items()
      effective_workers = workers
      if num_images <= workers:
        images_per_worker = [1] * num_images
        effective_workers = list(range(num_images))
      elif num_images > workers:
        images_per_worker = [0] * workers
        for i in range(0, num_images):
          images_per_worker[i%workers] += 1
        ### Check if the processed can be broken down in a "fair" way ###
        effective_workers = list(range(workers))
      start_indexes = [start_idx] * len(effective_workers)
      for i in range(1, len(effective_workers)):
        start_indexes[i] = start_indexes[i-1] + images_per_worker[i-1]
      

      effective_args = {}
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
            'object_properties' : None,
            'num_images': None,
            'split':None,
            'start_idx':None,
        }})

      for argument, passed_value in collect_locals:
        if argument == 'clean_before' or argument == 'assemble_after':
          continue
        if argument in ['num_images','split','start_idx','workers']:
          if argument == 'num_images':
            for i in effective_workers:
                effective_args[f'worker_{i}'].update({'num_images':images_per_worker[i]})
          elif argument == 'split':
            for i in effective_workers:
              effective_args[f'worker_{i}'].update({'split': passed_value})
          elif argument == 'start_idx':
            for i in effective_workers:
              effective_args[f'worker_{i}'].update({'start_idx': start_indexes[i]})
          elif argument == 'workers':
            pass
          else:
            raise ValueError("You passed an argument that made it up to here?")
          #################################################################################
        else:
          for i in effective_workers:
            for j in range(images_per_worker[i]):
              effective_args[f'worker_{i}'][argument].append(passed_value[start_indexes[i] - min(start_indexes) + j])
      
      for i in effective_workers:
        effective_args[f'worker_{i}']['object_properties'] = multi_scene_creation(list_of_shape_list = effective_args[f'worker_{i}']['per_image_shapes'],
                                                                                  list_of_color_list = effective_args[f'worker_{i}']['per_image_colors'],
                                                                                  list_of_material_list = effective_args[f'worker_{i}']['per_image_materials'],
                                                                                  list_of_size_list = effective_args[f'worker_{i}']['per_image_sizes'],
                                                                                  list_of_x_list = effective_args[f'worker_{i}']['per_image_x'],
                                                                                  list_of_y_list = effective_args[f'worker_{i}']['per_image_y'],
                                                                                  list_of_theta_list = effective_args[f'worker_{i}']['per_image_theta'],
                                                                                  )
      return effective_args

    """### Let's Make a Function that will act as the command template
    ***
    """

    def dict_to_binary(the_dict):
        string = json.dumps(the_dict)
        binary = '__'.join(format(ord(letter), 'b') for letter in string)
        return binary


    def binary_to_dict(the_binary):
        jsn = ''.join(chr(int(x, 2)) for x in the_binary.split('__'))
        d = json.loads(jsn)  
        return d

        
    def command_template(num_images,
                        key_light_jitter,
                        fill_light_jitter,
                        back_light_jitter,
                        camera_jitter,
                        num_objects,
                        object_properties,
                        split,
                        start_idx,
                        output_image_dir=output_image_dir,
                        output_scene_dir=output_scene_dir,
                        output_scene_file=output_scene_file,
                        **kwargs
                        ):
      
      key_light_jitter = [str(f) for f in key_light_jitter]
      fill_light_jitter = [str(f) for f in fill_light_jitter]
      back_light_jitter = [str(f) for f in back_light_jitter]
      camera_jitter = [str(f) for f in camera_jitter]
      num_objects = [str(f) for f in num_objects]
      cmd_template = f'{up_to_here}/generation/blender2.79/blender -noaudio --background --python {up_to_here}/generation/det_render_images.py -- --num_images={num_images} \
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
          --use_gpu={USE_GPU} --render_num_samples=256 --width=480 --height=320 --start_idx={start_idx}'
      return cmd_template

    def question_template(input_scene_file,
                          output_questions_file,
                          templates_per_image,
                          instances_per_template,
                          start_idx,
                          **kwargs):
      cmd_template = f'python ./generation/generate_questions.py \
                      --input_scene_file={input_scene_file} \
                      --output_questions_file={output_questions_file} \
                      --templates_per_image={templates_per_image} \
                      --instances_per_template={instances_per_template} \
                      --scene_start_idx={start_idx}'
      return cmd_template

    """### Let's Create a Python Function that can be hi-jacked into creating images
    ***
    """

    def restore_digits(number, digits=6):
        number_as_str = str(number)
        n_chars = len(number_as_str)
        n_pad = max(0,digits - n_chars)
        padded_number = ['0'] * n_pad + [f for f in number_as_str]
        return ''.join(padded_number)

    def render_image(key_light_jitter=[1,2,3,4,5],
                    fill_light_jitter=[1,2,3,4,5],
                    back_light_jitter=[1,2,3,4,5],
                    camera_jitter=[1,2,3,4,5],
                    num_objects=[2,2,2,2,1],
                    per_image_shapes = [[None,None],[None,None],[None,None],[None,None],[None]],
                    per_image_colors = [[None,None],[None,None],[None,None],[None,None],[None]],
                    per_image_materials = [[None,None],[None,None],[None,None],[None,None],[None]],
                    per_image_sizes = [[None,None],[None,None],[None,None],[None,None],[None]],
                    per_image_x = [[2.0, -3.0], [2.0, -3.0], [2.0, -3.0], [2.0, -3.0], [2.0]],
                    per_image_y = [[2.5, 1.5],[2.5, 1.5],[2.5, 1.5],[2.5, 1.5],[2.5]],
                    per_image_theta = [[None,None],[None,None],[None,None],[None,None],[None]],
                    num_images=5,
                    split='rendered',
                    start_idx=1,
                    workers=3,
                    clean_before=False,
                    assemble_after=False,
                    ):
        collected_locals = locals().items()
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
                        os.remove(output_scene_dir + '/' +target)
                    except:
                        pass
        effective_args = distribute(**dict(collected_locals))
        cmds = [command_template(**effective_args[f]) for f in effective_args.keys()]
        

        args = [shlex.split(cmd) for cmd in cmds]
        procs = [Popen(arg) for arg in args]
        for i, proc in enumerate(procs):
          proc.wait()
          print(f"Process {i} rendering finished...")

        ### Assemble Images
        assembled_image_paths = [f"{output_image_dir}/CLEVR_{split}_{restore_digits(start_idx + f)}.png" for f in range(num_images)]
        ### Check Validity
        assembled_images = [(f,1) if os.path.exists(f) else (f,0) for f in assembled_image_paths]

        return assembled_images

    """### Warm-Up the Creation of Cuda Cache
    ***
    """
    if USE_GPU == 1:
      print("Warming Up Cuda, please wait...")
      render_image([1],[1],[1],[1],[1],[[None]],[[None]],[[None]],[[None]],[[1.9]],[[2.5]],[[0]],1,'warmup',0)
      print("Warmup Complete!")

    """### Let's Make a Question Generator
    ***
    """

    def make_questions(input_scene_file=output_scene_file,
                      output_questions_file=output_question_file,
                      templates_per_image=10,
                      instances_per_template=1,
                      start_idx=0,
                      word_replace_dict=None
                    ):
        collected_locals = locals().items()
        cmd = question_template(**dict(collected_locals))
        arg = shlex.split(cmd)
        proc = Popen(arg, stderr=PIPE)
        out, err = proc.communicate()
        if err == bytes(('').encode('utf-8')):
          #print("Questions were generated succesfully!")
          pass
        else:
            print(err)
        with open(output_questions_file, 'r') as fin:
          data = json.load(fin)

        pairs_to_yield ={}

        for idx in range(0,len(data['questions'])):
          image = output_image_dir + '/' + data['questions'][idx]['image_filename']
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
            pairs_to_yield.update({image:{'questions':[question],'answers':[answer]}})

        def one_shot_gen():
            for image in pairs_to_yield:
                questions = pairs_to_yield[image]['questions']
                answers   = pairs_to_yield[image]['answers']
                yield image, questions, answers
        return one_shot_gen


    class InformedRandomAgent(nn.Module):
        def __init__(self, padding_number=-10, scenes=f'{up_to_here}/official_val/CLEVR_val_scenes.json'):
            super(InformedRandomAgent, self).__init__()
            self.padding_number = padding_number
            if scenes is not None:
                try:
                    with open(scenes, 'r') as fout:
                        data = json.loads(fout.read())
                    scenes = data['scenes']
                except FileNotFoundError:
                    print("Official Validation Scenes not found at: {up_to_here}/official_val/CLEVR_val_scenes.json ...")
                    scenes = None
            self.scenes = scenes
            self.noise_gen = lambda x: x
            return

        def register_noise_gen(self, noise_gen):
            self.noise_gen = noise_gen
            return 

    #staticmethod
        def translate_object_3d_pos(self,object_3d_pos):
            per_image_x = []
            per_image_y = []
            per_image_theta = []

            for batch_id in range(object_3d_pos.shape[0]):
                num_objects = sum((object_3d_pos[batch_id][:, 0] > padding_number) * 1)
                batch_x = object_3d_pos[batch_id][:, 0][0:num_objects]
                batch_y = object_3d_pos[batch_id][:, 1][0:num_objects]
            
                per_image_x.append([f for f in batch_x])
                per_image_y.append([f for f in batch_y])
                per_image_theta.append([f for f in np.random.uniform(size=(num_objects))])


            return per_image_x, per_image_y, per_image_theta

        #staticmethod
        def translate_object_scm(self,object_scms):
            per_image_objects = []
            per_image_shapes = []
            per_image_colors = []
            per_image_materials = []
            per_image_sizes = []

            for batch_id in range(object_scms.shape[0]):
                num_objects = sum((object_scms[batch_id][:, 0] > padding_number) * 1)
                batch_shapes = object_scms[batch_id][:, 0][0:num_objects]
                batch_colors = object_scms[batch_id][:, 1][0:num_objects]
                batch_materials = object_scms[batch_id][:, 2][0:num_objects]
                batch_sizes = object_scms[batch_id][:, 3][0:num_objects]

                per_image_objects.append(num_objects)
                per_image_shapes.append([f for f in batch_shapes])
                per_image_colors.append([f for f in batch_colors])
                per_image_materials.append([f for f in batch_materials])
                per_image_sizes.append([f for f in batch_sizes])

            return per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes

        #staticmethod
        def translate_camera_feature(self,cm):
            key_light = cm[:,0]
            key_light = [f for f in key_light]
            fill_light = cm[:,1]
            fill_light = [f for f in fill_light]
            back_light = cm[:,2]
            back_light = [f for f in back_light]
            camera_jitter =cm[:,3]
            camera_jitter = [f for f in camera_jitter]
            return key_light, fill_light, back_light, camera_jitter

        def retrieve_configs(self, start_idx, batch_size):
            n_objects = []
            xs = []
            ys = []
            thetas = []
            colors =[]
            materials = []
            shapes = []
            sizes = []
            ###################################
            for idx in range(start_idx, start_idx + batch_size):
                gobj = self.scenes[idx]
                #image_index = gobj['image_index']
                n_objects.append(len(gobj['objects']))
                xs_ = []
                ys_ = []
                thetas_ = []
                colors_ = []
                materials_ = []
                shapes_ = []
                sizes_ = []
                for obj in gobj['objects']:
                    xs_.append(self.noise_gen(obj['3d_coords'][0]))
                    ys_.append(self.noise_gen(obj['3d_coords'][1]))
                    thetas_.append(self.noise_gen(obj['3d_coords'][2]) % 360)
                    colors_.append(translation[obj['color']])
                    #colors_.append(random.randint(0,7))
                    materials_.append(translation[obj['material']])
                    #materials_.append(random.randint(0,1))
                    shapes_.append(translation[obj['shape']])
                    #shapes_.append(random.randint(0,2))
                    sizes_.append(translation[obj['size']])
                    #sizes_.append(random.randint(0,1))
                xs.append(xs_)
                ys.append(ys_)
                thetas.append(thetas_)
                colors.append(colors_)
                materials.append(materials_)
                shapes.append(shapes_)
                sizes.append(sizes_)
            return (n_objects, shapes, colors, materials, sizes), (xs, ys, thetas)

        def forward(self, objects=-1, start_idx=0, batch_size=1, randomize_idx=False):
            assert start_idx >= 0 and start_idx <= 14990
            if randomize_idx:
                idx = np.random.randint(0,14990)
            
            if self.scenes is None:
                if objects == -1:
                    object_number  = torch.randint(low=2,  high=7, size=(batch_size,1),requires_grad=False)
                else:
                    object_number = torch.tensor(objects).expand(batch_size,1)

                object_scms = torch.ones(size=(batch_size,6,4),requires_grad=False) * self.padding_number
                object_3d_pos = torch.ones(size=(batch_size,6,2),requires_grad=False) * self.padding_number
                for j in range(batch_size):
                    for i in range(object_number[j][0]):
                        object_scms[j][i][0] = torch.randint(low=0,  high=3, size=(1,1), requires_grad=False)
                        object_scms[j][i][1] = torch.randint(low=0,  high=8, size=(1,1), requires_grad=False)
                        object_scms[j][i][2] = torch.randint(low=0,  high=2, size=(1,1), requires_grad=False)
                        object_scms[j][i][3] = torch.randint(low=0,  high=2, size=(1,1), requires_grad=False)
                        object_3d_pos[j][i] = 6.0 * (torch.rand(size=(2,)) - 0.5)
                camera_control = 3 * torch.randn(size=(batch_size,4))
                #### Move to Numpy Format
                key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = sellf.translate_camera_feature(camera_control.numpy().astype(float))
                per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes = self.translate_object_scm(object_scms.numpy().astype(int))
                per_image_x, per_image_y, per_image_theta = self.translate_object_3d_pos(object_3d_pos.numpy())
            else:
                a = torch.abs(0.2 * torch.randn(size=(batch_size,1)) + 0.5)
                b = torch.abs(torch.randn(size=(batch_size,3)) + 1.0)
                camera_control = torch.cat([a,b], dim=1)
                key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = self.translate_camera_feature(camera_control.numpy().astype(float))
                (per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (per_image_x, per_image_y, per_image_theta) = self.retrieve_configs(start_idx, batch_size)
            return (key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter), (per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (per_image_x, per_image_y, per_image_theta)

    """### Optimize speed by loading the ResNet Feature Extractor outside the inference loop
    ***
    """

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

    #cnn = load_resnet_backbone(dtype = torch.cuda.FloatTensor)


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



    IMG_OFFSET = img_offset
    episodes = 0 + IMG_OFFSET
    MAX_EPISODES = max_episodes  + IMG_OFFSET
    BATCH_SIZE = batch_size
    global_accuracy_sa = []
    global_accuracy_iep = []
    global_accuracy_film = []
    global_generation_success_rate = 0
    agent = InformedRandomAgent()

    def add_gaussian(x, loc=0, scale=0.1):
        return x + np.random.normal(loc, scale)

    def add_uniform(x, low=-0.1, high=0.1):
        return x + np.random.uniform(low, high)

    agent.register_noise_gen(add_uniform)

    start_time = time.time()
    while episodes < MAX_EPISODES:

        ## Suggest a configuration
        camera_control, object_scms, object_3d_pos = agent(objects=-1, start_idx=episodes*BATCH_SIZE, batch_size=BATCH_SIZE, randomize_idx=False)
        
        ### Move to Numpy Format
        key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = camera_control
        per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes = object_scms
        per_image_x, per_image_y, per_image_theta = object_3d_pos
        #### Render it ###
        attempted_images = render_image(key_light_jitter,
                                        fill_light_jitter,
                                        back_light_jitter,
                                        camera_jitter,
                                        per_image_objects,
                                        per_image_shapes,
                                        per_image_colors,
                                        per_image_materials,
                                        per_image_sizes,
                                        per_image_x,
                                        per_image_y,
                                        per_image_theta,
                                        num_images=BATCH_SIZE,
                                        workers=1,
                                        split=f'{SPLIT}',
                                        assemble_after=False,
                                        start_idx=episodes*BATCH_SIZE
                                        )
        correct_images = [f[0] for f in attempted_images if f[1] == 1]
        print(f"Made {len(correct_images)} out of {BATCH_SIZE} Images")
        global_generation_success_rate += len(correct_images)
        episodes +=1
    end_time = time.time()
    print(f"Took {end_time - start_time} time for {global_generation_success_rate} images")
    print(f"Seconds per Image:  {round(end_time - start_time,2) / global_generation_success_rate}")
    print(f"Generator Success Rate: {round(global_generation_success_rate / (MAX_EPISODES * BATCH_SIZE),2)}")
    print()
    print("Assembling Scenes into 1 scene...")
    all_scenes = []
    for scene_path in os.listdir(output_scene_dir):
        if not scene_path.endswith('.json') or 'scenes' in scene_path:
            continue
        else:
            with open(output_scene_dir + '/' +scene_path, 'r') as f:
                all_scenes.append(json.load(f))

    output = {
        'info': {
            'split': split,
        },
        'scenes': all_scenes
    }
    with open(output_scene_file, 'w') as f:
        json.dump(output, f)

    print("Generating Questions...")
    generator = make_questions(word_replace_dict={'True':'yes','False':'no'}, output_questions_file = f'{up_to_here}/questions/CLEVR_{SPLIT}_questions.json')
    print("Questions Created...")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', default=1)
    parser.add_argument('--split', default='rendered')
    parser.add_argument('--img_offset', default=0)
    parser.add_argument('--max_episodes', default=1)
    parser.add_argument('--batch_size', default=1)

    args = parser.parse_args()
    if __name__ == '__main__':
        main(args.use_gpu, args.split
             args.img_offset, args.max_episodes, args.batch_size)
        