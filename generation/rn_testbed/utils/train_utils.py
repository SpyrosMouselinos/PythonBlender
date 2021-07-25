import json
import os
import os.path as osp
import pickle
import sys

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import trange
import numpy as np
import h5py
sys.path.insert(0, osp.abspath('..'))
import torch
import yaml
from torch.utils.data import Dataset
from natsort import natsorted


def _print(something):
    print(something, flush=True)
    return


class BatchSizeScheduler:
    def __init__(self, train_ds, initial_bs, step_size, gamma, max_bs):
        self.train_ds = train_ds
        self.current_bs = initial_bs
        self.max_bs = max_bs
        self.step_size = step_size
        self.gamma = gamma
        self._current_steps = 0

    def reset(self):
        self._current_steps = 0
        return

    def step(self):
        if self.step_size != -1:
            self._current_steps += 1
            if self._current_steps % self.step_size == 0 and self._current_steps > 0:
                self.current_bs = min(self.current_bs * self.gamma, self.max_bs)
            return torch.utils.data.DataLoader(self.train_ds, batch_size=self.current_bs, shuffle=True)
        else:
            return torch.utils.data.DataLoader(self.train_ds, batch_size=self.current_bs, shuffle=True)

    def state_dict(self):
        info = {
            'current_bs': self.current_bs,
            'max_bs': self.max_bs,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'current_steps': self._current_steps
        }
        return info

    def load_state_dict(self, state_dict):
        self.current_bs = state_dict['current_bs']
        self.max_bs = state_dict['max_bs']
        self.step_size = state_dict['step_size']
        self.gamma = state_dict['gamma']
        self._current_steps = state_dict['current_steps']


def image_finder(clvr_path='data/', mode='val'):
    ### Discover and return all available images ###
    good_images = []
    available = os.listdir(clvr_path + f'/images/{mode}')
    for candidate in available:
        if mode in candidate and candidate.endswith('.png'):
            good_images.append(candidate)
    return natsorted(good_images)


def scene_parser(scenes_path='data/', mode='val'):
    with open(scenes_path + f'/CLEVR_{mode}_scenes.json', 'r') as fin:
        parsed_json = json.load(fin)
        scenes = parsed_json['scenes']
    return scenes


def question_parser(questions_path='data/', mode='val'):
    with open(questions_path + f'/CLEVR_{mode}_questions.json', 'r') as fin:
        parsed_json = json.load(fin)
        questions = parsed_json['questions']
    return questions


def single_scene_translator(scene: dict, translation: dict):
    image_index = scene['image_index']
    n_objects = len(scene['objects'])

    xs = []
    ys = []
    thetas = []
    colors = []
    materials = []
    shapes = []
    sizes = []
    for obj in scene['objects']:
        xs.append(obj['3d_coords'][0] / 3)
        ys.append(obj['3d_coords'][1] / 3)
        thetas.append(obj['3d_coords'][2] / 360)
        colors.append(translation[obj['color']])
        materials.append(translation[obj['material']])
        shapes.append(translation[obj['shape']])
        sizes.append(translation[obj['size']])

    #######################################################
    object_positions_x = torch.FloatTensor(xs + (10 - n_objects) * [0]).unsqueeze(1)
    object_positions_y = torch.FloatTensor(ys + (10 - n_objects) * [0]).unsqueeze(1)
    object_positions_t = torch.FloatTensor(thetas + (10 - n_objects) * [0]).unsqueeze(1)

    object_positions = torch.cat([object_positions_x, object_positions_y, object_positions_t], 1).view(10, 3)
    object_colors = torch.LongTensor(colors + (10 - n_objects) * [0])

    object_shapes = torch.LongTensor(shapes + (10 - n_objects) * [0])
    object_materials = torch.LongTensor(materials + (10 - n_objects) * [0])
    object_sizes = torch.LongTensor(sizes + (10 - n_objects) * [0])

    return image_index, n_objects, object_positions, object_colors, object_shapes, object_materials, object_sizes


def single_question_parser(question: dict, word_replace_dict: dict, q2index: dict, a2index: dict):
    image_index = question['image_index']
    q = str(question['question'])
    a = str(question['answer'])
    if word_replace_dict is None:
        pass
    else:
        for word, replacement in word_replace_dict.items():
            q = q.replace(word, replacement)
            a = a.replace(word, replacement)
    if q2index is None:
        pass
    else:
        q = '<START>' + ' ' + q + ' ' + '<END>'
        tokenized_q = []
        for word in q.split(' '):
            if 'bullet' in word or 'butterfly' in word:
                return image_index, None, None, None
            elif '?' in word or ';' in word:
                tokenized_q.append(q2index[word[:-1]])
                tokenized_q.append(q2index[';'])
            else:
                tokenized_q.append(q2index[word])
        q = torch.LongTensor(tokenized_q + [0] * (50 - len(tokenized_q))).view(50)
    if a2index is None:
        pass
    else:
        a = torch.LongTensor([a2index[a] - 4])

    return image_index, len(tokenized_q), q, a


def scene_image_matcher(split, translation, q2index, a2index, scenes_path='data/', questions_path='data/'):
    ### All scenes ###
    scenes = scene_parser(scenes_path, split)

    ### All questions ###
    questions = question_parser(questions_path, split)

    x_samples = []
    y_samples = []
    question_counter = 0
    for scene_counter in trange(len(scenes)):
        image_index_scene, n_objects, object_positions, object_colors, object_shapes, object_materials, object_sizes = \
            single_scene_translator(scene=scenes[scene_counter], translation=translation)
        while question_counter < len(questions):
            image_index_question, n_tokens, q, a = single_question_parser(questions[question_counter],
                                                                          word_replace_dict={'True': 'yes',
                                                                                             'False': 'no'},
                                                                          q2index=q2index,
                                                                          a2index=a2index)
            # Bad question Move on #
            if q is None and a is None:
                question_counter += 1
                continue

            if image_index_scene == image_index_question:
                types = [1] * n_objects + [0] * (10 - n_objects) + [2] * n_tokens + [0] * (50 - n_tokens)
                types = torch.LongTensor(types).view(60)
                positions = torch.LongTensor([0] * 10 + list(range(1, n_tokens + 1)) + [0] * (50 - n_tokens)).view(60)
                x_samples.append({'positions': positions,
                                  'types': types,
                                  'object_positions': object_positions,
                                  'object_colors': object_colors,
                                  'object_shapes': object_shapes,
                                  'object_materials': object_materials,
                                  'object_sizes': object_sizes,
                                  'question': q,
                                  })
                y_samples.append(a)

                # Increment and Loop #
                question_counter += 1
            else:
                # Question is for the next image #
                break
    return x_samples, y_samples


def visual_image_matcher(split, q2index, a2index, clvr_path='data/', questions_path='data/'):
    ### All images ###
    images = image_finder(clvr_path, split)

    ### All questions ###
    questions = question_parser(questions_path, split)

    x_samples = []
    y_samples = []
    question_counter = 0
    for scene_counter in trange(len(images)):
        image_index_scene = int(images[scene_counter].split('.png')[0].split(f'{split}_')[-1])
        while question_counter < len(questions):
            image_index_question, n_tokens, q, a = single_question_parser(questions[question_counter],
                                                                          word_replace_dict={'True': 'yes',
                                                                                             'False': 'no'},
                                                                          q2index=q2index,
                                                                          a2index=a2index)
            # Bad question Move on #
            if q is None and a is None:
                question_counter += 1
                continue

            if image_index_scene == image_index_question:
                x_samples.append({'image_filename': images[scene_counter],
                                  'question': q,
                                  })
                y_samples.append(a)

                # Increment and Loop #
                question_counter += 1
            elif image_index_scene < image_index_question:
                # Question is for the next image #
                break
            elif image_index_scene > image_index_question:
                # Question is for a previous image #
                question_counter += 1
    return x_samples, y_samples


class StateCLEVR(Dataset):
    """CLEVR dataset made from Scene States."""

    def __init__(self, config=None, split='val', scenes_path='data/', questions_path='data/', clvr_path=None,
                 use_cache=False):
        if osp.exists(f'data/{split}_dataset.pt'):
            with open(f'data/{split}_dataset.pt', 'rb')as fin:
                info = pickle.load(fin)
            self.split = info['split']
            self.translation = info['translation']
            self.q2index = info['q2index']
            self.a2index = info['a2index']
            self.x = info['x']
            self.y = info['y']
            print("Dataset loaded succesfully!\n")
        else:
            with open(osp.dirname(osp.dirname(__file__)) + '/translation_tables.yaml', 'r') as fin:
                translation = yaml.load(fin, Loader=yaml.FullLoader)['translation']
            with open(f'data/vocab.json', 'r') as fin:
                parsed_json = json.load(fin)
                q2index = parsed_json['question_token_to_idx']
                a2index = parsed_json['answer_token_to_idx']

            self.split = split
            # self.config = config
            self.translation = translation
            self.q2index = q2index
            self.a2index = a2index
            x, y = scene_image_matcher(self.split, self.translation, self.q2index, self.a2index, scenes_path,
                                       questions_path)
            self.x = x
            self.y = y
            print("Dataset loaded succesfully!...Saving\n")
            info = {
                'split': self.split,
                'translation': self.translation,
                'q2index': self.q2index,
                'a2index': self.a2index,
                'x': self.x,
                'y': self.y
            }
            with open(f'data/{self.split}_dataset.pt', 'wb') as fout:
                pickle.dump(info, fout)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]


class ImageCLEVR(Dataset):
    """CLEVR dataset made from Images."""

    def __init__(self, config=None, split='val', use_cache=False, clvr_path='data/', questions_path='data/',
                 scenes_path=None):
        self.use_cache = use_cache
        self.clvr_path = clvr_path
        if split == 'train':
            self.transform = transforms.Compose([transforms.Resize((128, 128)),
                                                 transforms.Pad(8),
                                                 transforms.RandomCrop((128, 128)),
                                                 transforms.RandomRotation(2.8),  # .05 rad
                                                 transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.Resize((128, 128)),
                                                 transforms.ToTensor()])
        if osp.exists(f'data/{split}_image_dataset.pt'):
            with open(f'data/{split}_image_dataset.pt', 'rb')as fin:
                info = pickle.load(fin)
            self.split = info['split']
            self.q2index = info['q2index']
            self.a2index = info['a2index']
            self.x = info['x']
            self.y = info['y']
            _print("Dataset loaded succesfully!\n")
        else:
            self.split = split
            with open(f'data/vocab.json', 'r') as fin:
                parsed_json = json.load(fin)
                self.q2index = parsed_json['question_token_to_idx']
                self.a2index = parsed_json['answer_token_to_idx']
            x, y = visual_image_matcher(split, self.q2index, self.a2index, clvr_path, questions_path)
            self.x = x
            self.y = y
            _print("Dataset loaded succesfully!...Saving\n")
            info = {
                'split': self.split,
                'q2index': self.q2index,
                'a2index': self.a2index,
                'x': self.x,
                'y': self.y
            }
            with open(f'data/{self.split}_image_dataset.pt', 'wb') as fout:
                pickle.dump(info, fout)

        self.cached_images = {}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_image_fn = self.x[idx]['image_filename']
        question = self.x[idx]['question']
        if self.use_cache:
            if current_image_fn not in self.cached_images:
                image = Image.open(self.clvr_path + f'/images/{self.split}/{current_image_fn}').convert('RGB')
                image = self.transform(image)
                self.cached_images.update({current_image_fn: image})
            else:
                image = self.cached_images[current_image_fn]
        else:
            image = Image.open(self.clvr_path + f'/images/{self.split}/{current_image_fn}').convert('RGB')
            image = self.transform(image)

        answer = self.y[idx]

        return {'image': image, 'question': question}, answer


class ImageCLEVR_HDF5(Dataset):
    """CLEVR dataset made from Images in HDF5 format."""

    def __init__(self, config=None, split='val', clvr_path='data/', questions_path='data/',
                 scenes_path=None, use_cache=False):

        self.clvr_path = clvr_path
        if split == 'train':
            self.transform = transforms.Compose([transforms.Pad(8),
                                                 transforms.RandomCrop((128, 128)),
                                                 transforms.RandomRotation(2.8),  # .05 rad
                                                 transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if osp.exists(f'data/{split}_image_dataset.pt'):
            with open(f'data/{split}_image_dataset.pt', 'rb') as fin:
                info = pickle.load(fin)
            self.split = info['split']
            self.q2index = info['q2index']
            self.a2index = info['a2index']
            self.x = info['x']
            self.y = info['y']
            _print("Dataset loaded succesfully!\n")
        else:
            self.split = split
            with open(f'data/vocab.json', 'r') as fin:
                parsed_json = json.load(fin)
                self.q2index = parsed_json['question_token_to_idx']
                self.a2index = parsed_json['answer_token_to_idx']
            x, y = visual_image_matcher(split, self.q2index, self.a2index, clvr_path, questions_path)
            self.x = x
            self.y = y
            _print("Dataset matched succesfully!\n")
            info = {
                'split': self.split,
                'q2index': self.q2index,
                'a2index': self.a2index,
                'x': self.x,
                'y': self.y
            }
            with open(f'data/{self.split}_image_dataset.pt', 'wb') as fout:
                pickle.dump(info, fout)
        if osp.exists(f'data/{split}_images.h5'):
            self.hdf5_file = np.array(h5py.File(f'data/{split}_images.h5', 'r')['image']).astype("uint8")
            self.n_images = self.hdf5_file.shape[0]
            _print("Image HDF5 loaded succesfully!\n")
        else:
            available_images = natsorted(os.listdir(self.clvr_path + f'/images/{self.split}/'))
            image_train_shape = (len(available_images), 128, 128, 3)

            f = h5py.File(f'data/{split}_images.h5', mode='w')
            f.create_dataset("image", image_train_shape, h5py.h5t.STD_U8BE)

            for i, img_addr in enumerate(available_images):
                image = Image.open(self.clvr_path + f'/images/{split}/{img_addr}').convert('RGB').resize((128, 128), 3)
                f["image"][i] = image
            f.close()
            _print("Image HDF5 written succesfully!\n")
            self.hdf5_file = np.array(h5py.File(f'data/{split}_images.h5', 'r')['image']).astype("uint8")
            self.n_images = self.hdf5_file.shape[0]
            _print("Image HDF5 loaded succesfully!\n")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        question = self.x[idx]['question']
        current_image_fn = self.x[idx]['image_filename']
        current_image_index = int(current_image_fn.split('.png')[0].split(f'{self.split}_')[-1])
        image_data = Image.fromarray(self.hdf5_file[current_image_index], 'RGB')
        image = self.transform(image_data)
        answer = self.y[idx]

        return {'image': image, 'question': question}, answer









# import matplotlib.pyplot as plt
# with open('../config.yaml', 'r') as fin:
#     config = yaml.load(fin, Loader=yaml.FullLoader)
#
# with open('../translation_tables.yaml', 'r') as fin:
#     trans = yaml.load(fin, Loader=yaml.FullLoader)
# test_loader = torch.utils.data.DataLoader(StateCLEVR(config=config, split='val'), batch_size=1, shuffle=False)
#
# with open(f'../data/vocab.json', 'r') as fin:
#     parsed_json = json.load(fin)
#     q2index = parsed_json['question_token_to_idx']
#     a2index = parsed_json['answer_token_to_idx']
#
# index2q = {v: k for k, v in q2index.items()}
# index2a = {v: k for k, v in a2index.items()}
#
# for index, (x, y) in enumerate(test_loader):
#     positions = x['positions'][0].numpy()
#     types = x['types'][0].numpy()
#     object_positions = x['object_positions'][0].numpy()
#     object_colors = [trans['reverse_translation_color'][f] if f != 0 else None for f in x['object_colors'][0].numpy()]
#     object_shapes = [trans['reverse_translation_shape'][f] if f != 0 else None for f in x['object_shapes'][0].numpy()]
#     object_materials = [trans['reverse_translation_material'][f] if f != 0 else None for f in
#                         x['object_materials'][0].numpy()]
#     object_sizes = [trans['reverse_translation_size'][f] if f != 0 else None for f in x['object_sizes'][0].numpy()]
#     q = [index2q[f] for f in x['question'][0].numpy()]
#     a = index2a[y.item()]
#
#     mmt = {
#         'cube': 's',
#         'cylinder': 'h',
#         'sphere': 'o',
#     }
#
#     mst = {
#         'large': 8,
#         'small': 6
#     }
#     plt.figure(figsize=(10, 10))
#     plt.title(f"Bird Eye View of image: {index}")
#     #plt.rcParams['axes.facecolor'] = 'black'
#     for oi in range(0, 10):
#         x = object_positions[oi][0]
#         y = object_positions[oi][1]
#         if x != 0 and y != 0:
#             plt.scatter(x=x, y=y, c=object_colors[oi], s=mst[object_sizes[oi]]**2, marker=mmt[object_shapes[oi]])
#     print(f"Question: {' '.join(q)}")
#     print(f"Answer: {a}")
#     plt.xlim(-1,1)
#     plt.ylim(-1,1)
#     plt.show()
#     if index == 20:
#         break
