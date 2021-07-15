import itertools
import json

import numpy as np
import torch
import torch.nn as nn

from bandits.Bandits import UCBBandit, TSBandit
from experiment_utils.constants import translation


class RandAgent(nn.Module):
    def __init__(self, padding_number=-10):
        super(RandAgent, self).__init__()
        self.padding_number = padding_number
        return

    def translate_object_3d_pos(self, object_3d_pos):
        per_image_x = []
        per_image_y = []
        per_image_theta = []

        for batch_id in range(object_3d_pos.shape[0]):
            num_objects = sum((object_3d_pos[batch_id][:, 0] > -10) * 1)
            batch_x = object_3d_pos[batch_id][:, 0][0:num_objects]
            batch_y = object_3d_pos[batch_id][:, 1][0:num_objects]

            per_image_x.append([f for f in batch_x])
            per_image_y.append([f for f in batch_y])
            per_image_theta.append([f for f in np.random.uniform(size=(num_objects))])

        return per_image_x, per_image_y, per_image_theta

    def translate_object_scm(self, object_scms):
        per_image_objects = []
        per_image_shapes = []
        per_image_colors = []
        per_image_materials = []
        per_image_sizes = []

        for batch_id in range(object_scms.shape[0]):
            num_objects = sum((object_scms[batch_id][:, 0] > -10) * 1)
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

    def translate_camera_feature(self, cm):
        key_light = cm[:, 0]
        key_light = [f for f in key_light]
        fill_light = cm[:, 1]
        fill_light = [f for f in fill_light]
        back_light = cm[:, 2]
        back_light = [f for f in back_light]
        camera_jitter = cm[:, 3]
        camera_jitter = [f for f in camera_jitter]
        return key_light, fill_light, back_light, camera_jitter

    def forward(self, objects=-1, start_idx=0, batch_size=1, randomize_idx=False):
        assert start_idx >= 0 and start_idx <= 14990
        if randomize_idx:
            idx = np.random.randint(0, 14990)
        if objects == -1:
            object_number = torch.randint(low=2, high=7, size=(batch_size, 1), requires_grad=False)
        else:
            object_number = torch.tensor(objects).expand(batch_size, 1)

        object_scms = torch.ones(size=(batch_size, 6, 4), requires_grad=False) * self.padding_number
        object_3d_pos = torch.ones(size=(batch_size, 6, 2), requires_grad=False) * self.padding_number
        for j in range(batch_size):
            for i in range(object_number[j][0]):
                object_scms[j][i][0] = torch.randint(low=0, high=3, size=(1, 1), requires_grad=False)
                object_scms[j][i][1] = torch.randint(low=0, high=8, size=(1, 1), requires_grad=False)
                object_scms[j][i][2] = torch.randint(low=0, high=2, size=(1, 1), requires_grad=False)
                object_scms[j][i][3] = torch.randint(low=0, high=2, size=(1, 1), requires_grad=False)
                object_3d_pos[j][i] = 6.0 * (torch.rand(size=(2,)) - 0.5)
        camera_control = 3 * torch.randn(size=(batch_size, 4))
        #### Move to Numpy Format
        key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = self.translate_camera_feature(
            camera_control.numpy().astype(float))
        per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes = self.translate_object_scm(
            object_scms.numpy().astype(int))
        per_image_x, per_image_y, per_image_theta = self.translate_object_3d_pos(object_3d_pos.numpy())
        return (key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter), (
            per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
                   per_image_x, per_image_y, per_image_theta)


class InformedRandomAgent(RandAgent):
    def __init__(self, padding_number=-10,
                 scenes='/content/gdrive/MyDrive/blender_agents/official_val/CLEVR_val_scenes.json'):
        super(InformedRandomAgent, self).__init__()
        self.padding_number = padding_number
        if scenes is not None:
            with open(scenes, 'r') as fout:
                data = json.loads(fout.read())
            scenes = data['scenes']
        self.scenes = scenes
        self.noise_gen = lambda x: x
        return

    def register_noise_gen(self, noise_gen):
        self.noise_gen = noise_gen
        return

    def forward(self, objects=-1, start_idx=0, batch_size=1, randomize_idx=False):
        assert start_idx >= 0 and start_idx <= 14990
        if randomize_idx:
            idx = np.random.randint(0, 14990)

        if self.scenes is None:
            if objects == -1:
                object_number = torch.randint(low=2, high=7, size=(batch_size, 1), requires_grad=False)
            else:
                object_number = torch.tensor(objects).expand(batch_size, 1)

            object_scms = torch.ones(size=(batch_size, 6, 4), requires_grad=False) * self.padding_number
            object_3d_pos = torch.ones(size=(batch_size, 6, 2), requires_grad=False) * self.padding_number
            for j in range(batch_size):
                for i in range(object_number[j][0]):
                    object_scms[j][i][0] = torch.randint(low=0, high=3, size=(1, 1), requires_grad=False)
                    object_scms[j][i][1] = torch.randint(low=0, high=8, size=(1, 1), requires_grad=False)
                    object_scms[j][i][2] = torch.randint(low=0, high=2, size=(1, 1), requires_grad=False)
                    object_scms[j][i][3] = torch.randint(low=0, high=2, size=(1, 1), requires_grad=False)
                    object_3d_pos[j][i] = 6.0 * (torch.rand(size=(2,)) - 0.5)
            camera_control = 3 * torch.randn(size=(batch_size, 4))
            #### Move to Numpy Format
            key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = self.translate_camera_feature(
                camera_control.numpy().astype(float))
            per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes = self.translate_object_scm(
                object_scms.numpy().astype(int))
            per_image_x, per_image_y, per_image_theta = self.translate_object_3d_pos(object_3d_pos.numpy())
        else:
            a = torch.abs(0.2 * torch.randn(size=(batch_size, 1)) + 0.5)
            b = torch.abs(torch.randn(size=(batch_size, 3)) + 1.0)
            camera_control = torch.cat([a, b], dim=1)
            key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = self.translate_camera_feature(
                camera_control.numpy().astype(float))
            (per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
                per_image_x, per_image_y, per_image_theta) = self.retrieve_configs(start_idx, batch_size)
        return (key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter), (
            per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
                   per_image_x, per_image_y, per_image_theta)


class RL_Bandit_Agent(RandAgent):
    def __init__(self, scenes,padding_number=-10,
                  bandit_mode='UCB'):
        super(RL_Bandit_Agent, self).__init__()
        self.padding_number = padding_number
        if scenes is not None:
            with open(scenes, 'r') as fout:
                data = json.loads(fout.read())
            scenes = data['scenes']
        self.scenes = scenes
        self.bandit_mode = bandit_mode
        self.initialize_bandits()
        self.init_arm_translator()
        return

    def initialize_bandits(self):
        # 324 arms #
        if self.bandit_mode == 'UCB':
            self.CombineBandit = UCBBandit(k_arm=4 * 3 ** 4, epsilon=0.15, initial=0, c=1)
        elif self.bandit_mode == 'TS':
            self.CombineBandit = TSBandit(k_arm=4 * 3 ** 4, epsilon=0.15, initial=0)
        self.CombineBandit.reset()
        return

    def init_arm_translator(self, combinations_per_arm=[3, 3, 3, 3, 4]):
        hashlist = []
        for f in combinations_per_arm:
            hashlist.append(list(range(0, f)))
        hashmap = {}
        for index, arm_codes in enumerate(itertools.product(*hashlist)):
            hashmap.update({index: arm_codes})

        self.idx2armcode = hashmap
        return

    def arm_sample(self, actions):
        camera_marks = np.linspace(0, 3, 4)
        object_marks = [0.05, 0.1, 0.25, 0.5]

        res = np.zeros((len(actions), 7))
        for i, action in enumerate(actions):
            codes = self.idx2armcode[action]
            res[i, 0] = np.random.uniform(low=camera_marks[codes[0]], high=camera_marks[codes[0]] + 1)
            res[i, 1] = np.random.uniform(low=camera_marks[codes[1]], high=camera_marks[codes[1]] + 1)
            res[i, 2] = np.random.uniform(low=camera_marks[codes[2]], high=camera_marks[codes[2]] + 1)
            res[i, 3] = np.random.uniform(low=camera_marks[codes[3]], high=camera_marks[codes[3]] + 1)
            res[i, 4] = np.random.normal(0, scale=object_marks[codes[4]])
            res[i, 5] = np.random.normal(0, scale=object_marks[codes[4]])
            res[i, 6] = np.random.normal(0, scale=object_marks[codes[4]])
        return res

    def retrieve_configs(self, start_idx, batch_size, perturbations=None):
        n_objects = []
        xs = []
        ys = []
        thetas = []
        colors = []
        materials = []
        shapes = []
        sizes = []
        ###################################
        for idx in range(start_idx, start_idx + batch_size):
            gobj = self.scenes[idx]
            n_objects.append(len(gobj['objects']))
            xs_ = []
            ys_ = []
            thetas_ = []
            colors_ = []
            materials_ = []
            shapes_ = []
            sizes_ = []
            for obj in gobj['objects']:
                ######## Affected by the Object Bandit #######
                if perturbations is None:
                    xs_.append(obj['3d_coords'][0])
                    ys_.append(obj['3d_coords'][1])
                    thetas_.append(obj['3d_coords'][2] % 360)
                else:
                    x_per = perturbations[idx - start_idx, 0]
                    y_per = perturbations[idx - start_idx, 1]
                    t_per = perturbations[idx - start_idx, 2]
                    xs_.append(obj['3d_coords'][0] + x_per)
                    ys_.append(obj['3d_coords'][1] + y_per)
                    thetas_.append(obj['3d_coords'][2] + t_per % 360)

                ######## Unaffected by the Bandit #######
                colors_.append(translation[obj['color']])
                materials_.append(translation[obj['material']])
                shapes_.append(translation[obj['shape']])
                sizes_.append(translation[obj['size']])
                #########################################
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
            idx = np.random.randint(0, 14990)

        actions = self.CombineBandit.batch_act(batch_size=batch_size)
        perturbations = self.arm_sample(actions)

        ##### Affected by the Camera Bandit #####
        camera_control = perturbations[:, :-3]

        key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = self.translate_camera_feature(
            camera_control.astype(float))
        (per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
            per_image_x, per_image_y, per_image_theta) = self.retrieve_configs(start_idx, batch_size,
                                                                               perturbations=perturbations[:, -3:])

        return (key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter), (
            per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
                   per_image_x, per_image_y, per_image_theta), actions

    def learn(self, actions, rewards):
        self.CombineBandit.batch_update_step(actions=actions, rewards=rewards)
        print(self.CombineBandit.q_estimation)
        print(self.CombineBandit.action_count)
        return


class RL_Disjoint_Bandit_Agent(RandAgent):
    def __init__(self, padding_number=-10,
                 scenes='/content/gdrive/MyDrive/blender_agents/official_val/CLEVR_val_scenes.json'):
        super(RL_Disjoint_Bandit_Agent, self).__init__()
        self.padding_number = padding_number
        if scenes is not None:
            with open(scenes, 'r') as fout:
                data = json.loads(fout.read())
            scenes = data['scenes']
        self.scenes = scenes
        self.initialize_bandits()
        self.init_camera_arm_translator()
        self.init_object_arm_translator()
        return

    def initialize_bandits(self):
        # 6 intensities per choice - 4 choices = 6 arms x 4 Bandits #
        self.CameraBandits = [UCBBandit(k_arm=5, epsilon=0.1, initial=0, c=1)] * 4
        for f in self.CameraBandits:
            f.reset()
            # Color/Shape/Material/Size (Unchanged)
            # -->Arms are perturbation movement magnitude #
            # 10 intensities per choice - 3 choices = 10 arms x 3 Bandits #
        self.ObjectBandits = [UCBBandit(k_arm=10, epsilon=0.1, initial=0, c=1)] * 3
        for f in self.ObjectBandits:
            f.reset()
        return

    def init_camera_arm_translator(self, combinations_per_arm=6):
        MIN_CAMERA_SETTING = 0
        MAX_CAMERA_SETTING = 3
        CAMERA_RANGE = np.linspace(MIN_CAMERA_SETTING, MAX_CAMERA_SETTING, combinations_per_arm)
        self.idx2camera_action = {i: CAMERA_RANGE[i] for i in range(0, combinations_per_arm)}
        return

    def init_object_arm_translator(self, combinations_per_arm=10):
        MIN_OBJECT_PERTURBATION = -1
        MAX_OBJECT_PERTURBATION = 1
        OBJECT_PERTURBATION_RANGE = np.linspace(MIN_OBJECT_PERTURBATION, MAX_OBJECT_PERTURBATION, combinations_per_arm)
        self.idx2object_action = {i: OBJECT_PERTURBATION_RANGE[i] for i in range(0, combinations_per_arm)}
        return

    def arm_sample(self, camera_actions, object_actions):
        res = np.zeros((len(camera_actions[0]), 7))
        for camera_setting_id, camera_settings in enumerate(camera_actions):
            for batch_id, action in enumerate(camera_settings):
                code = self.idx2camera_action[action]
                res[batch_id, camera_setting_id] = code

        for object_perturbation_id, object_settings in enumerate(object_actions):
            for batch_id, action in enumerate(object_settings):
                code = self.idx2object_action[action]
                res[batch_id, object_perturbation_id + 4] = code
        return res

    def retrieve_configs(self, start_idx, batch_size, perturbations=None):
        n_objects = []
        xs = []
        ys = []
        thetas = []
        colors = []
        materials = []
        shapes = []
        sizes = []
        ###################################
        for idx in range(start_idx, start_idx + batch_size):
            gobj = self.scenes[idx]
            n_objects.append(len(gobj['objects']))
            xs_ = []
            ys_ = []
            thetas_ = []
            colors_ = []
            materials_ = []
            shapes_ = []
            sizes_ = []
            for obj in gobj['objects']:
                ######## Affected by the Object Bandit #######
                if perturbations is None:
                    xs_.append(obj['3d_coords'][0])
                    ys_.append(obj['3d_coords'][1])
                    thetas_.append(obj['3d_coords'][2] % 360)
                else:
                    x_per = perturbations[idx - start_idx, 0]
                    y_per = perturbations[idx - start_idx, 1]
                    t_per = perturbations[idx - start_idx, 2]
                    xs_.append(obj['3d_coords'][0] + x_per)
                    ys_.append(obj['3d_coords'][1] + y_per)
                    thetas_.append(obj['3d_coords'][2] + t_per % 360)

                ######## Unaffected by the Bandit #######
                colors_.append(translation[obj['color']])
                materials_.append(translation[obj['material']])
                shapes_.append(translation[obj['shape']])
                sizes_.append(translation[obj['size']])
                #########################################
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
            idx = np.random.randint(0, 14990)

        camera_actions = [f.batch_act(batch_size=batch_size) for f in self.CameraBandits]
        object_actions = [f.batch_act(batch_size=batch_size) for f in self.ObjectBandits]
        perturbations = self.arm_sample(camera_actions, object_actions)

        ##### Affected by the Camera Bandit #####
        camera_control = perturbations[:, :-3]

        key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = self.translate_camera_feature(
            camera_control.astype(float))
        (per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
            per_image_x, per_image_y, per_image_theta) = self.retrieve_configs(start_idx, batch_size,
                                                                               perturbations=perturbations[:, -3:])

        return (key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter), (
            per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
                   per_image_x, per_image_y, per_image_theta), camera_actions, object_actions

    def learn(self, camera_actions, object_actions, rewards):
        for f in self.CameraBandits:
            f.batch_update_step(actions=camera_actions, rewards=rewards)

        for f in self.ObjectBandits:
            f.batch_update_step(actions=object_actions, rewards=rewards)
        return


class RL_Stateful_Agent(RandAgent):
    def __init__(self, padding_number=-10,
                 scenes='/content/gdrive/MyDrive/blender_agents/official_val/CLEVR_val_scenes.json'):
        super(RL_Stateful_Agent, self).__init__()
        self.padding_number = padding_number
        if scenes is not None:
            with open(scenes, 'r') as fout:
                data = json.loads(fout.read())
            scenes = data['scenes']
        self.scenes = scenes
        self.initialize_bandits()
        self.init_arm_translator()
        return

    def initialize_bandits(self):
        # 324 arms #
        if self.bandit_mode == 'UCB':
            self.CombineBandit = UCBBandit(k_arm=4 * 3 ** 4, epsilon=0.15, initial=0, c=1)
        elif self.bandit_mode == 'TS':
            self.CombineBandit = TSBandit(k_arm=4 * 3 ** 4, epsilon=0.15, initial=0)
        self.CombineBandit.reset()
        return

    def init_arm_translator(self, combinations_per_arm=[3, 3, 3, 3, 4]):
        hashlist = []
        for f in combinations_per_arm:
            hashlist.append(list(range(0, f)))
        hashmap = {}
        for index, arm_codes in enumerate(itertools.product(*hashlist)):
            hashmap.update({index: arm_codes})

        self.idx2armcode = hashmap
        return

    def arm_sample(self, actions):
        camera_marks = np.linspace(0, 3, 4)
        object_marks = [0.05, 0.1, 0.25, 0.5]

        res = np.zeros((len(actions), 7))
        for i, action in enumerate(actions):
            codes = self.idx2armcode[action]
            res[i, 0] = np.random.uniform(low=camera_marks[codes[0]], high=camera_marks[codes[0]] + 1)
            res[i, 1] = np.random.uniform(low=camera_marks[codes[1]], high=camera_marks[codes[1]] + 1)
            res[i, 2] = np.random.uniform(low=camera_marks[codes[2]], high=camera_marks[codes[2]] + 1)
            res[i, 3] = np.random.uniform(low=camera_marks[codes[3]], high=camera_marks[codes[3]] + 1)
            res[i, 4] = np.random.normal(0, scale=object_marks[codes[4]])
            res[i, 5] = np.random.normal(0, scale=object_marks[codes[4]])
            res[i, 6] = np.random.normal(0, scale=object_marks[codes[4]])
        return res

    def retrieve_configs(self, start_idx, batch_size, perturbations=None):
        n_objects = []
        xs = []
        ys = []
        thetas = []
        colors = []
        materials = []
        shapes = []
        sizes = []
        ###################################
        for idx in range(start_idx, start_idx + batch_size):
            gobj = self.scenes[idx]
            n_objects.append(len(gobj['objects']))
            xs_ = []
            ys_ = []
            thetas_ = []
            colors_ = []
            materials_ = []
            shapes_ = []
            sizes_ = []
            for obj in gobj['objects']:
                ######## Affected by the Object Bandit #######
                if perturbations is None:
                    xs_.append(obj['3d_coords'][0])
                    ys_.append(obj['3d_coords'][1])
                    thetas_.append(obj['3d_coords'][2] % 360)
                else:
                    x_per = perturbations[idx - start_idx, 0]
                    y_per = perturbations[idx - start_idx, 1]
                    t_per = perturbations[idx - start_idx, 2]
                    xs_.append(obj['3d_coords'][0] + x_per)
                    ys_.append(obj['3d_coords'][1] + y_per)
                    thetas_.append(obj['3d_coords'][2] + t_per % 360)

                ######## Unaffected by the Bandit #######
                colors_.append(translation[obj['color']])
                materials_.append(translation[obj['material']])
                shapes_.append(translation[obj['shape']])
                sizes_.append(translation[obj['size']])
                #########################################
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
            idx = np.random.randint(0, 14990)

        actions = self.CombineBandit.batch_act(batch_size=batch_size)
        perturbations = self.arm_sample(actions)

        ##### Affected by the Camera Bandit #####
        camera_control = perturbations[:, :-3]

        key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = self.translate_camera_feature(
            camera_control.astype(float))
        (per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
            per_image_x, per_image_y, per_image_theta) = self.retrieve_configs(start_idx, batch_size,
                                                                               perturbations=perturbations[:, -3:])

        return (key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter), (
            per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
                   per_image_x, per_image_y, per_image_theta), actions

    def learn(self, actions, rewards):
        self.CombineBandit.batch_update_step(actions=actions, rewards=rewards)
        print(self.CombineBandit.q_estimation)
        print(self.CombineBandit.action_count)
        return
