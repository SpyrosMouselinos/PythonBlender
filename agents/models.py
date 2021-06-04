import json
import numpy as np
import torch
import torch.nn as nn

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

    def retrieve_configs(self, start_idx, batch_size):
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
            # image_index = gobj['image_index']
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
                # colors_.append(random.randint(0,7))
                materials_.append(translation[obj['material']])
                # materials_.append(random.randint(0,1))
                shapes_.append(translation[obj['shape']])
                # shapes_.append(random.randint(0,2))
                sizes_.append(translation[obj['size']])
                # sizes_.append(random.randint(0,1))
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


class UCB_Agent(RandAgent):
    def __init__(self, padding_number=-10,
                 scenes='/content/gdrive/MyDrive/blender_agents/official_val/CLEVR_val_scenes.json'):
        super(UCB_Agent, self).__init__()
        self.padding_number = padding_number
        if scenes is not None:
            with open(scenes, 'r') as fout:
                data = json.loads(fout.read())
            scenes = data['scenes']
        self.scenes = scenes
        self.CameraBandit = None
        self.ObjectBandit = None
        return

    def retrieve_configs(self, start_idx, batch_size):
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
                xs_.append(obj['3d_coords'][0])
                ys_.append(obj['3d_coords'][1])
                thetas_.append(obj['3d_coords'][2] % 360)

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

        ##### Affected by the Camera Bandit #####
        a = torch.abs(0.2 * torch.randn(size=(batch_size, 1)) + 0.5)
        b = torch.abs(torch.randn(size=(batch_size, 3)) + 1.0)
        camera_control = torch.cat([a, b], dim=1)
        #########################################

        key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = self.translate_camera_feature(
            camera_control.numpy().astype(float))
        (per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
        per_image_x, per_image_y, per_image_theta) = self.retrieve_configs(start_idx, batch_size)

        return (key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter), (
        per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes), (
               per_image_x, per_image_y, per_image_theta)