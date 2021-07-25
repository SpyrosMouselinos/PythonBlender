import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module as Module


class RelationalLayerBase(Module):
    def __init__(self, config: dict):
        super().__init__()

        self.f_fc1 = nn.Linear(config["g_layers"][-1], config["f_fc1"])
        self.f_fc2 = nn.Linear(config["f_fc1"], config["f_fc2"])
        self.f_fc3 = nn.Linear(config["f_fc2"], config['num_output_classes'])
        self.dropout = nn.Dropout(p=config["rl_dropout"])


class RelationalLayer(RelationalLayerBase):
    def __init__(self, config:dict):
        super().__init__(config=config)

        self.quest_inject_position = config["question_injection_position"]
        if config['model_architecture'] == 'DeltaRN':
            self.in_size = 2 * config['hidden_dim']
        elif config['model_architecture'] == 'DeltaRNFP':
            self.in_size = 2 * config['visual_hidden_dim']
        self.qst_size = config['max_question_tokens_per_scene']

        # create all g layers
        self.g_layers = []
        self.g_layers_size = config["g_layers"]
        for idx, g_layer_size in enumerate(config["g_layers"]):
            in_s = self.in_size if idx == 0 else config["g_layers"][idx - 1]
            out_s = g_layer_size
            if idx == self.quest_inject_position:
                # create the h layer. Now, for better code organization, it is part of the g layers pool.
                l = nn.Linear(in_s + config['hidden_dim'], out_s)
            else:
                # create a standard g layer.
                l = nn.Linear(in_s, out_s)
            self.g_layers.append(l)
        self.g_layers = nn.ModuleList(self.g_layers)

    def forward(self, x, qst):
        """g"""
        b, d, k = x.size()
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1, d, 1)
        qst = torch.unsqueeze(qst, 2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x, 1)
        x_i = x_i.repeat(1, d, 1, 1)
        x_j = torch.unsqueeze(x, 2)
        x_j = x_j.repeat(1, 1, d, 1)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)

        # reshape for passing through network
        x_ = x_full.view(b * d ** 2, self.in_size)

        # create g and inject the question at the position pointed by quest_inject_position.
        for idx, (g_layer, g_layer_size) in enumerate(zip(self.g_layers, self.g_layers_size)):
            if idx == self.quest_inject_position:
                in_size = self.in_size if idx == 0 else self.g_layers_size[idx - 1]

                # questions inserted
                x_img = x_.view(b, d, d, in_size)
                qst = qst.repeat(1, 1, d, 1)
                x_concat = torch.cat([x_img, qst], 3)

                # h layer
                x_ = x_concat.view(b * (d ** 2), in_size + qst.size()[-1])
                x_ = g_layer(x_)
                x_ = F.relu(x_)
            else:
                x_ = g_layer(x_)
                x_ = F.relu(x_)

        # reshape again and sum
        x_g = x_.view(b, d ** 2, self.g_layers_size[-1])
        x_g = x_g.sum(1).squeeze(1)

        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = self.dropout(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)

        return F.log_softmax(x_f, dim=1)


