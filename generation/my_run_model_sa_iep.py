import json

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize as imresize
from torch.autograd import Variable

SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


### Performance of SA+MLP - Got 109544 / 149991 = 73.03 correct
### Performance of 9k IEP - Got 132473 / 149991 = 88.32 correct
### Performance of 18k IEP - Got 142719 / 149991 = 95.15 correct
### Performance of 70k IEP - Got 145390 / 149991 = 96.93 correct

def _dataset_to_tensor(dset, mask=None):
    arr = np.asarray(dset, dtype=np.int64)
    if mask is not None:
        arr = arr[mask]
    tensor = torch.LongTensor(arr)
    return tensor


def prefix_to_tree(program_prefix):
    program_prefix = [x for x in program_prefix]

    def helper():
        cur = program_prefix.pop(0)
        return {
            'function': cur['function'],
            'value_inputs': [x for x in cur['value_inputs']],
            'inputs': [helper() for _ in range(get_num_inputs(cur))],
        }

    return helper()


def tree_to_list(program_tree):
    # First count nodes
    def count_nodes(cur):
        return 1 + sum(count_nodes(x) for x in cur['inputs'])


def prefix_to_list(program_prefix):
    return tree_to_list(prefix_to_tree(program_prefix))


class ClevrDataset(Dataset):
    def __init__(self, question_h5, feature_h5, vocab, mode='prefix',
                 image_h5=None, max_samples=None, question_families=None,
                 image_idx_start_from=None):
        mode_choices = ['prefix', 'postfix']
        if mode not in mode_choices:
            raise ValueError('Invalid mode "%s"' % mode)
        self.image_h5 = image_h5
        self.vocab = vocab
        self.feature_h5 = feature_h5
        self.mode = mode
        self.max_samples = max_samples

        mask = None
        if question_families is not None:
            # Use only the specified families
            all_families = np.asarray(question_h5['question_families'])
            N = all_families.shape[0]
            print(question_families)
            target_families = np.asarray(question_families)[:, None]
            mask = (all_families == target_families).any(axis=0)
        if image_idx_start_from is not None:
            all_image_idxs = np.asarray(question_h5['image_idxs'])
            mask = all_image_idxs >= image_idx_start_from

        # Data from the question file is small, so read it all into memory
        print('Reading question data into memory')
        self.all_questions = _dataset_to_tensor(question_h5['questions'], mask)
        self.all_image_idxs = _dataset_to_tensor(question_h5['image_idxs'], mask)
        self.all_programs = None
        if 'programs' in question_h5:
            self.all_programs = _dataset_to_tensor(question_h5['programs'], mask)
        self.all_answers = _dataset_to_tensor(question_h5['answers'], mask)

    def __getitem__(self, index):
        question = self.all_questions[index]
        image_idx = self.all_image_idxs[index]
        answer = self.all_answers[index]
        program_seq = None
        if self.all_programs is not None:
            program_seq = self.all_programs[index]

        image = None
        if self.image_h5 is not None:
            image = self.image_h5['images'][image_idx]
            image = torch.FloatTensor(np.asarray(image, dtype=np.float32))

        feats = self.feature_h5['features'][image_idx]
        feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))

        program_json = None
        if program_seq is not None:
            program_json_seq = []
            for fn_idx in program_seq:
                fn_str = self.vocab['program_idx_to_token'][int(fn_idx.cpu().numpy())]
                if fn_str == '<START>' or fn_str == '<END>': continue
                fn = str_to_function(fn_str)
                program_json_seq.append(fn)
            if self.mode == 'prefix':
                program_json = prefix_to_list(program_json_seq)

        return (question, image, feats, answer, program_seq, program_json)

    def __len__(self):
        if self.max_samples is None:
            return self.all_questions.size(0)
        else:
            return min(self.max_samples, self.all_questions.size(0))


class ClevrDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'question_h5' not in kwargs:
            raise ValueError('Must give question_h5')
        if 'feature_h5' not in kwargs:
            raise ValueError('Must give feature_h5')
        if 'vocab' not in kwargs:
            raise ValueError('Must give vocab')

        feature_h5_path = kwargs.pop('feature_h5')
        print('Reading features from ', feature_h5_path)
        self.feature_h5 = h5py.File(feature_h5_path, 'r')

        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = kwargs.pop('image_h5')
            print('Reading images from ', image_h5_path)
            self.image_h5 = h5py.File(image_h5_path, 'r')

        vocab = kwargs.pop('vocab')
        mode = kwargs.pop('mode', 'prefix')

        question_families = kwargs.pop('question_families', None)
        max_samples = kwargs.pop('max_samples', None)
        question_h5_path = kwargs.pop('question_h5')
        image_idx_start_from = kwargs.pop('image_idx_start_from', None)
        print('Reading questions from ', question_h5_path)
        with h5py.File(question_h5_path, 'r') as question_h5:
            self.dataset = ClevrDataset(question_h5, self.feature_h5, vocab, mode,
                                        image_h5=self.image_h5,
                                        max_samples=max_samples,
                                        question_families=question_families,
                                        image_idx_start_from=image_idx_start_from)
        kwargs['collate_fn'] = clevr_collate
        super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)

    def close(self):
        if self.image_h5 is not None:
            self.image_h5.close()
        if self.feature_h5 is not None:
            self.feature_h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def clevr_collate(batch):
    transposed = list(zip(*batch))
    question_batch = default_collate(transposed[0])
    image_batch = transposed[1]
    if any(img is not None for img in image_batch):
        image_batch = default_collate(image_batch)
    feat_batch = transposed[2]
    if any(f is not None for f in feat_batch):
        feat_batch = default_collate(feat_batch)
    answer_batch = default_collate(transposed[3])
    program_seq_batch = transposed[4]
    if transposed[4][0] is not None:
        program_seq_batch = default_collate(transposed[4])
    program_struct_batch = transposed[5]
    return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch, program_struct_batch]


def inference_with_cnn_sa(questions='Is the number of cubes less than the number of small metal things?',
                          input_questions_h5='../data/val_questions.h5',
                          image='../img/CLEVR_val_000013.png',
                          feats=None,
                          input_features_h5='../data/val_features.h5',
                          baseline_model='../data/models/CLEVR/cnn_lstm_sa_mlp.pt',
                          use_gpu=1,
                          cnn_model='resnet101',
                          cnn_model_stage=3,
                          image_width=224,
                          image_height=224,
                          external=False,
                          batch_size=1,
                          num_samples=None,
                          family_split_file=None,
                          sample_argmax=1,
                          temperature=1.0,
                          vocab_json='../data/vocab.json'):
    collect_locals = dict(locals())
    model, _ = load_baseline(baseline_model)
    model.eval()
    if batch_size == 1:
        answer = run_single_example(collect_locals, model)
        return answer
    else:
        run_batch(collect_locals, model)
        return


def inference_with_iep(questions='Is the number of cubes less than the number of small metal things?',
                       image='../img/CLEVR_val_000013.png',
                       feats=None,
                       input_questions_h5='../data/val_questions.h5',
                       input_features_h5='../data/val_features.h5',
                       program_generator='../data/models/CLEVR/program_generator_9k.pt',
                       execution_engine='../data/models/CLEVR/execution_engine_9k.pt',
                       use_gpu=1,
                       cnn_model='resnet101',
                       cnn_model_stage=3,
                       image_width=224,
                       image_height=224,
                       external=False,
                       batch_size=1,
                       num_samples=None,
                       family_split_file=None,
                       sample_argmax=1,
                       temperature=1.0,
                       vocab_json='../data/vocab.json'):
    collect_locals = dict(locals())
    program_generator, _ = load_program_generator(program_generator)
    program_generator.eval()
    execution_engine, _ = load_execution_engine(execution_engine, verbose=False)
    execution_engine.eval()
    model = (program_generator, execution_engine)
    if batch_size == 1:
        answer = run_single_example(collect_locals, model)
        return answer
    else:
        run_batch(collect_locals, model)
        return


def tokenize(s, delim=' ',
             add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    token_to_count = {}
    tokenize_kwargs = {
        'delim': delim,
        'punct_to_keep': punct_to_keep,
        'punct_to_remove': punct_to_remove,
    }
    for seq in sequences:
        seq_tokens = tokenize(seq, **tokenize_kwargs,
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def expand_embedding_vocab(embed, token_to_idx, word2vec=None, std=0.01):
    old_weight = embed.weight.data
    old_N, D = old_weight.size()
    new_N = 1 + max(idx for idx in token_to_idx.values())
    new_weight = old_weight.new(new_N, D).normal_().mul_(std)
    new_weight[:old_N].copy_(old_weight)

    if word2vec is not None:
        num_found = 0
        assert D == word2vec['vecs'].size(1), 'Word vector dimension mismatch'
        word2vec_token_to_idx = {w: i for i, w in enumerate(word2vec['words'])}
        for token, idx in token_to_idx.items():
            word2vec_idx = word2vec_token_to_idx.get(token, None)
            if idx >= old_N and word2vec_idx is not None:
                vec = word2vec['vecs'][word2vec_idx]
                new_weight[idx].copy_(vec)
                num_found += 1
    embed.num_embeddings = new_N
    embed.weight.data = new_weight
    return embed


class StackedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StackedAttention, self).__init__()
        self.Wv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)
        self.hidden_dim = hidden_dim
        self.attention_maps = None

    def forward(self, v, u):
        """
        Input:
        - v: N x D x H x W
        - u: N x D

        Returns:
        - next_u: N x D
        """
        N, K = v.size(0), self.hidden_dim
        D, H, W = v.size(1), v.size(2), v.size(3)
        v_proj = self.Wv(v)  # N x K x H x W
        u_proj = self.Wu(u)  # N x K
        u_proj_expand = u_proj.view(N, K, 1, 1).expand(N, K, H, W)
        h = F.tanh(v_proj + u_proj_expand)
        p = F.softmax(self.Wp(h).view(N, H * W)).view(N, 1, H, W)
        self.attention_maps = p.data.clone()

        v_tilde = (p.expand_as(v) * v).sum(2).sum(2).view(N, D)
        next_u = u + v_tilde
        return next_u


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True):
        if out_dim is None:
            out_dim = in_dim
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.bn2 = nn.BatchNorm2d(out_dim)
        self.with_residual = with_residual
        if in_dim == out_dim or not with_residual:
            self.proj = None
        else:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.with_batchnorm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(F.relu(self.conv1(x)))
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            out = F.relu(res + out)
        else:
            out = F.relu(out)
        return out


def build_cnn(feat_dim=(1024, 14, 14),
              res_block_dim=128,
              num_res_blocks=0,
              proj_dim=512,
              pooling='maxpool2'):
    C, H, W = feat_dim
    layers = []
    if num_res_blocks > 0:
        layers.append(nn.Conv2d(C, res_block_dim, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        C = res_block_dim
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(C))
    if proj_dim > 0:
        layers.append(nn.Conv2d(C, proj_dim, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        C = proj_dim
    if pooling == 'maxpool2':
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        H, W = H // 2, W // 2
    return nn.Sequential(*layers), (C, H, W)


def build_mlp(input_dim, hidden_dims, output_dim,
              use_batchnorm=False, dropout=0):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU(inplace=True))
        D = dim
    layers.append(nn.Linear(D, output_dim))
    return nn.Sequential(*layers)


def logical_and(x, y):
    return x * y


def logical_or(x, y):
    return (x + y).clamp_(0, 1)


def logical_not(x):
    return x == 0


class LstmEncoder(nn.Module):
    def __init__(self, token_to_idx, wordvec_dim=300,
                 rnn_dim=256, rnn_num_layers=2, rnn_dropout=0):
        super(LstmEncoder, self).__init__()
        self.token_to_idx = token_to_idx
        self.NULL = token_to_idx['<NULL>']
        self.START = token_to_idx['<START>']
        self.END = token_to_idx['<END>']

        self.embed = nn.Embedding(len(token_to_idx), wordvec_dim)
        self.rnn = nn.LSTM(wordvec_dim, rnn_dim, rnn_num_layers,
                           dropout=rnn_dropout, batch_first=True)

    def expand_vocab(self, token_to_idx, word2vec=None, std=0.01):
        expand_embedding_vocab(self.embed, token_to_idx,
                               word2vec=word2vec, std=std)

    def forward(self, x):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence
        x_cpu = x.data.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data).long()
        idx = Variable(idx, requires_grad=False)

        hs, _ = self.rnn(self.embed(x))
        idx = idx.view(N, 1, 1).expand(N, 1, hs.size(2))
        H = hs.size(2)
        return hs.gather(1, idx).view(N, H)


class CnnLstmSaModel(nn.Module):
    def __init__(self, vocab,
                 rnn_wordvec_dim=300, rnn_dim=256, rnn_num_layers=2, rnn_dropout=0,
                 cnn_feat_dim=(1024, 14, 14),
                 stacked_attn_dim=512, num_stacked_attn=2,
                 fc_use_batchnorm=False, fc_dropout=0, fc_dims=(1024,)):
        super(CnnLstmSaModel, self).__init__()
        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': rnn_wordvec_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = LstmEncoder(**rnn_kwargs)

        C, H, W = cnn_feat_dim
        self.image_proj = nn.Conv2d(C, rnn_dim, kernel_size=1, padding=0)
        self.stacked_attns = []
        for i in range(num_stacked_attn):
            sa = StackedAttention(rnn_dim, stacked_attn_dim)
            self.stacked_attns.append(sa)
            self.add_module('stacked-attn-%d' % i, sa)

        classifier_args = {
            'input_dim': rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answer_token_to_idx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.classifier = build_mlp(**classifier_args)

    def forward(self, questions, feats):
        u = self.rnn(questions)
        v = self.image_proj(feats)

        for sa in self.stacked_attns:
            u = sa(v, u)

        scores = self.classifier(u)
        return scores


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder_vocab_size=100,
                 decoder_vocab_size=100,
                 wordvec_dim=300,
                 hidden_dim=256,
                 rnn_num_layers=2,
                 rnn_dropout=0,
                 null_token=0,
                 start_token=1,
                 end_token=2,
                 encoder_embed=None
                 ):
        super(Seq2Seq, self).__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_rnn = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        self.multinomial_outputs = None

    def expand_encoder_vocab(self, token_to_idx, word2vec=None, std=0.01):
        expand_embedding_vocab(self.encoder_embed, token_to_idx,
                               word2vec=word2vec, std=std)

    def get_dims(self, x=None, y=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.decoder_embed.num_embeddings
        D = self.encoder_embed.embedding_dim
        H = self.encoder_rnn.hidden_size
        L = self.encoder_rnn.num_layers

        N = x.size(0) if x is not None else None
        N = y.size(0) if N is None and y is not None else N
        T_in = x.size(1) if x is not None else None
        T_out = y.size(1) if y is not None else None
        return V_in, V_out, D, H, L, N, T_in, T_out

    def before_rnn(self, x, replace=0):
        # TODO: Use PackedSequence instead of manually plucking out the last
        # non-NULL entry of each sequence; it is cleaner and more efficient.
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence. Is there a clean
        # way to do this?
        x_cpu = x.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu.data[i, t] != self.NULL and x_cpu.data[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data)
        x[x.data == self.NULL] = replace
        return x, Variable(idx)

    def encoder(self, x):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        embed = self.encoder_embed(x)
        h0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
        c0 = Variable(torch.zeros(L, N, H).type_as(embed.data))

        out, _ = self.encoder_rnn(embed, (h0, c0))

        # Pull out the hidden state for the last non-null value in each input
        idx = idx.view(N, 1, 1).expand(N, 1, H)
        return out.gather(1, idx).view(N, H)

    def decoder(self, encoded, y, h0=None, c0=None):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)

        if T_out > 1:
            y, _ = self.before_rnn(y)
        y_embed = self.decoder_embed(y)
        encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
        rnn_input = torch.cat([encoded_repeat, y_embed], 2)
        if h0 is None:
            h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        if c0 is None:
            c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        rnn_output, (ht, ct) = self.decoder_rnn(rnn_input, (h0, c0))

        rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
        output_logprobs = self.decoder_linear(rnn_output_2d).view(N, T_out, V_out)

        return output_logprobs, ht, ct

    def compute_loss(self, output_logprobs, y):
        """
        Compute loss. We assume that the first element of the output sequence y is
        a start token, and that each element of y is left-aligned and right-padded
        with self.NULL out to T_out. We want the output_logprobs to predict the
        sequence y, shifted by one timestep so that y[0] is fed to the network and
        then y[1] is predicted. We also don't want to compute loss for padded
        timesteps.

        Inputs:
        - output_logprobs: Variable of shape (N, T_out, V_out)
        - y: LongTensor Variable of shape (N, T_out)
        """
        self.multinomial_outputs = None
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
        mask = y.data != self.NULL
        y_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        y_mask[:, 1:] = mask[:, 1:]
        y_masked = y[y_mask]
        out_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        out_mask[:, :-1] = mask[:, 1:]
        out_mask = out_mask.view(N, T_out, 1).expand(N, T_out, V_out)
        out_masked = output_logprobs[out_mask].view(-1, V_out)
        loss = F.cross_entropy(out_masked, y_masked)
        return loss

    def forward(self, x, y):
        encoded = self.encoder(x)
        output_logprobs, _, _ = self.decoder(encoded, y)
        loss = self.compute_loss(output_logprobs, y)
        return loss

    def sample(self, x, max_length=50):
        # TODO: Handle sampling for minibatch inputs
        # TODO: Beam search?
        self.multinomial_outputs = None
        assert x.size(0) == 1, "Sampling minibatches not implemented"
        encoded = self.encoder(x)
        y = [self.START]
        h0, c0 = None, None
        while True:
            cur_y = Variable(torch.LongTensor([y[-1]]).type_as(x.data).view(1, 1))
            logprobs, h0, c0 = self.decoder(encoded, cur_y, h0=h0, c0=c0)
            _, next_y = logprobs.data.max(2)
            y.append(next_y[0, 0, 0])
            if len(y) >= max_length or y[-1] == self.END:
                break
        return y

    def reinforce_sample(self, x, max_length=30, temperature=1.0, argmax=False):
        N, T = x.size(0), max_length
        encoded = self.encoder(x)
        y = torch.LongTensor(N, T).fill_(self.NULL)
        done = torch.ByteTensor(N).fill_(0)
        cur_input = Variable(x.data.new(N, 1).fill_(self.START))
        h, c = None, None
        self.multinomial_outputs = []
        self.multinomial_probs = []
        for t in range(T):
            # logprobs is N x 1 x V
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            logprobs = logprobs / temperature
            probs = F.softmax(logprobs.view(N, -1))  # Now N x V
            if argmax:
                _, cur_output = probs.max(1)
            else:
                cur_output = probs.multinomial()  # Now N x 1
            self.multinomial_outputs.append(cur_output)
            self.multinomial_probs.append(probs)
            cur_output_data = cur_output.data.cpu()
            not_done = logical_not(done)
            y[:, t][not_done] = cur_output_data[not_done]
            done = logical_or(done, cur_output_data.cpu() == self.END)
            cur_input = cur_output.view(N, 1).expand(N, 1)
            if done.sum() == N:
                break
        return Variable(y.type_as(x.data))

    def reinforce_backward(self, reward, output_mask=None):
        """
        If output_mask is not None, then it should be a FloatTensor of shape (N, T)
        giving a multiplier to the output.
        """
        assert self.multinomial_outputs is not None, 'Must call reinforce_sample first'
        grad_output = []

        def gen_hook(mask):
            def hook(grad):
                return grad * mask.contiguous().view(-1, 1).expand_as(grad)

            return hook

        if output_mask is not None:
            for t, probs in enumerate(self.multinomial_probs):
                mask = Variable(output_mask[:, t])
                probs.register_hook(gen_hook(mask))

        for sampled_output in self.multinomial_outputs:
            sampled_output.reinforce(reward)
            grad_output.append(None)
        torch.autograd.backward(self.multinomial_outputs, grad_output, retain_variables=True)


class ConcatBlock(nn.Module):
    def __init__(self, dim, with_residual=True, with_batchnorm=True):
        super(ConcatBlock, self).__init__()
        self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)
        self.res_block = ResidualBlock(dim, with_residual=with_residual,
                                       with_batchnorm=with_batchnorm)

    def forward(self, x, y):
        out = torch.cat([x, y], 1)  # Concatentate along depth
        out = F.relu(self.proj(out))
        out = self.res_block(out)
        return out


class GlobalAveragePool(nn.Module):
    def forward(self, x):
        N, C = x.size(0), x.size(1)
        return x.view(N, C, -1).mean(2).squeeze(2)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def build_stem(feature_dim, module_dim, num_layers=2, with_batchnorm=True):
    layers = []
    prev_dim = feature_dim
    for i in range(num_layers):
        layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=3, padding=1))
        if with_batchnorm:
            layers.append(nn.BatchNorm2d(module_dim))
        layers.append(nn.ReLU(inplace=True))
        prev_dim = module_dim
    return nn.Sequential(*layers)


def build_classifier(module_C, module_H, module_W, num_answers,
                     fc_dims=[], proj_dim=None, downsample='maxpool2',
                     with_batchnorm=True, dropout=0):
    layers = []
    prev_dim = module_C * module_H * module_W
    if proj_dim is not None and proj_dim > 0:
        layers.append(nn.Conv2d(module_C, proj_dim, kernel_size=1))
        if with_batchnorm:
            layers.append(nn.BatchNorm2d(proj_dim))
        layers.append(nn.ReLU(inplace=True))
        prev_dim = proj_dim * module_H * module_W
    if downsample == 'maxpool2':
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        prev_dim //= 4
    elif downsample == 'maxpool4':
        layers.append(nn.MaxPool2d(kernel_size=4, stride=4))
        prev_dim //= 16
    layers.append(Flatten())
    for next_dim in fc_dims:
        layers.append(nn.Linear(prev_dim, next_dim))
        if with_batchnorm:
            layers.append(nn.BatchNorm1d(next_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        prev_dim = next_dim
    layers.append(nn.Linear(prev_dim, num_answers))
    return nn.Sequential(*layers)


def str_to_function(s):
    if '[' not in s:
        return {
            'function': s,
            'value_inputs': [],
        }
    name, value_str = s.replace(']', '').split('[')
    return {
        'function': name,
        'value_inputs': value_str.split(','),
    }


def get_num_inputs(f):
    # This is a litle hacky; it would be better to look up from metadata.json
    if type(f) is str:
        f = str_to_function(f)
    name = f['function']
    if name == 'scene':
        return 0
    if 'equal' in name or name in ['union', 'intersect', 'less_than', 'greater_than']:
        return 2
    return 1


def function_to_str(f):
    value_str = ''
    if f['value_inputs']:
        value_str = '[%s]' % ','.join(f['value_inputs'])
    return '%s%s' % (f['function'], value_str)


class ModuleNet(nn.Module):
    def __init__(self, vocab, feature_dim=(1024, 14, 14),
                 stem_num_layers=2,
                 stem_batchnorm=False,
                 module_dim=128,
                 module_residual=True,
                 module_batchnorm=False,
                 classifier_proj_dim=512,
                 classifier_downsample='maxpool2',
                 classifier_fc_layers=(1024,),
                 classifier_batchnorm=False,
                 classifier_dropout=0,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.stem = build_stem(feature_dim[0], module_dim,
                               num_layers=stem_num_layers,
                               with_batchnorm=stem_batchnorm)
        if verbose:
            print('Here is my stem:')
            print(self.stem)

        num_answers = len(vocab['answer_idx_to_token'])
        module_H, module_W = feature_dim[1], feature_dim[2]
        self.classifier = build_classifier(module_dim, module_H, module_W, num_answers,
                                           classifier_fc_layers,
                                           classifier_proj_dim,
                                           classifier_downsample,
                                           with_batchnorm=classifier_batchnorm,
                                           dropout=classifier_dropout)
        if verbose:
            print('Here is my classifier:')
            print(self.classifier)
        self.stem_times = []
        self.module_times = []
        self.classifier_times = []
        self.timing = False

        self.function_modules = {}
        self.function_modules_num_inputs = {}
        self.vocab = vocab
        for fn_str in vocab['program_token_to_idx']:
            num_inputs = get_num_inputs(fn_str)
            self.function_modules_num_inputs[fn_str] = num_inputs
            if fn_str == 'scene' or num_inputs == 1:
                mod = ResidualBlock(module_dim,
                                    with_residual=module_residual,
                                    with_batchnorm=module_batchnorm)
            elif num_inputs == 2:
                mod = ConcatBlock(module_dim,
                                  with_residual=module_residual,
                                  with_batchnorm=module_batchnorm)
            self.add_module(fn_str, mod)
            self.function_modules[fn_str] = mod

        self.save_module_outputs = False

    def expand_answer_vocab(self, answer_to_idx, std=0.01, init_b=-50):
        # TODO: This is really gross, dipping into private internals of Sequential
        final_linear_key = str(len(self.classifier._modules) - 1)
        final_linear = self.classifier._modules[final_linear_key]

        old_weight = final_linear.weight.data
        old_bias = final_linear.bias.data
        old_N, D = old_weight.size()
        new_N = 1 + max(answer_to_idx.values())
        new_weight = old_weight.new(new_N, D).normal_().mul_(std)
        new_bias = old_bias.new(new_N).fill_(init_b)
        new_weight[:old_N].copy_(old_weight)
        new_bias[:old_N].copy_(old_bias)

        final_linear.weight.data = new_weight
        final_linear.bias.data = new_bias

    def _forward_modules_json(self, feats, program):
        def gen_hook(i, j):
            def hook(grad):
                self.all_module_grad_outputs[i][j] = grad.data.cpu().clone()

            return hook

        self.all_module_outputs = []
        self.all_module_grad_outputs = []
        # We can't easily handle minibatching of modules, so just do a loop
        N = feats.size(0)
        final_module_outputs = []
        for i in range(N):
            if self.save_module_outputs:
                self.all_module_outputs.append([])
                self.all_module_grad_outputs.append([None] * len(program[i]))
            module_outputs = []
            for j, f in enumerate(program[i]):
                f_str = function_to_str(f)
                module = self.function_modules[f_str]
                if f_str == 'scene':
                    module_inputs = [feats[i:i + 1]]
                else:
                    module_inputs = [module_outputs[j] for j in f['inputs']]
                module_outputs.append(module(*module_inputs))
                if self.save_module_outputs:
                    self.all_module_outputs[-1].append(module_outputs[-1].data.cpu().clone())
                    module_outputs[-1].register_hook(gen_hook(i, j))
            final_module_outputs.append(module_outputs[-1])
        final_module_outputs = torch.cat(final_module_outputs, 0)
        return final_module_outputs

    def _forward_modules_ints_helper(self, feats, program, i, j):
        used_fn_j = True
        if j < program.size(1):
            fn_idx = int(program.data[i, j].cpu().numpy())
            fn_str = self.vocab['program_idx_to_token'][fn_idx]
        else:
            used_fn_j = False
            fn_str = 'scene'
        if fn_str == '<NULL>':
            used_fn_j = False
            fn_str = 'scene'
        elif fn_str == '<START>':
            used_fn_j = False
            return self._forward_modules_ints_helper(feats, program, i, j + 1)
        if used_fn_j:
            self.used_fns[i, j] = 1
        j += 1
        module = self.function_modules[fn_str]
        if fn_str == 'scene':
            module_inputs = [feats[i:i + 1]]
        else:
            num_inputs = self.function_modules_num_inputs[fn_str]
            module_inputs = []
            while len(module_inputs) < num_inputs:
                cur_input, j = self._forward_modules_ints_helper(feats, program, i, j)
                module_inputs.append(cur_input)
        module_output = module(*module_inputs)
        return module_output, j

    def _forward_modules_ints(self, feats, program):
        """
        feats: FloatTensor of shape (N, C, H, W) giving features for each image
        program: LongTensor of shape (N, L) giving a prefix-encoded program for
          each image.
        """
        N = feats.size(0)
        final_module_outputs = []
        self.used_fns = torch.Tensor(program.size()).fill_(0)
        for i in range(N):
            cur_output, _ = self._forward_modules_ints_helper(feats, program, i, 0)
            final_module_outputs.append(cur_output)
        self.used_fns = self.used_fns.type_as(program.data).float()
        final_module_outputs = torch.cat(final_module_outputs, 0)
        return final_module_outputs

    def forward(self, x, program):
        N = x.size(0)
        assert N == len(program)

        feats = self.stem(x)

        if type(program) is list or type(program) is tuple:
            final_module_outputs = self._forward_modules_json(feats, program)
        elif type(program) is torch.Tensor and program.dim() == 2:
            final_module_outputs = self._forward_modules_ints(feats, program)
        else:
            raise ValueError('Unrecognized program format')

        # After running modules for each input, concatenat the outputs from the
        # final module and run the classifier.
        out = self.classifier(final_module_outputs)
        return out


def load_cpu(path):
    """
    Loads a torch checkpoint, remapping all Tensors to CPU
    """
    return torch.load(path, map_location=lambda storage, loc: storage)


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_baseline(path):
    model_cls_dict = {
        'CNN+LSTM+SA': CnnLstmSaModel,
    }
    checkpoint = load_cpu(path)
    baseline_type = checkpoint['baseline_type']
    kwargs = checkpoint['baseline_kwargs']
    state = checkpoint['baseline_state']

    model = model_cls_dict[baseline_type](**kwargs)
    model.load_state_dict(state)
    return model, kwargs


def load_program_generator(path):
    checkpoint = load_cpu(path)
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    model = Seq2Seq(**kwargs)
    model.load_state_dict(state)
    return model, kwargs


def load_execution_engine(path, verbose=True):
    checkpoint = load_cpu(path)
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    kwargs['verbose'] = verbose
    model = ModuleNet(**kwargs)
    model.load_state_dict(state)
    return model, kwargs


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab


def build_cnn2(args, dtype):
    if not hasattr(torchvision.models, args['cnn_model']):
        raise ValueError('Invalid model "%s"' % args['cnn_model'])
    if not 'resnet' in args['cnn_model']:
        raise ValueError('Feature extraction only supports ResNets')
    whole_cnn = getattr(torchvision.models, args['cnn_model'])(pretrained=True)
    layers = [
        whole_cnn.conv1,
        whole_cnn.bn1,
        whole_cnn.relu,
        whole_cnn.maxpool,
    ]
    for i in range(args['cnn_model_stage']):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(whole_cnn, name))
    cnn = torch.nn.Sequential(*layers)
    cnn.type(dtype)
    cnn.eval()
    return cnn


def run_single_example(args, model):
    dtype = torch.FloatTensor
    if args['use_gpu'] == 1:
        dtype = torch.cuda.FloatTensor
        
    # Build the CNN to use for feature extraction
    if args['external'] == False or args['external'] == 'False':
        cnn = build_cnn2(args, dtype)
        # Load and preprocess the image
        img_size = (args['image_height'], args['image_width'])
        img = imread(args['image'])
        img = rgba2rgb(img)
        img = imresize(img, img_size)
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)[None]
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
        img = (img - mean) / std
        # Use CNN to extract features for the image
        img_var = torch.FloatTensor(img).type(dtype)
        with torch.no_grad():
            feats_var = cnn(img_var)
    elif args['external'] == True or args['external'] == 'True':
        feats_var = args['feats']
        feats_var = torch.FloatTensor(feats_var).type(dtype)
    else:
        feats_var = None

    # Tokenize the question(s)
    vocab = load_vocab(args['vocab_json'])
    if isinstance(args['questions'], str):
        questions = [args['questions']]
    elif isinstance(args['questions'], list):
        questions = args['questions']
    else:
        questions = None

    if type(model) is tuple:
        program_generator, execution_engine = model
        program_generator.type(dtype)
        execution_engine.type(dtype)
    else:
        program_generator = None
        execution_engine = None
        model.type(dtype)

    predicted_answers = []
    if type(model) is tuple:
        for question in questions:
            question_tokens = tokenize(question,
                                       punct_to_keep=[';', ','],
                                       punct_to_remove=['?', '.'])
            question_encoded = encode(question_tokens,
                                      vocab['question_token_to_idx'],
                                      allow_unk=True)
            question_encoded = torch.LongTensor(question_encoded).view(1, -1)
            question_encoded = question_encoded.type(dtype).long()
            question_var = Variable(question_encoded)
            predicted_program = program_generator.reinforce_sample(
                question_var,
                temperature=args['temperature'],
                argmax=(args['sample_argmax'] == 1))
            with torch.no_grad():
                scores = execution_engine(feats_var, predicted_program)
            _, predicted_answer_idx = scores.data.cpu()[0].max(dim=0)
            predicted_answer = vocab['answer_idx_to_token'][predicted_answer_idx.item()]
            predicted_answers.append(predicted_answer)
    else:
        for question in questions:
            question_tokens = tokenize(question,
                                       punct_to_keep=[';', ','],
                                       punct_to_remove=['?', '.'])
            question_encoded = encode(question_tokens,
                                      vocab['question_token_to_idx'],
                                      allow_unk=True)
            question_encoded = torch.LongTensor(question_encoded).view(1, -1)
            question_encoded = question_encoded.type(dtype).long()
            question_var = Variable(question_encoded)
            with torch.no_grad():
                scores = model(question_var, feats_var)
            _, predicted_answer_idx = scores.data.cpu()[0].max(dim=0)
            predicted_answer = vocab['answer_idx_to_token'][predicted_answer_idx.item()]
            predicted_answers.append(predicted_answer)
    return predicted_answers


def run_batch(args, model):
    dtype = torch.FloatTensor
    if args['use_gpu'] == 1:
        dtype = torch.cuda.FloatTensor

    vocab = load_vocab(args['vocab_json'])
    loader_kwargs = {
        'question_h5': args['input_questions_h5'],
        'feature_h5': args['input_features_h5'],
        'vocab': vocab,
        'batch_size': 32,
    }
    if args['num_samples'] is not None and args['num_samples'] > 0:
        loader_kwargs['max_samples'] = args['num_samples']
    if args['family_split_file'] is not None:
        with open(args['family_split_file'], 'r') as f:
            loader_kwargs['question_families'] = json.load(f)

    loader = ClevrDataLoader(**loader_kwargs)

    if type(model) is tuple:
        program_generator, execution_engine = model
        run_our_model_batch(program_generator, execution_engine, loader, dtype)
    else:
        run_baseline_batch(model, loader, dtype)


def validate_performance(model, split):
    if model == 'sa':
        inference = inference_with_cnn_sa
    elif model == 'iep':
        inference = inference_with_iep
    else:
        inference = None

    with open(f'../data/CLEVR_v1.0/questions/CLEVR_{split}_questions.json', 'r') as fin:
        js = json.loads(fin.readlines()[0])['questions']

    prefix = f'../data/CLEVR_v1.0/images/{split}/'
    loader = [(prefix + f['image_filename'], f['question'], f['answer'], f['question_family_index']) for f in js]

    question_family_score = {}
    question_family_count = {}
    overall_score = 0
    overall_count = 0
    for i, q, a, fn in loader:
        overall_count += 1
        if fn in question_family_count:
            question_family_count[fn] += 1
        else:
            question_family_count.update({fn: 1})
        answer, _ = inference(q, i)
        print()
        print(answer)
        print(a)
        print()
        if str(answer).lower() == str(a).lower():
            overall_score += 1
            if fn in question_family_score:
                question_family_score[fn] += 1
            else:
                question_family_score.update({fn: 1})
        # if overall_count % 100 == 0:
        print(f'Running Accuracy: {round(1.0 * overall_score / overall_count, 2)}')


def run_baseline_batch(model, loader, dtype):
    model.type(dtype)
    model.eval()

    all_scores, all_probs = [], []
    num_correct, num_samples = 0, 0
    print(f"Testing for {len(loader) * 32} samples")
    print()
    for batch in loader:
        questions, images, feats, answers, _, _ = batch

        questions_var = Variable(questions.type(dtype).long())
        feats_var = Variable(feats.type(dtype))
        scores = model(questions_var, feats_var)

        _, preds = scores.data.cpu().max(1)
        all_scores.append(scores.data.cpu().clone())

        num_correct += (preds == answers).sum()
        num_samples += preds.size(0)
        if num_samples % 1000 == 0:
            print(f'Ran {num_samples} samples at {float(num_correct) / num_samples} accuracy')

    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))


def run_our_model_batch(program_generator, execution_engine, loader, dtype):
    program_generator.type(dtype)
    program_generator.eval()
    execution_engine.type(dtype)
    execution_engine.eval()

    all_scores, all_programs = [], []
    all_probs = []
    num_correct, num_samples = 0, 0
    print(f"Testing for {len(loader) * 32} samples")
    print()
    for batch in loader:
        questions, images, feats, answers, _, _ = batch

        questions_var = Variable(questions.type(dtype).long())
        feats_var = Variable(feats.type(dtype))

        programs_pred = program_generator.reinforce_sample(
            questions_var,
            temperature=1,
            argmax=True)

        scores = execution_engine(feats_var, programs_pred)
        probs = F.softmax(scores)

        _, preds = scores.data.cpu().max(1)
        all_programs.append(programs_pred.data.cpu().clone())
        all_scores.append(scores.data.cpu().clone())
        all_probs.append(probs.data.cpu().clone())

        num_correct += (preds == answers).sum()
        num_samples += preds.size(0)
        if num_samples % 16000 == 0:
            print(f'Ran {num_samples} samples at {float(num_correct) / num_samples} accuracy')

    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
