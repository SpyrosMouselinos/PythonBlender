import inspect
import json
import sys
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize as imresize
from torch.autograd import Variable
from torch.nn.init import kaiming_normal, kaiming_uniform_

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

def invert_dict(d):
    return {value: key for (key, value) in d.items()}


with open('../models/CLEVR/vocab.json', 'r') as fin:
    data = json.loads(fin.read())

question_token_to_idx = data['question_token_to_idx']
program_token_to_idx = data['program_token_to_idx']
answer_token_to_idx = data['answer_token_to_idx']

idx_to_question_token = invert_dict(question_token_to_idx)
idx_to_program_token = invert_dict(program_token_to_idx)
idx_to_answer_token = invert_dict(answer_token_to_idx)


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

        return (question, image, feats, answer, program_seq, program_json, image_idx)

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
    image_index_batch = transposed[6]
    return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch, program_struct_batch,
            image_index_batch]


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
                       program_generator='../data/models/CLEVR/program_generator_700k.pt',
                       execution_engine='../data/models/CLEVR/execution_engine_700k.pt',
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
    program_generator, _ = load_program_generator(program_generator, 'Seq2Seq')
    program_generator.eval()
    execution_engine, _ = load_execution_engine(execution_engine, verbose=False, model_type='EE')
    execution_engine.eval()
    model = (program_generator, execution_engine)
    if batch_size == 1:
        answer = run_single_example(collect_locals, model)
        return answer
    else:
        run_batch(collect_locals, model)
        return


def inference_with_film(questions='Is the number of cubes less than the number of small metal things?',
                        image='../img/CLEVR_val_000013.png',
                        feats=None,
                        input_questions_h5='../data/val_questions.h5',
                        input_features_h5='../data/val_features.h5',
                        program_generator='../data/models/CLEVR/film.pt',
                        execution_engine='../data/models/CLEVR/film.pt',
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
    program_generator, _ = load_program_generator(program_generator, 'FiLM')
    program_generator.eval()
    execution_engine, _ = load_execution_engine(execution_engine, verbose=False, model_type='FiLM')
    execution_engine.eval()
    model = (program_generator, execution_engine)
    if batch_size == 1:
        answer = run_single_example(collect_locals, model, tuple_mode='FiLM')
        return answer
    else:
        run_batch(collect_locals, model, tuple_mode='FiLM')
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


def decode(seq_idx, idx_to_token=None, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def decode_question(question):
    try:
        question = question.cpu().numpy()
    except:
        pass
    return decode(seq_idx=question, idx_to_token=idx_to_question_token)


def decode_answer(answer):
    try:
        answer = answer.cpu().numpy()
    except:
        pass
    return decode(seq_idx=[answer], idx_to_token=idx_to_answer_token)


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


def build_stem_film(feature_dim, module_dim, num_layers=2, with_batchnorm=True,
                    kernel_size=3, stride=1, padding=None):
    layers = []
    prev_dim = feature_dim
    if padding is None:  # Calculate default padding when None provided
        if kernel_size % 2 == 0:
            raise (NotImplementedError)
        padding = kernel_size // 2
    for i in range(num_layers):
        layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=kernel_size, stride=stride,
                                padding=padding))
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


def build_classifier_film(module_C, module_H, module_W, num_answers,
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
    if 'maxpool' in downsample or 'avgpool' in downsample:
        pool = nn.MaxPool2d if 'maxpool' in downsample else nn.AvgPool2d
        if 'full' in downsample:
            if module_H != module_W:
                assert (NotImplementedError)
            pool_size = module_H
        else:
            pool_size = int(downsample[-1])
        # Note: Potentially sub-optimal padding for non-perfectly aligned pooling
        padding = 0 if ((module_H % pool_size == 0) and (module_W % pool_size == 0)) else 1
        layers.append(pool(kernel_size=pool_size, stride=pool_size, padding=padding))
        prev_dim = proj_dim * math.ceil(module_H / pool_size) * math.ceil(module_W / pool_size)
    if downsample == 'aggressive':
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.AvgPool2d(kernel_size=module_H // 2, stride=module_W // 2))
        prev_dim = proj_dim
        fc_dims = []  # No FC layers here
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


arg_value_updates = {
    'condition_method': {
        'block-input-fac': 'block-input-film',
        'block-output-fac': 'block-output-film',
        'cbn': 'bn-film',
        'conv-fac': 'conv-film',
        'relu-fac': 'relu-film',
    },
    'module_input_proj': {
        True: 1,
    },
}


def get_updated_args(kwargs, object_class):
    """
    Returns kwargs with renamed args or arg valuesand deleted, deprecated, unused args.
    Useful for loading older, trained models.
    If using this function is neccessary, use immediately before initializing object.
    """
    # Update arg values
    for arg in arg_value_updates:
        if arg in kwargs and kwargs[arg] in arg_value_updates[arg]:
            kwargs[arg] = arg_value_updates[arg][kwargs[arg]]

    # Delete deprecated, unused args
    valid_args = inspect.getargspec(object_class.__init__)[0]
    new_kwargs = {valid_arg: kwargs[valid_arg] for valid_arg in valid_args if valid_arg in kwargs}
    return new_kwargs


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class FiLMedNet(nn.Module):
    def __init__(self, vocab, feature_dim=(1024, 14, 14),
                 stem_num_layers=2,
                 stem_batchnorm=False,
                 stem_kernel_size=3,
                 stem_stride=1,
                 stem_padding=None,
                 num_modules=4,
                 module_num_layers=1,
                 module_dim=128,
                 module_residual=True,
                 module_batchnorm=False,
                 module_batchnorm_affine=False,
                 module_dropout=0,
                 module_input_proj=1,
                 module_kernel_size=3,
                 classifier_proj_dim=512,
                 classifier_downsample='maxpool2',
                 classifier_fc_layers=(1024,),
                 classifier_batchnorm=False,
                 classifier_dropout=0,
                 condition_method='bn-film',
                 condition_pattern=[],
                 use_gamma=True,
                 use_beta=True,
                 use_coords=1,
                 debug_every=float('inf'),
                 print_verbose_every=float('inf'),
                 verbose=True,
                 ):
        super(FiLMedNet, self).__init__()

        num_answers = len(vocab['answer_idx_to_token'])

        self.stem_times = []
        self.module_times = []
        self.classifier_times = []
        self.timing = False

        self.num_modules = num_modules
        self.module_num_layers = module_num_layers
        self.module_batchnorm = module_batchnorm
        self.module_dim = module_dim
        self.condition_method = condition_method
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.use_coords_freq = use_coords
        self.debug_every = debug_every
        self.print_verbose_every = print_verbose_every

        # Initialize helper variables
        self.stem_use_coords = (stem_stride == 1) and (self.use_coords_freq > 0)
        self.condition_pattern = condition_pattern
        if len(condition_pattern) == 0:
            self.condition_pattern = []
            for i in range(self.module_num_layers * self.num_modules):
                self.condition_pattern.append(self.condition_method != 'concat')
        else:
            self.condition_pattern = [i > 0 for i in self.condition_pattern]
        self.extra_channel_freq = self.use_coords_freq
        self.block = FiLMedResBlock
        self.num_cond_maps = 2 * self.module_dim if self.condition_method == 'concat' else 0
        self.fwd_count = 0
        self.num_extra_channels = 2 if self.use_coords_freq > 0 else 0
        if self.debug_every <= -1:
            self.print_verbose_every = 1
        module_H = feature_dim[1] // (stem_stride ** stem_num_layers)  # Rough calc: work for main cases
        module_W = feature_dim[2] // (stem_stride ** stem_num_layers)  # Rough calc: work for main cases
        self.coords = coord_map((module_H, module_W))
        self.default_weight = Variable(torch.ones(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)
        self.default_bias = Variable(torch.zeros(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)

        # Initialize stem
        stem_feature_dim = feature_dim[0] + self.stem_use_coords * self.num_extra_channels
        self.stem = build_stem_film(stem_feature_dim, module_dim,
                                    num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
                                    kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding)

        # Initialize FiLMed network body
        self.function_modules = {}
        self.vocab = vocab
        for fn_num in range(self.num_modules):
            with_cond = self.condition_pattern[self.module_num_layers * fn_num:
                                               self.module_num_layers * (fn_num + 1)]
            mod = self.block(module_dim, with_residual=module_residual, with_batchnorm=module_batchnorm,
                             with_cond=with_cond,
                             dropout=module_dropout,
                             num_extra_channels=self.num_extra_channels,
                             extra_channel_freq=self.extra_channel_freq,
                             with_input_proj=module_input_proj,
                             num_cond_maps=self.num_cond_maps,
                             kernel_size=module_kernel_size,
                             batchnorm_affine=module_batchnorm_affine,
                             num_layers=self.module_num_layers,
                             condition_method=condition_method,
                             debug_every=self.debug_every)
            self.add_module(str(fn_num), mod)
            self.function_modules[fn_num] = mod

        # Initialize output classifier
        self.classifier = build_classifier_film(module_dim + self.num_extra_channels, module_H, module_W,
                                           num_answers, classifier_fc_layers, classifier_proj_dim,
                                           classifier_downsample, with_batchnorm=classifier_batchnorm,
                                           dropout=classifier_dropout)

        init_modules(self.modules())

    def forward(self, x, film, save_activations=False):
        # Initialize forward pass and externally viewable activations
        self.fwd_count += 1
        if save_activations:
            self.feats = None
            self.module_outputs = []
            self.cf_input = None

        # Prepare FiLM layers
        gammas = None
        betas = None
        if self.condition_method == 'concat':
            # Use parameters usually used to condition via FiLM instead to condition via concatenation
            cond_params = film[:, :, :2 * self.module_dim]
            cond_maps = cond_params.unsqueeze(3).unsqueeze(4).expand(cond_params.size() + x.size()[-2:])
        else:
            gammas, betas = torch.split(film[:, :, :2 * self.module_dim], self.module_dim, dim=-1)
            if not self.use_gamma:
                gammas = self.default_weight.expand_as(gammas)
            if not self.use_beta:
                betas = self.default_bias.expand_as(betas)

        # Propagate up image features CNN
        batch_coords = None
        if self.use_coords_freq > 0:
            batch_coords = self.coords.unsqueeze(0).expand(torch.Size((x.size(0), *self.coords.size())))
        if self.stem_use_coords:
            x = torch.cat([x, batch_coords], 1)
        feats = self.stem(x)
        if save_activations:
            self.feats = feats
        N, _, H, W = feats.size()

        # Propagate up the network from low-to-high numbered blocks
        module_inputs = Variable(torch.zeros(feats.size()).unsqueeze(1).expand(
            N, self.num_modules, self.module_dim, H, W)).type(torch.cuda.FloatTensor)
        module_inputs[:, 0] = feats
        for fn_num in range(self.num_modules):
            if self.condition_method == 'concat':
                layer_output = self.function_modules[fn_num](module_inputs[:, fn_num],
                                                             extra_channels=batch_coords,
                                                             cond_maps=cond_maps[:, fn_num])
            else:
                layer_output = self.function_modules[fn_num](module_inputs[:, fn_num],
                                                             gammas[:, fn_num, :], betas[:, fn_num, :], batch_coords)

            # Store for future computation
            if save_activations:
                self.module_outputs.append(layer_output)
            if fn_num == (self.num_modules - 1):
                final_module_output = layer_output
            else:
                module_inputs_updated = module_inputs.clone()
                module_inputs_updated[:, fn_num + 1] = module_inputs_updated[:, fn_num + 1] + layer_output
                module_inputs = module_inputs_updated

        # Run the final classifier over the resultant, post-modulated features.
        if self.use_coords_freq > 0:
            final_module_output = torch.cat([final_module_output, batch_coords], 1)
        if save_activations:
            self.cf_input = final_module_output
        out = self.classifier(final_module_output)
        return out


class FiLMedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True,
                 with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
                 with_input_proj=0, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,
                 num_layers=1, condition_method='bn-film', debug_every=float('inf')):
        if out_dim is None:
            out_dim = in_dim
        super(FiLMedResBlock, self).__init__()
        self.with_residual = with_residual
        self.with_batchnorm = with_batchnorm
        self.with_cond = with_cond
        self.dropout = dropout
        self.extra_channel_freq = 0 if num_extra_channels == 0 else extra_channel_freq
        self.with_input_proj = with_input_proj  # Kernel size of input projection
        self.num_cond_maps = num_cond_maps
        self.kernel_size = kernel_size
        self.batchnorm_affine = batchnorm_affine
        self.num_layers = num_layers
        self.condition_method = condition_method
        self.debug_every = debug_every

        if self.with_input_proj % 2 == 0:
            raise (NotImplementedError)
        if self.kernel_size % 2 == 0:
            raise (NotImplementedError)
        if self.num_layers >= 2:
            raise (NotImplementedError)

        if self.condition_method == 'block-input-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_input_proj:
            self.input_proj = nn.Conv2d(in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                                        in_dim, kernel_size=self.with_input_proj, padding=self.with_input_proj // 2)

        self.conv1 = nn.Conv2d(in_dim + self.num_cond_maps +
                               (num_extra_channels if self.extra_channel_freq >= 2 else 0),
                               out_dim, kernel_size=self.kernel_size,
                               padding=self.kernel_size // 2)
        if self.condition_method == 'conv-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
        if self.condition_method == 'bn-film' and self.with_cond[0]:
            self.film = FiLM()
        if dropout > 0:
            self.drop = nn.Dropout2d(p=self.dropout)
        if ((self.condition_method == 'relu-film' or self.condition_method == 'block-output-film')
                and self.with_cond[0]):
            self.film = FiLM()

        init_modules(self.modules())

    def forward(self, x, gammas=None, betas=None, extra_channels=None, cond_maps=None):

        if self.condition_method == 'block-input-film' and self.with_cond[0]:
            x = self.film(x, gammas, betas)

        # ResBlock input projection
        if self.with_input_proj:
            if extra_channels is not None and self.extra_channel_freq >= 1:
                x = torch.cat([x, extra_channels], 1)
            x = F.relu(self.input_proj(x))
        out = x

        # ResBlock body
        if cond_maps is not None:
            out = torch.cat([out, cond_maps], 1)
        if extra_channels is not None and self.extra_channel_freq >= 2:
            out = torch.cat([out, extra_channels], 1)
        out = self.conv1(out)
        if self.condition_method == 'conv-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)
        if self.with_batchnorm:
            out = self.bn1(out)
        if self.condition_method == 'bn-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)
        if self.dropout > 0:
            out = self.drop(out)
        out = F.relu(out)
        if self.condition_method == 'relu-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)

        # ResBlock remainder
        if self.with_residual:
            out = x + out
        if self.condition_method == 'block-output-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)
        return out


def coord_map(shape, start=-1, end=1):
    """
    Gives, a 2d shape tuple, returns two mxn coordinate maps,
    Ranging min-max in the x and y directions, respectively.
    """
    m, n = shape
    x_coord_row = torch.linspace(start, end, steps=n).type(torch.cuda.FloatTensor)
    y_coord_row = torch.linspace(start, end, steps=m).type(torch.cuda.FloatTensor)
    x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
    y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
    return Variable(torch.cat([x_coords, y_coords], 0))


class FiLMGen(nn.Module):
    def __init__(self,
                 null_token=0,
                 start_token=1,
                 end_token=2,
                 encoder_embed=None,
                 encoder_vocab_size=100,
                 decoder_vocab_size=100,
                 wordvec_dim=200,
                 hidden_dim=512,
                 rnn_num_layers=1,
                 rnn_dropout=0,
                 output_batchnorm=False,
                 bidirectional=False,
                 encoder_type='gru',
                 decoder_type='linear',
                 gamma_option='linear',
                 gamma_baseline=1,
                 num_modules=4,
                 module_num_layers=1,
                 module_dim=128,
                 parameter_efficient=False,
                 debug_every=float('inf'),
                 ):
        super(FiLMGen, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.output_batchnorm = output_batchnorm
        self.bidirectional = bidirectional
        self.num_dir = 2 if self.bidirectional else 1
        self.gamma_option = gamma_option
        self.gamma_baseline = gamma_baseline
        self.num_modules = num_modules
        self.module_num_layers = module_num_layers
        self.module_dim = module_dim
        self.debug_every = debug_every
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        if self.bidirectional:
            if decoder_type != 'linear':
                raise (NotImplementedError)
            hidden_dim = (int)(hidden_dim / self.num_dir)

        self.func_list = {
            'linear': None,
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'exp': torch.exp,
        }

        self.cond_feat_size = 2 * self.module_dim * self.module_num_layers  # FiLM params per ResBlock
        if not parameter_efficient:  # parameter_efficient=False only used to load older trained models
            self.cond_feat_size = 4 * self.module_dim + 2 * self.num_modules

        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = init_rnn(self.encoder_type, wordvec_dim, hidden_dim, rnn_num_layers,
                                    dropout=rnn_dropout, bidirectional=self.bidirectional)
        self.decoder_rnn = init_rnn(self.decoder_type, hidden_dim, hidden_dim, rnn_num_layers,
                                    dropout=rnn_dropout, bidirectional=self.bidirectional)
        self.decoder_linear = nn.Linear(
            hidden_dim * self.num_dir, self.num_modules * self.cond_feat_size)
        if self.output_batchnorm:
            self.output_bn = nn.BatchNorm1d(self.cond_feat_size, affine=True)

        init_modules(self.modules())

    def expand_encoder_vocab(self, token_to_idx, word2vec=None, std=0.01):
        expand_embedding_vocab(self.encoder_embed, token_to_idx,
                               word2vec=word2vec, std=std)

    def get_dims(self, x=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.cond_feat_size
        D = self.encoder_embed.embedding_dim
        H = self.encoder_rnn.hidden_size
        H_full = self.encoder_rnn.hidden_size * self.num_dir
        L = self.encoder_rnn.num_layers * self.num_dir

        N = x.size(0) if x is not None else None
        T_in = x.size(1) if x is not None else None
        T_out = self.num_modules
        return V_in, V_out, D, H, H_full, L, N, T_in, T_out

    def before_rnn(self, x, replace=0):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence.
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
        V_in, V_out, D, H, H_full, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)  # Tokenized word sequences (questions), end index
        embed = self.encoder_embed(x)
        h0 = Variable(torch.zeros(L, N, H).type_as(embed.data))

        if self.encoder_type == 'lstm':
            c0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
            out, _ = self.encoder_rnn(embed, (h0, c0))
        elif self.encoder_type == 'gru':
            out, _ = self.encoder_rnn(embed, h0)

        # Pull out the hidden state for the last non-null value in each input
        idx = idx.view(N, 1, 1).expand(N, 1, H_full)
        return out.gather(1, idx).view(N, H_full)

    def decoder(self, encoded, dims, h0=None, c0=None):
        V_in, V_out, D, H, H_full, L, N, T_in, T_out = dims

        if self.decoder_type == 'linear':
            # (N x H) x (H x T_out*V_out) -> (N x T_out*V_out) -> N x T_out x V_out
            return self.decoder_linear(encoded).view(N, T_out, V_out), (None, None)

        encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
        if not h0:
            h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))

        if self.decoder_type == 'lstm':
            if not c0:
                c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
            rnn_output, (ht, ct) = self.decoder_rnn(encoded_repeat, (h0, c0))
        elif self.decoder_type == 'gru':
            ct = None
            rnn_output, ht = self.decoder_rnn(encoded_repeat, h0)

        rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
        linear_output = self.decoder_linear(rnn_output_2d)
        if self.output_batchnorm:
            linear_output = self.output_bn(linear_output)
        output_shaped = linear_output.view(N, T_out, V_out)
        return output_shaped, (ht, ct)

    def forward(self, x):
        encoded = self.encoder(x)
        film_pre_mod, _ = self.decoder(encoded, self.get_dims(x=x))
        film = self.modify_output(film_pre_mod, gamma_option=self.gamma_option,
                                  gamma_shift=self.gamma_baseline)
        return film

    def modify_output(self, out, gamma_option='linear', gamma_scale=1, gamma_shift=0,
                      beta_option='linear', beta_scale=1, beta_shift=0):
        gamma_func = self.func_list[gamma_option]
        beta_func = self.func_list[beta_option]

        gs = []
        bs = []
        for i in range(self.module_num_layers):
            gs.append(slice(i * (2 * self.module_dim), i * (2 * self.module_dim) + self.module_dim))
            bs.append(slice(i * (2 * self.module_dim) + self.module_dim, (i + 1) * (2 * self.module_dim)))

        if gamma_func is not None:
            for i in range(self.module_num_layers):
                out[:, :, gs[i]] = gamma_func(out[:, :, gs[i]])
        if gamma_scale != 1:
            for i in range(self.module_num_layers):
                out[:, :, gs[i]] = out[:, :, gs[i]] * gamma_scale
        if gamma_shift != 0:
            for i in range(self.module_num_layers):
                out[:, :, gs[i]] = out[:, :, gs[i]] + gamma_shift
        if beta_func is not None:
            for i in range(self.module_num_layers):
                out[:, :, bs[i]] = beta_func(out[:, :, bs[i]])
            # out[:, :, b2] = beta_func(out[:, :, b2])
        if beta_scale != 1:
            for i in range(self.module_num_layers):
                out[:, :, bs[i]] = out[:, :, bs[i]] * beta_scale
        if beta_shift != 0:
            for i in range(self.module_num_layers):
                out[:, :, bs[i]] = out[:, :, bs[i]] + beta_shift
        return out


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform_
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_params(m.weight)


def init_rnn(rnn_type, hidden_dim1, hidden_dim2, rnn_num_layers,
             dropout=0, bidirectional=False):
    if rnn_type == 'gru':
        return nn.GRU(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                      batch_first=True, bidirectional=bidirectional)
    elif rnn_type == 'lstm':
        return nn.LSTM(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                       batch_first=True, bidirectional=bidirectional)
    elif rnn_type == 'linear':
        return None
    else:
        print('RNN type ' + str(rnn_type) + ' not yet implemented.')
        raise (NotImplementedError)


def load_program_generator(path, model_type='PG+EE'):
    checkpoint = load_cpu(path)
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    if model_type == 'FiLM':
        #print('Loading FiLMGen from ' + path)
        kwargs = get_updated_args(kwargs, FiLMGen)
        model = FiLMGen(**kwargs)
    else:
        #print('Loading PG from ' + path)
        model = Seq2Seq(**kwargs)
    model.load_state_dict(state)
    return model, kwargs


def load_execution_engine(path, verbose=True, model_type='PG+EE'):
    checkpoint = load_cpu(path)
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    kwargs['verbose'] = verbose
    if model_type == 'FiLM':
        #print('Loading FiLMedNet from ' + path)
        kwargs = get_updated_args(kwargs, FiLMedNet)
        model = FiLMedNet(**kwargs)
    else:
        #print('Loading EE from ' + path)
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


def run_single_example(args, model, tuple_mode='IEP'):
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
            if tuple_mode == 'IEP':
                predicted_program = program_generator.reinforce_sample(
                    question_var,
                    temperature=args['temperature'],
                    argmax=(args['sample_argmax'] == 1))
                with torch.no_grad():
                    scores = execution_engine(feats_var, predicted_program)
            else:
                predicted_program = program_generator(question_var)
                with torch.no_grad():
                    scores = execution_engine(feats_var, predicted_program, save_activations=False)
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


def run_batch(args, model, tuple_mode='IEP'):
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
        run_our_model_batch(program_generator, execution_engine, loader, dtype, tuple_mode)
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
        answer = inference(q, i)
        if str(answer[0]).lower() == str(a).lower():
            overall_score += 1
            if fn in question_family_score:
                question_family_score[fn] += 1
            else:
                question_family_score.update({fn: 1})
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


def run_our_model_batch(program_generator, execution_engine, loader, dtype, tuple_mode='IEP'):
    program_generator.type(dtype)
    program_generator.eval()
    execution_engine.type(dtype)
    execution_engine.eval()

    num_correct, num_samples = 0, 0
    print(f"Testing for {len(loader) * 32} samples")
    print()
    for batch in loader:
        questions, images, feats, answers, _, _, image_index = batch

        questions_var = Variable(questions.type(dtype).long())
        feats_var = Variable(feats.type(dtype))

        if tuple_mode == 'IEP':
            programs_pred = program_generator.reinforce_sample(
                questions_var,
                temperature=1,
                argmax=True)

            scores = execution_engine(feats_var, programs_pred)
        else:
            programs_pred = program_generator(questions_var)
            scores = execution_engine(feats_var, programs_pred, save_activations=False)

        _, preds = scores.data.cpu().max(1)

        ##################################
        model_answers = preds.numpy()
        real_answers = answers.numpy()
        hmm = np.where(model_answers != real_answers)[0]

        with open('./validation_culprits.txt', 'a+') as fout:
            for hmmidx in hmm:
                fout.write(f"Image ID: {image_index[hmmidx].view(1, ).numpy()[0]}\n")
                fout.write(f"Question: {' '.join(decode_question(questions[hmmidx]))}\n")
                fout.write(f"Model Answer: {decode_answer(model_answers[hmmidx])[0]}\n")
                fout.write(f"Correct Answer: {decode_answer(real_answers[hmmidx])[0]}\n")
                fout.write('\n')

        ##################################

        num_correct += (preds == answers).sum()
        num_samples += preds.size(0)
        if num_samples % (32 * 100) == 0:
            print(f'Ran {num_samples} samples at {float(num_correct) / num_samples} accuracy')
            sys.exit(1)

    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))

