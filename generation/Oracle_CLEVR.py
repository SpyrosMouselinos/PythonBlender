import json
import random

import question_engine as qeng


### Utilities.py ###
def node_shallow_copy(node):
    new_node = {
        'type': node['type'],
        'inputs': node['inputs'],
    }
    if 'side_inputs' in node:
        new_node['side_inputs'] = node['side_inputs']
    return new_node


def precompute_filter_options(scene_struct, metadata):
    # Keys are tuples (size, color, shape, material) (where some may be None)
    # and values are lists of object idxs that match the filter criterion
    attribute_map = {}

    if metadata['dataset'] == 'CLEVR-v1.0':
        attr_keys = ['size', 'color', 'material', 'shape']
    else:
        assert False, 'Unrecognized dataset'

    # Precompute masks
    masks = []
    for i in range(2 ** len(attr_keys)):
        mask = []
        for j in range(len(attr_keys)):
            mask.append((i // (2 ** j)) % 2)
        masks.append(mask)

    for object_idx, obj in enumerate(scene_struct['objects']):
        if metadata['dataset'] == 'CLEVR-v1.0':
            keys = [tuple(obj[k] for k in attr_keys)]

        for mask in masks:
            for key in keys:
                masked_key = []
                for a, b in zip(key, mask):
                    if b == 1:
                        masked_key.append(a)
                    else:
                        masked_key.append(None)
                masked_key = tuple(masked_key)
                if masked_key not in attribute_map:
                    attribute_map[masked_key] = set()
                attribute_map[masked_key].add(object_idx)

    scene_struct['_filter_options'] = attribute_map


def find_filter_options(object_idxs, scene_struct, metadata):
    # Keys are tuples (size, color, shape, material) (where some may be None)
    # and values are lists of object idxs that match the filter criterion

    if '_filter_options' not in scene_struct:
        precompute_filter_options(scene_struct, metadata)

    attribute_map = {}
    object_idxs = set(object_idxs)
    for k, vs in scene_struct['_filter_options'].items():
        attribute_map[k] = sorted(list(object_idxs & vs))
    return attribute_map


def find_relate_filter_options(object_idx, scene_struct, metadata,
                               unique=False, include_zero=False, trivial_frac=0.1):
    options = {}
    if '_filter_options' not in scene_struct:
        precompute_filter_options(scene_struct, metadata)

    # TODO: Right now this is only looking for nontrivial combinations; in some
    # cases I may want to add trivial combinations, either where the intersection
    # is empty or where the intersection is equal to the filtering output.
    trivial_options = {}
    for relationship in scene_struct['relationships']:
        related = set(scene_struct['relationships'][relationship][object_idx])
        for filters, filtered in scene_struct['_filter_options'].items():
            intersection = related & filtered
            trivial = (intersection == filtered)
            if unique and len(intersection) != 1: continue
            if not include_zero and len(intersection) == 0: continue
            if trivial:
                trivial_options[(relationship, filters)] = sorted(list(intersection))
            else:
                options[(relationship, filters)] = sorted(list(intersection))

    N, f = len(options), trivial_frac
    num_trivial = int(round(N * f / (1 - f)))
    trivial_options = list(trivial_options.items())
    random.shuffle(trivial_options)
    for k, v in trivial_options[:num_trivial]:
        options[k] = v

    return options


def add_empty_filter_options(attribute_map, metadata, num_to_add):
    # Add some filtering criterion that do NOT correspond to objects

    if metadata['dataset'] == 'CLEVR-v1.0':
        attr_keys = ['Size', 'Color', 'Material', 'Shape']
    else:
        assert False, 'Unrecognized dataset'

    attr_vals = [metadata['types'][t] + [None] for t in attr_keys]
    if '_filter_options' in metadata:
        attr_vals = metadata['_filter_options']

    target_size = len(attribute_map) + num_to_add
    while len(attribute_map) < target_size:
        k = (random.choice(v) for v in attr_vals)
        if k not in attribute_map:
            attribute_map[k] = []


### Utilities.py ###
def instantiate_templates_dfs(scene_struct, program, metadata):

    q = {'nodes': program}
    outputs = qeng.answer_question(q, metadata, scene_struct,  all_outputs=True)
    answer = outputs[-1]
    if answer == '__INVALID__':
        return answer
    return answer

with open('./metadata.json', 'r') as f:
    metadata = json.load(f)
    dataset = metadata['dataset']
    functions_by_name = {}
    for f in metadata['functions']:
        functions_by_name[f['name']] = f
    metadata['_functions_by_name'] = functions_by_name

scene_struct = dict({'split': 'Rendered', 'directions': {'below': [-0.0, -0.0, -1.0],
                                                         'front': [0.754490315914154, -0.6563112735748291, -0.0],
                                                         'above': [0.0, 0.0, 1.0],
                                                         'right': [0.6563112735748291, 0.7544902563095093, -0.0],
                                                         'behind': [-0.754490315914154, 0.6563112735748291, 0.0],
                                                         'left': [-0.6563112735748291, -0.7544902563095093, 0.0]},
                     'objects': [{'3d_coords': [-1.4927695989608765, -2.0407912731170654, 0.699999988079071],
                                  'shape': 'cylinder', 'rotation': 0.7320286457359669, 'size': 'large',
                                  'color': 'brown', 'pixel_coords': [117, 135, 10.785210609436035],
                                  'material': 'rubber'},
                                 {'3d_coords': [1.5566600561141968, -2.1519246101379395, 0.699999988079071],
                                  'shape': 'cube', 'rotation': 0.7320286457359669, 'size': 'large', 'color': 'gray',
                                  'pixel_coords': [203, 197, 8.6880521774292], 'material': 'rubber'},
                                 {'3d_coords': [-2.341233015060425, -0.5676895380020142, 0.3499999940395355],
                                  'shape': 'cylinder', 'rotation': 0.38202865169643135, 'size': 'small',
                                  'color': 'green', 'pixel_coords': [157, 118, 12.36081600189209],
                                  'material': 'rubber'},
                                 {'3d_coords': [-0.8063592314720154, 1.8669357299804688, 0.699999988079071],
                                  'shape': 'sphere', 'rotation': 0.7320286457359669, 'size': 'large', 'color': 'purple',
                                  'pixel_coords': [277, 98, 12.562734603881836], 'material': 'metal'},
                                 {'3d_coords': [2.677332878112793, -0.01264934055507183, 0.3499999940395355],
                                  'shape': 'cube', 'rotation': 0.38202865169643135, 'size': 'small', 'color': 'gray',
                                  'pixel_coords': [338, 198, 9.331548690795898], 'material': 'metal'}],
                     'relationships': {'right': [[1, 2, 3, 4], [3, 4], [1, 3, 4], [4], []],
                                       'front': [[1, 4], [], [0, 1, 4], [0, 1, 2, 4], [1]],
                                       'behind': [[2, 3], [0, 2, 3, 4], [3], [], [0, 2, 3]],
                                       'left': [[], [0, 2], [0], [0, 1, 2], [0, 1, 2, 3]]},
                     'image_filename': 'CLEVR_Rendered_000000.png', 'image_index': 0})

with open('../questions/CLEVR_Rendered_questions.json', 'r') as fin:
    questions = json.load(fin)
    questions = questions['questions']

for trial in questions:
    program = []
    dirty_program = trial['program']
    for dirty_entry in dirty_program:
        if dirty_entry['type'] == 'scene':
            pass
        else:
            dirty_entry.pop('_output')
        program.append(dirty_entry)
    answer = trial['answer']
    qq = trial['question']
    print(f"Original Question: [{qq}]\n")
    print(f"Original Answer: [{answer}]\n")
    print(f"Assumed Answer:")
    print(instantiate_templates_dfs(scene_struct=scene_struct, program=program, metadata=metadata))
    print("\n")