import argparse
import random
import time
import warnings

import numpy as np
import torch

from agents.models import RandAgent, RL_Bandit_Agent
from experiment_utils.constants import SPLIT_, OUTPUT_IMAGE_DIR_, OUTPUT_SCENE_DIR_, OUTPUT_SCENE_FILE_, \
    OUTPUT_QUESTION_FILE_, UP_TO_HERE_
from experiment_utils.helpers import render_image, make_questions, extract_features, load_resnet_backbone, \
    initialize_paths
from generation.my_run_model import load_cnn_sa, load_iep, inference_with_iep, inference_with_cnn_sa

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(666)
np.random.seed(666)


def test_generate_images_with_agent(max_images=1000, batch_size=16, max_episodes=10000, workers=1):
    global_generation_success_rate = 0
    agent = RandAgent()

    ### MAIN LOOP ###
    episodes = 0
    start_time = time.time()
    while episodes < max_episodes:

        ## Suggest a configuration
        camera_control, object_scms, object_3d_pos = agent(objects=-1, start_idx=episodes * batch_size,
                                                           batch_size=batch_size,
                                                           randomize_idx=False)

        ### Move to Numpy Format
        key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = camera_control
        num_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes = object_scms
        per_image_x, per_image_y, per_image_theta = object_3d_pos
        #### Render it ###
        attempted_images = render_image(key_light_jitter=key_light_jitter,
                                        fill_light_jitter=fill_light_jitter,
                                        back_light_jitter=back_light_jitter,
                                        camera_jitter=camera_jitter,
                                        num_objects=num_objects,
                                        per_image_shapes=per_image_shapes,
                                        per_image_colors=per_image_colors,
                                        per_image_materials=per_image_materials,
                                        per_image_sizes=per_image_sizes,
                                        per_image_x=per_image_x,
                                        per_image_y=per_image_y,
                                        per_image_theta=per_image_theta,
                                        num_images=batch_size,
                                        workers=workers,
                                        split=SPLIT_,
                                        assemble_after=False,
                                        start_idx=episodes * batch_size,
                                        )
        correct_images = [f[0] for f in attempted_images if f[1] == 1]
        global_generation_success_rate += len(correct_images)
        if global_generation_success_rate >= max_images:
            break
        episodes += 1
    end_time = time.time()
    ips = global_generation_success_rate / round(end_time - start_time, 2)
    duration = round(end_time - start_time, 2)
    print(f"Took {end_time - start_time} time for {global_generation_success_rate} images")
    print(f"Images per second {ips}")
    print(f"Generator Success Rate: {round(global_generation_success_rate / (max_episodes * batch_size), 2)}")
    return duration


def test_generate_images_and_answer_with_fbai_testbed(max_images=1000, batch_size=16, max_episodes=10000, workers=1):
    cnn = load_resnet_backbone(dtype=torch.cuda.FloatTensor)

    agent = RL_Bandit_Agent(bandit_mode='UCB')

    cnn_sa = load_cnn_sa('../models/CLEVR/cnn_lstm_sa_mlp.pt')
    iep = load_iep('../models/CLEVR/program_generator_700k.pt', '../models/CLEVR/execution_engine_700k_strong.pt')
    # film = load_film('../models/CLEVR/film.pt', '../models/CLEVR/film.pt'

    global_accuracy_sa = []
    global_accuracy_iep = []
    global_agent_rewards = []
    global_generation_success_rate = 0

    ### MAIN LOOP ###
    episodes = 0
    start_time = time.time()
    while episodes < max_episodes:
        ## Suggest a configuration
        camera_control, object_scms, object_3d_pos, actions = agent(objects=-1, start_idx=episodes * batch_size,
                                                                    batch_size=batch_size, randomize_idx=False)

        ### Move to Numpy Format
        key_light_jitter, fill_light_jitter, back_light_jitter, camera_jitter = camera_control
        per_image_objects, per_image_shapes, per_image_colors, per_image_materials, per_image_sizes = object_scms
        per_image_x, per_image_y, per_image_theta = object_3d_pos

        #### Render it ###
        attempted_images = render_image(key_light_jitter=key_light_jitter,
                                        fill_light_jitter=fill_light_jitter,
                                        back_light_jitter=back_light_jitter,
                                        camera_jitter=camera_jitter,
                                        per_image_shapes=per_image_shapes,
                                        per_image_colors=per_image_colors,
                                        per_image_materials=per_image_materials,
                                        per_image_sizes=per_image_sizes,
                                        per_image_x=per_image_x,
                                        per_image_y=per_image_y,
                                        per_image_theta=per_image_theta,
                                        num_images=batch_size,
                                        workers=workers,
                                        split=SPLIT_,
                                        assemble_after=False,
                                        start_idx=episodes * batch_size,
                                        )
        correct_images = [f[0] for f in attempted_images if f[1] == 1]
        action_mask = np.array([f[1] for f in attempted_images])
        global_generation_success_rate += len(correct_images)
        if global_generation_success_rate >= max_images:
            break

        actions = np.array(actions)[np.where(action_mask == 1)[0]]
        if len(correct_images) > 0:
            generator = make_questions(input_scene_file=None,
                                       word_replace_dict={'True': 'yes', 'False': 'no'},
                                       output_questions_file='/content/gdrive/MyDrive/blender_agents/official_val/CLEVR_val_questions.json',
                                       mock=True, mock_name='Rendered')
            score_sa = 0
            score_iep = 0
            examples = 0
            rewards = []
            for i, q, a in generator():
                agent_reward = 0
                try:
                    feats = extract_features(i, cnn, dtype=torch.cuda.FloatTensor)
                except FileNotFoundError:
                    continue
                answers_sa = inference_with_cnn_sa(questions=q,
                                                   image=None,
                                                   feats=feats,
                                                   vocab_json='./models/CLEVR/vocab.json',
                                                   input_questions_h5=None,
                                                   input_features_h5=None,
                                                   use_gpu=1,
                                                   external=True,
                                                   batch_size=1, model=cnn_sa)
                answers_iep = inference_with_iep(questions=q,
                                                 image=None,
                                                 feats=feats,
                                                 vocab_json='./models/CLEVR/vocab.json',
                                                 input_questions_h5=None,
                                                 input_features_h5=None,
                                                 use_gpu=1,
                                                 external=True,
                                                 batch_size=1, model=iep)
                for idx, (answer_sa, answer_iep, correct_answer) in enumerate(zip(answers_sa, answers_iep, a)):
                    if str(answer_sa).lower() == str(correct_answer).lower():
                        score_sa += 1
                    else:
                        agent_reward += 1

                    if str(answer_iep).lower() == str(correct_answer).lower():
                        score_iep += 1
                    else:
                        agent_reward += 1

                    examples += 1
                # Score of Single Image #
                agent_reward = agent_reward / (idx + 1)
                agent_reward = agent_reward / 3.0
                rewards.append(agent_reward)
                global_agent_rewards.append(agent_reward)
                print(f"Image Action Reward: {round(agent_reward, 3)}")
            print(f"Episode SA Accuracy: {round(score_sa / examples, 3)}")
            print(f"Episode IEP Accuracy: {round(score_iep / examples, 3)}")

            global_accuracy_sa.append(score_sa / examples)
            global_accuracy_iep.append(score_iep / examples)

        print(actions)
        print(rewards)
        agent.learn(actions, rewards)
        episodes += 1
    end_time = time.time()
    print(f"Took {end_time - start_time} time for {global_generation_success_rate} images")
    print(f"Images per second {round(end_time - start_time, 2) / global_generation_success_rate}")
    print(f"Generator Success Rate: {round(global_generation_success_rate / (max_episodes * batch_size), 2)}")
    return


#


#

#
#
#
#         global_generation_success_rate += len(correct_images)
#         if global_generation_success_rate >= max_images:
#             break
#         episodes += 1
#     end_time = time.time()
#     ips = global_generation_success_rate / round(end_time - start_time, 2)
#     duration = round(end_time - start_time, 2)
#     print(f"Took {end_time - start_time} time for {global_generation_success_rate} images")
#     print(f"Images per second {ips}")
#     print(f"Generator Success Rate: {round(global_generation_success_rate / (max_episodes * batch_size), 2)}")
#     return duration
#
#
#
#
#
#             for idx, (answer_sa, answer_iep, answer_film, correct_answer) in enumerate(
#                     zip(answers_sa, answers_iep, answers_film, a)):
#                 if str(answer_sa).lower() == str(correct_answer).lower():
#                     score_sa += 1
#                 else:
#                     agent_reward += 1
#
#                 if str(answer_iep).lower() == str(correct_answer).lower():
#                     score_iep += 1
#                 else:
#                     agent_reward += 1
#
#                 if str(answer_film).lower() == str(correct_answer).lower():
#                     score_film += 1
#                 else:
#                     agent_reward += 1
#
#                 examples += 1
#             # Score of Single Image #
#             agent_reward = agent_reward / (idx + 1)
#             agent_reward = agent_reward / 3.0
#             rewards.append(agent_reward)
#             global_agent_rewards.append(agent_reward)
#             print(f"Image Action Reward: {round(agent_reward, 3)}")
#         print(f"Episode SA Accuracy: {round(score_sa / examples, 3)}")
#         print(f"Episode IEP Accuracy: {round(score_iep / examples, 3)}")
#         print(f"Episode FiLM Accuracy: {round(score_film / examples, 3)}")
#
#         global_accuracy_sa.append(score_sa / examples)
#         global_accuracy_iep.append(score_iep / examples)
#         global_accuracy_film.append(score_film / examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpus', type=str, default=1)
    parser.add_argument('--output_image_dir', type=str, default=OUTPUT_IMAGE_DIR_)
    parser.add_argument('--output_scene_dir', type=str, default=OUTPUT_SCENE_DIR_)
    parser.add_argument('--output_scene_file', type=str, default=OUTPUT_SCENE_FILE_)
    parser.add_argument('--output_question_file', type=str, default=OUTPUT_QUESTION_FILE_)

    args = parser.parse_args()
    ngpus = str(args.ngpus)
    output_image_dir = str(args.output_image_dir)
    output_scene_dir = str(args.output_scene_dir)
    output_scene_file = str(args.output_scene_file)
    output_question_file = str(args.output_question_file)

    initialize_paths(output_scene_dir=OUTPUT_SCENE_DIR_, output_image_dir=OUTPUT_IMAGE_DIR_, up_to_here=UP_TO_HERE_)
    print("Results\n")
    print(
        f"{test_generate_images_with_agent(max_images=400, batch_size=16, max_episodes=10000, workers=1)} | BS 16 | W 1 | GPU {ngpus}")
    print(
        f"{test_generate_images_with_agent(max_images=400, batch_size=16, max_episodes=10000, workers=4)} | BS 16 | W 4 | GPU {ngpus}")
    print(
        f"{test_generate_images_with_agent(max_images=400, batch_size=16, max_episodes=10000, workers=8)} | BS 16 | W 8 | GPU {ngpus}")
