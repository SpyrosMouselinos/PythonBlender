import argparse
import random
import time
import warnings

import numpy as np

from agents.models import InformedRandomAgent
from experiment_utils.constants import UP_TO_HERE_, \
    OUTPUT_SCENE_FILE_, \
    OUTPUT_QUESTION_FILE_, \
    OUTPUT_SCENE_DIR_, \
    OUTPUT_IMAGE_DIR_
from experiment_utils.helpers import render_image, initialize_paths

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(666)
np.random.seed(666)


def generate_images_with_agent(output_image_dir: str,
                               output_scene_dir: str,
                               output_scene_file: str,
                               up_to_here: str,
                               split: str,
                               max_images=6, batch_size=4, max_episodes=100):
    global_generation_success_rate = 0
    agent = InformedRandomAgent()

    def add_gaussian(x, loc=0, scale=1):
        return x + np.random.normal(loc, scale)

    def add_uniform(x, low=-0.1, high=0.1):
        return x + np.random.uniform(low, high)

    agent.register_noise_gen(add_gaussian)

    ### MAIN LOOP ###
    episodes = 0
    start_time = time.time()
    while episodes < max_episodes:

        ## Suggest a configuration
        camera_control, object_scms, object_3d_pos = agent(objects=-1, start_idx=episodes * batch_size,
                                                           batch_size=batch_size,
                                                           # indexes=list(range(episodes * batch_size,
                                                           #                   (episodes + 1) * batch_size)),
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
                                        workers=16,
                                        split=split,
                                        assemble_after=False,
                                        image_indexes=list(range(episodes * batch_size, episodes + 1 * batch_size)),
                                        output_image_dir=output_image_dir,
                                        output_scene_dir=output_scene_dir,
                                        output_scene_file=output_scene_file,
                                        up_to_here=up_to_here,
                                        )
        correct_images = [f[0] for f in attempted_images if f[1] == 1]
        global_generation_success_rate += len(correct_images)
        if global_generation_success_rate >= max_images:
            break
        episodes += 1
    end_time = time.time()
    print(f"Took {end_time - start_time} time for {global_generation_success_rate} images")
    print(f"Images per second {round(end_time - start_time, 2) / global_generation_success_rate}")
    print(f"Generator Success Rate: {round(global_generation_success_rate / (max_episodes * batch_size), 2)}")


# def conduct_experiments_on_images():
#     global_accuracy_sa = []
#     global_accuracy_iep = []
#     global_accuracy_film = []
#
#     ########### Loop Over the Image / Question Pairs ###########################
#     score_sa = score_iep = score_film = 0
#     examples = 0
#     not_found = 0
#     episodes = 0
#     # generator = make_questions(word_replace_dict={'True':'yes','False':'no'},
#     #                             output_questions_file = '/content/gdrive/MyDrive/blender_agents/official_val/CLEVR_val_questions.json',
#     #                             mock=True, mock_name='Gauss_Scale_1')
#     # generator = make_questions(word_replace_dict={'True':'yes','False':'no'},
#     #                             input_scene_file = '/content/gdrive/MyDrive/blender_agents/scenes/CLEVR_Gauss_Scale_1_scenes.json',
#     #                             output_questions_file = '/content/gdrive/MyDrive/blender_agents/questions/CLEVR_Gauss_Scale_1_questions.json',
#     #                             mock=False, mock_name=None)
#     generator = make_questions(word_replace_dict={'True': 'yes', 'False': 'no'},
#                                output_questions_file='/content/gdrive/MyDrive/blender_agents/official_val/CLEVR_val_questions.json',
#                                mock=True, mock_name=None,
#                                mock_image_dir='/content/gdrive/MyDrive/blender_agents/official_val/CLEVR_v1.0/images/val')
#
#     for i, q, a in generator():
#         try:
#             if os.path.exists('/content/gdrive/MyDrive/blender_agents/images/CLEVR_Gauss_Scale_1_' + i.split('_')[-1]):
#                 yyy = os.path.exists(i)
#                 print(f"Gauss Val Exists, Original reciprocate..{yyy}")
#             else:
#                 continue
#             feats = extract_features(i, cnn, dtype=torch.cuda.FloatTensor)
#         except:
#             print(f"Image {i} not found...")
#             not_found += 1
#             if not_found > 1000:
#                 break
#             else:
#                 continue
#         answers_sa = inference_with_cnn_sa(questions=q,
#                                            image=None,
#                                            feats=feats,
#                                            baseline_model='../models/CLEVR/cnn_lstm_sa_mlp.pt',
#                                            vocab_json='../models/CLEVR/vocab.json',
#                                            input_questions_h5=None,
#                                            input_features_h5=None,
#                                            use_gpu=1,
#                                            external=True,
#                                            batch_size=1)
#         answers_iep = inference_with_iep(questions=q,
#                                          image=None,
#                                          feats=feats,
#                                          program_generator='../models/CLEVR/program_generator_700k.pt',
#                                          execution_engine='../models/CLEVR/execution_engine_700k_strong.pt',
#                                          vocab_json='../models/CLEVR/vocab.json',
#                                          input_questions_h5=None,
#                                          input_features_h5=None,
#                                          use_gpu=1,
#                                          external=True,
#                                          batch_size=1)
#         answers_film = inference_with_film(questions=q,
#                                            image=None,
#                                            feats=feats,
#                                            program_generator='../models/CLEVR/film.pt',
#                                            execution_engine='../models/CLEVR/film.pt',
#                                            vocab_json='../models/CLEVR/vocab.json',
#                                            input_questions_h5=None,
#                                            input_features_h5=None,
#                                            use_gpu=1,
#                                            external=True,
#                                            batch_size=1)
#
#         for idx, (answer_sa, answer_iep, answer_film, correct_answer) in enumerate(
#                 zip(answers_sa, answers_iep, answers_film, a)):
#             if str(answer_sa).lower() == str(correct_answer).lower():
#                 score_sa += 1
#             if str(answer_iep).lower() == str(correct_answer).lower():
#                 score_iep += 1
#             if str(answer_film).lower() == str(correct_answer).lower():
#                 score_film += 1
#
#             examples += 1
#         print(f"Episode SA Accuracy: {round(score_sa / examples, 3)}")
#         print(f"Episode IEP Accuracy: {round(score_iep / examples, 3)}")
#         print(f"Episode FiLM Accuracy: {round(score_film / examples, 3)}")
#
#         global_accuracy_sa.append(score_sa / examples)
#         global_accuracy_iep.append(score_iep / examples)
#         global_accuracy_film.append(score_film / examples)
#         episodes += 1
#     return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='Gauss_1')
    parser.add_argument('--output_image_dir', type=str, default=OUTPUT_IMAGE_DIR_)
    parser.add_argument('--output_scene_dir', type=str, default=OUTPUT_SCENE_DIR_)
    parser.add_argument('--output_scene_file', type=str, default=OUTPUT_SCENE_FILE_)
    parser.add_argument('--output_question_file', type=str, default=OUTPUT_QUESTION_FILE_)

    args = parser.parse_args()
    split = str(args.split)
    output_image_dir = str(args.output_image_dir)
    output_scene_dir = str(args.output_scene_dir)
    output_scene_file = str(args.output_scene_file)
    output_question_file = str(args.output_question_file)

    initialize_paths(output_scene_dir=output_scene_dir, output_image_dir=output_image_dir, up_to_here=UP_TO_HERE_)
    generate_images_with_agent(output_image_dir=output_image_dir, output_scene_dir=output_scene_dir,
                               output_scene_file=output_scene_file, split=split, up_to_here=UP_TO_HERE_)
