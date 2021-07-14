import argparse
import os
import random
import time
import warnings
import numpy as np

from agents.models import RandAgent
from experiment_utils.constants import UP_TO_HERE_, \
    OUTPUT_SCENE_FILE_, \
    OUTPUT_QUESTION_FILE_, \
    OUTPUT_SCENE_DIR_, \
    OUTPUT_IMAGE_DIR_, SPLIT_
from experiment_utils.helpers import render_image, initialize_paths, make_questions

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(666)
np.random.seed(666)


def generate_images_with_agent(max_images=6, batch_size=4, max_episodes=100):
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
                                        workers=1,
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
    print(f"Took {end_time - start_time} time for {global_generation_success_rate} images")
    print(f"Images per second {global_generation_success_rate / round(end_time - start_time, 2)}")
    print(f"Generator Success Rate: {round(global_generation_success_rate / (max_episodes * batch_size), 2)}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_image_dir', type=str, default=OUTPUT_IMAGE_DIR_)
    parser.add_argument('--output_scene_dir', type=str, default=OUTPUT_SCENE_DIR_)
    parser.add_argument('--output_scene_file', type=str, default=OUTPUT_SCENE_FILE_)
    parser.add_argument('--output_question_file', type=str, default=OUTPUT_QUESTION_FILE_)

    args = parser.parse_args()
    output_image_dir = str(args.output_image_dir)
    output_scene_dir = str(args.output_scene_dir)
    output_scene_file = str(args.output_scene_file)
    output_question_file = str(args.output_question_file)

    initialize_paths(output_scene_dir=OUTPUT_SCENE_DIR_, output_image_dir=OUTPUT_IMAGE_DIR_, up_to_here=UP_TO_HERE_)
    generate_images_with_agent()
