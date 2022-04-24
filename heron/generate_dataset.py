from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datetime import datetime
import cv2
import os

from tools.data_helper import generate_state_img, process_batch, sample_transition

seed = 42
max_steps_per_episode = 10000
show = False
episode_count = 0

sample_mode = 0
batch_size = 50
num_processed_batches = 0
num_batches = 20
save_format = 'pickle'

# Prepare environment
env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

model = keras.models.load_model('models/23.04.2022_16.56.08_model_500')

# Create directory for data
parent_dir = 'datasets'
dt_string = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
path = os.path.join(parent_dir, dt_string)
os.mkdir(path)

while num_processed_batches < num_batches:
    state = np.array(env.reset())
    batch = []
    
    for timestep in range(1, max_steps_per_episode):
        # Act according to our policy
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()

        state_next, _, done, _ = env.step(action)
        state_next = np.array(state_next)
        
        if sample_transition(timestep, episode_count, sample_mode):
            transition = [state, state_next, action]
            batch.append(transition)

        # batch data to prevent memory leaks and to organize data
        if len(batch) >= batch_size:
            process_batch(path, batch, num_processed_batches, num_batches, save_format)
            num_processed_batches += 1
            batch = []

        if show:
            winname=f'episode {episode_count}'
            generate_state_img(state, winname=winname, scale=3, display=True)
            
        state = state_next

        if done or num_processed_batches >= num_batches:
            if show:
                cv2.destroyAllWindows()
            break

    episode_count += 1
   