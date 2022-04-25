from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datetime import datetime
import cv2
import os

seed = 42
max_steps_per_episode = 10000
show = True
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

model = keras.models.load_model('agents/23.04.2022_23.17.43/agent_ep=10000_seed=42')

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

        scale = 5
        width = int(state.shape[1] * scale)
        height = int(state.shape[0] * scale)
        dim = (width, height)

        img = cv2.resize(state, dim, interpolation = cv2.INTER_AREA)

        winname = f'episode {episode_count}'
        cv2.namedWindow(winname) 
        #cv2.moveWindow(winname, 40,30)
        cv2.imshow(winname, img)
        cv2.waitKey(0)
            
        state = state_next

        if done or num_processed_batches >= num_batches:
            if show:
                cv2.destroyAllWindows()
            break

    episode_count += 1
   