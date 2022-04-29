from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datetime import datetime
import cv2
import os

seed = 43
max_steps_per_episode = 2000
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

model = keras.models.load_model('agents/default')

# Create directory for data
parent_dir = 'datasets'
dt_string = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
path = os.path.join(parent_dir, dt_string)
os.mkdir(path)

write = True
if not write:
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
            cv2.waitKey(50)
                
            state = state_next

            if done or num_processed_batches >= num_batches:
                if show:
                    cv2.destroyAllWindows()
                break

        episode_count += 1

# OpenCV is behaving very strangely so this is just a workaround that saves all the frames and compiles them as a video
else:
    
    initialized = False
    state = np.array(env.reset())
    for timestep in range(1, max_steps_per_episode):
        # Act according to our policy
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()

        state_next, _, done, _ = env.step(action)
        state_next = np.array(state_next)

        if not initialized:
            scale = 5
            width = int(state.shape[1] * scale)
            height = int(state.shape[0] * scale)
            dim = (width, height)
            #fourcc = cv2.VideoWriter_fourcc(*'x264')
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            #writer = cv2.VideoWriter('videos/sample.avi', fourcc, 20, dim)
            writer = cv2.VideoWriter('C:/Users/evans/Desktop/Code/heron/videos/sample.avi', fourcc, 20, dim)

        initialized = True

        img = cv2.resize(state, dim, interpolation=cv2.INTER_AREA)
        img *= 255
        img = img.astype('uint8')
        time_str = str(timestep).zfill(5)
        cv2.imwrite(f'C:/Users/evans/Desktop/Code/heron/images/frame_{time_str}.png', img)
        #img = img.astype('uint8')
        #cv2.imshow('blah', img)
        #cv2.waitKey(2)
        #writer.write(img)
            
        state = state_next

    # writer.release()
    # cv2.destroyAllWindows()

    image_folder = 'C:/Users/evans/Desktop/Code/heron/images'
    video_name = 'video.mp4'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 20, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
