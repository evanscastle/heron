import cv2
import os
import pickle

def load_transition(filename, load_format):
    if load_format=='pickle':
        return pickle.load(open(filename, "rb" ))
        
    else:
        return False, 'Load format not understood'

# Write a transition to disk according to a specified save format
def save_transition(filename, transition, save_format):
    if save_format=='pickle':
        pickle.dump(transition, open(filename, "wb" ))

    else:
        assert False, 'Save format not understood'


# Save all of the data in a batch to disk according to save_format param
def process_batch(batch_dir, batch, num_processed_batches, save_format='pickle'):
    path = os.path.join(batch_dir, str(num_processed_batches))
    os.mkdir(path)

    for i, transition in enumerate(batch):
        name = f'{path}/{i}'
        save_transition(name, transition, save_format)


# A method to determine whether a particular transition will be included in the dataset
def sample_transition(timestep, episode_count, sample_mode):
    if sample_mode == 0:
        if timestep % 10 == 0:
            return True

        else:
            return False

    else:
        assert False, 'Invalid sample mode'


# Using state information, generate an image to be either displayed or returned
def generate_frame(state, winname=None, scale=1, show_action=False, display=False, ret=False):
    width = int(state.shape[1] * scale)
    height = int(state.shape[0] * scale)
    dim = (width, height)

    resized_state = cv2.resize(state, dim, interpolation = cv2.INTER_AREA)
    
    if display:
        if winname is None:
            winname = 'agent'
        cv2.namedWindow(winname) 
        cv2.moveWindow(winname, 40,30)
        cv2.imshow(winname, resized_state)
        cv2.waitKey(50)

    if ret:
        return resized_state