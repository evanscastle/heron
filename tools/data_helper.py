import os
import pickle
import cv2

from visualize import generate_transition_img

def load_transition(filename):
    save_format = filename.split('.')[-1]

    if save_format=='pkl':
        return pickle.load(open(filename, "rb"))

    else:
        return False, 'Load format not understood'


# Write a transition to disk according to a specified save format
def save_transition(name, transition, save_format):
    
    if save_format == 'pickle':
        
        filename = name + '.pkl'
        pickle.dump(transition, open(filename, "wb"))

    elif save_format == 'png':
        
        filename = name + '.png'
        img = generate_transition_img(transition, ret=True)
        cv2.imwrite(filename, img)

    else:
        assert False, 'Save format not understood'


# Save all of the data in a batch to disk according to save_format param
def process_batch(batch_dir, batch, num_processed_batches, num_batches, save_format):
    
    # Create directory for the batch
    path = os.path.join(batch_dir, 'batch_' + str(num_processed_batches).zfill(len(str(num_batches))))
    if not os.path.exists(path):
        os.mkdir(path)

    for i, transition in enumerate(batch):
        tran_num = str(i).zfill(len(str(len(batch))))
        name = f'{path}/{tran_num}'
        save_transition(name, transition, save_format)

    output_msg = f'Batch of {len(batch)} transitions successfully saved at {path}'
    print(output_msg)


# A method to determine whether a particular transition will be included in the dataset
def sample_transition(timestep, episode_count, sample_mode):
    
    # This sample mode is here for testing purposes; it samples every transition
    if sample_mode == 0:
        return True

    else:
        assert False, 'Invalid sample mode'
