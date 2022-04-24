import cv2
import os
import pickle

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


# Using state information, generate an image to be either displayed or returned
def generate_state_img(state, winname=None, scale=1, display=False, ret=False):
    
    width = int(state.shape[1] * scale)
    height = int(state.shape[0] * scale)
    dim = (width, height)

    img = cv2.resize(state, dim, interpolation = cv2.INTER_AREA)
    
    if display:
        if winname is None:
            winname = 'agent'
        cv2.namedWindow(winname) 
        cv2.moveWindow(winname, 40,30)
        cv2.imshow(winname, img)
        cv2.waitKey(0)

    if ret:
        return img


# Create an image of the transition for visualization purposes. Uses same args as generate_state_img
def generate_transition_img(transition, winname=None, scale=1, show_action=False, display=False, ret=False):
    orig = generate_state_img(transition[0], winname=winname, scale=scale, ret=True)
    new = generate_state_img(transition[1], winname=winname, scale=scale, ret=True)
    
    img = cv2.hconcat([orig, new])

    if display:
        if winname is None:
            winname = f'action: {transition[2]}'
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 40,30)
        cv2.imshow(winname, img)
        cv2.waitKey(0)

    if ret:
        return img
