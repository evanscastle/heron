import cv2

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
