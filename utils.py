import numpy as np

def rect_to_bb(rect):
    # get x,y,w,h from the predicted bounging from dlib
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the numpy array with 68 tuple of x,y for facial landmarks from dlib
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and change them to a tuple of (x, y)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    return coords

