from SkeletonWatershed import skel_watershed

def test_skel2d():
    import numpy as np
    import matplotlib.pyplot as plt
    img = np.zeros((100,100))
    img[1:20, 20:42] = 1
    img[15:60, 40:80] = 1
    img[55:96, 75:85]=1
    labs = skel_watershed(img, True)

def test_skel3d():
    import numpy as np
    import matplotlib.pyplot as plt
    img = np.zeros((100,100,100))
    img[1:20, 20:42, 20:40] = 1
    img[15:60, 40:80, 30:50] = 1
    img[55:96, 75:85, 10:20]=1
    labs = skel_watershed(img, True, True)
