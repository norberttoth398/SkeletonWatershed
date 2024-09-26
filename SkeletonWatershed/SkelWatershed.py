from skimage.morphology import medial_axis
import skimage
from skimage.segmentation import watershed
import numpy as np
import skimage
import tqdm

def get_simple_min_pos(labs, distance, label):
    """_summary_

    Args:
        labs (_type_): _description_
        distance (_type_): _description_
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    temp_dist = distance.copy()
    temp_dist[labs != label] = 100*np.max(temp_dist)

    min_pos = np.argwhere(temp_dist == np.min(temp_dist))[0]
    return min_pos

def convolve3d(image, filter, padding = (1, 1)):
    """_summary_

    Args:
        image (_type_): _description_
        filter (_type_): _description_
        padding (tuple, optional): _description_. Defaults to (1, 1).

    Returns:
        _type_: _description_
    """
    # For this to work neatly, filter and image should have the same number of channels
    # Alternatively, filter could have just 1 channel or 2 dimensions
    # from https://stackoverflow.com/questions/63036809/how-do-i-use-only-numpy-to-apply-filters-onto-images
    
    print(filter.shape, image.shape)
    #assert image.shape[-1] == filter.shape[-1]
    size_x, size_y, size_z = filter.shape[:3]
    width, height, depth = image.shape[:3]
    
    output_array = np.zeros(((width - size_x + 2*padding[0]) + 1, 
                             (height - size_y + 2*padding[1]) + 1,
                             (depth - size_z + 2*padding[2]) + 1)) # Convolution Output: [(W−K+2P)/S]+1
    
    padded_image = np.pad(image, [
        (padding[0], padding[0]),
        (padding[1], padding[1]),
        (padding[2], padding[2]),
    ])
    print(padded_image.shape)
    
    m_k = padded_image.shape[0]*padded_image.shape[1]*padded_image.shape[2]
    
    for x in tqdm.tqdm(range(padded_image.shape[0] - size_x + 1)): # -size_x + 1 is to keep the window within the bounds of the image
        for y in range(padded_image.shape[1] - size_y + 1):
            for z in range(padded_image.shape[2] - size_z + 1):
                

                # Creates the window with the same size as the filter
                window = padded_image[x:x + size_x, y:y + size_y, z:z+size_z]

                # Sums over the product of the filter and the window
                output_values = np.sum(filter * window, axis=(0, 1, 2))
                #print(output_values)
                

                # Places the calculated value into the output_array
                output_array[x, y, z] = output_values
                
    return output_array



def convolve(image, filter, padding = (1, 1)):
    """_summary_

    Args:
        image (_type_): _description_
        filter (_type_): _description_
        padding (tuple, optional): _description_. Defaults to (1, 1).

    Returns:
        _type_: _description_
    """
    # For this to work neatly, filter and image should have the same number of channels
    # Alternatively, filter could have just 1 channel or 2 dimensions
    # from https://stackoverflow.com/questions/63036809/how-do-i-use-only-numpy-to-apply-filters-onto-images
    
    if(image.ndim == 2):
        image = np.expand_dims(image, axis=-1) # Convert 2D grayscale images to 3D
    if(filter.ndim == 2):
        filter = np.repeat(np.expand_dims(filter, axis=-1), image.shape[-1], axis=-1) # Same with filters
    if(filter.shape[-1] == 1):
        filter = np.repeat(filter, image.shape[-1], axis=-1) # Give filter the same channel count as the image
    
    #print(filter.shape, image.shape)
    assert image.shape[-1] == filter.shape[-1]
    size_x, size_y = filter.shape[:2]
    width, height = image.shape[:2]
    
    output_array = np.zeros(((width - size_x + 2*padding[0]) + 1, 
                             (height - size_y + 2*padding[1]) + 1,
                             image.shape[-1])) # Convolution Output: [(W−K+2P)/S]+1
    
    padded_image = np.pad(image, [
        (padding[0], padding[0]),
        (padding[1], padding[1]),
        (0, 0)
    ])
    
    for x in range(padded_image.shape[0] - size_x + 1): # -size_x + 1 is to keep the window within the bounds of the image
        for y in range(padded_image.shape[1] - size_y + 1):

            # Creates the window with the same size as the filter
            window = padded_image[x:x + size_x, y:y + size_y]

            # Sums over the product of the filter and the window
            output_values = np.sum(filter * window, axis=(0, 1)) 

            # Places the calculated value into the output_array
            output_array[x, y] = output_values
            
    return output_array

def skel_markers(img, medial= True):
    """_summary_

    Args:
        img (_type_): _description_
        medial (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    
    print("running 2d")
    #get initial skeletonization
    skel, distance = medial_axis(img, return_distance=True)
    if medial is False:
        from skimage.morphology import skeletonize
        skel = skeletonize(img)
        skel = skel
        print(np.unique(skel))
    
    #for finding number of neighbours - if more than 2 then it must be a node
    filter = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ],dtype=np.float32)
    neigh = convolve(skel, filter).reshape(skel.shape[0], skel.shape[1])
    neigh = np.multiply(neigh, skel.astype("float64"))
    
    #remove the nodes connecting the lines - or can think of it as points of skeleton intersection
    rem_vert = skel.astype("float64").copy()
    rem_vert[neigh > 2] = 0
    labs = skimage.morphology.label(rem_vert)
    
    
    reg = skimage.measure.regionprops(labs)
    lines = {}
    for item in reg:
        lines[item.label] = [item.area, np.min(distance[labs == item.label])]
        
    #get new copy - label lines with specific label if they are not channels - need to check they touch or not?
    skel_labs = skel.astype("float64").copy()
    non_channel_labs = []
    channel_labs = []
    for item in lines.keys():
        if lines[item][0] <= lines[item][1]: #if we do not do this, we get massive oversegmentation
            non_channel_labs.append(item)
        else:
            channel_labs.append(item)
            
    new_img = np.zeros_like(skel_labs)
    for item in non_channel_labs:
        new_img[labs==item] = 1
    new_img[neigh > 2] = 1
    
    #need to get lowest value in distance matrix to separate regions in branch
    #then merge those regions with new_img and then run label to find the distinct regions
    min_pos = []
    for item in channel_labs:
        temp_dist = distance.copy()
        temp_dist[labs != item] = 100*np.max(temp_dist)
        #print(np.argwhere(temp_dist == np.min(temp_dist)))
        
        min_pos.append(get_simple_min_pos(labs, distance, item))
        
    final_skel = skel.copy()
    for item in min_pos:
        final_skel[item[0], item[1]] = 0
        
    final_labs = skimage.measure.label(final_skel)
    return final_labs, distance

def skel_markers3d(img):
    """_summary_

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("running 3D...")
    import scipy.ndimage as ndi
    distance = ndi.distance_transform_edt(img)
    
    from skimage.morphology import skeletonize
    skel = skeletonize(img)
    skel = skel/255
    print(np.unique(skel))
    
    filter = np.array([[
        [1,1,1],
        [1,1,1],
        [1,1,1]],[
        [1,1,1],
        [1,0,1],
        [1,1,1]],[
        [1,1,1],
        [1,1,1],
        [1,1,1]]],dtype=np.float32)
    neigh = convolve3d(skel, filter).reshape(skel.shape[0], skel.shape[1], skel.shape[2])
    neigh = np.multiply(neigh, skel.astype("float64"))
    
    rem_vert = skel.astype("float64").copy()
    rem_vert[neigh > 2] = 0
    labs = skimage.morphology.label(rem_vert)
    
    reg = skimage.measure.regionprops(labs)
    lines = {}
    for item in reg:
        lines[item.label] = [item.area, np.max(distance[labs == item.label])]
        
    #get new copy - label lines with specific label if they are not channels - need to check they touch or not?
    skel_labs = skel.astype("float64").copy()
    non_channel_labs = []
    channel_labs = []
    for item in lines.keys():
        if lines[item][0] <= lines[item][1]:
            non_channel_labs.append(item)
        else:
            channel_labs.append(item)
            
    new_img = np.zeros_like(skel_labs)
    for item in non_channel_labs:
        new_img[labs==item] = 1
    new_img[neigh > 2] = 1
    
    #need to get lowest value in distance matrix to separate regions in branch
    #then merge those regions with new_img and then run label to find the distinct regions
    min_pos = []
    for item in channel_labs:
        temp_dist = distance.copy()
        temp_dist[labs != item] = 100*np.max(temp_dist)
        #print(np.argwhere(temp_dist == np.min(temp_dist)))
        
        min_pos.append(get_simple_min_pos(labs, distance, item))
        
    final_skel = skel.copy()
    for item in min_pos:
        final_skel[item[0], item[1]] = 0
        
    final_labs = skimage.measure.label(final_skel)
    return final_labs, distance
    
def skel_watershed(img, medial = True, threeD = False):
    """_summary_

    Args:
        img (_type_): _description_
        medial (bool, optional): _description_. Defaults to True.
        threeD (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    if threeD:
        markers, distance = skel_markers3d(img)
    else:
        markers, distance = skel_markers(img, medial)
    wlabs = watershed(-distance, markers, mask=img)
    
    return wlabs