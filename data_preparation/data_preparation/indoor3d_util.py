import numpy as np
import glob
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

DATA_PATH = os.path.join(ROOT_DIR, 'data_preparation', 'Stanford3dDataset_v1.2_Aligned_Version')
g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/class_names.txt'))]
g_class2label = {cls: i for i,cls in enumerate(g_classes)}
g_class2color = {'T2T':	[255,255,255],     # white
                 'B2B':	[255,128,0],   # brown
                 'BH':	[255,0,255],   # magenta
                 'BL':	[0,255,255],   # cyan
                 'V2V':	[0,255,0],   # green
                 'OT':   [0,0,255]}  # blue
g_easy_view_labels = [11,12]
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}


# -----------------------------------------------------------------------------
# CONVERT ORIGINAL DATA TO OUR DATA_LABEL FILES
# -----------------------------------------------------------------------------

def collect_point_label(anno_path, out_filename, file_format='txt'):
    """ Convert TXT data (from annotation folder) and convert it to .npy (numpy)
        No loss of points! and merge Data&Labels, no change.
        We aggregated all the points from each instance in the room.
        Output: N*4 (XYZL); or N*7 (XYZRGBL)

    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    points_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]

        if cls not in g_classes: # note: in some room there is 'staris' class..
            cls = 'clutter'

        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * g_class2label[cls]

        points_list.append(np.concatenate([points, labels], 1))  # After: Nx(C+1),  raw TXT & Lb are merged

        # if points.shape[1]==3:   # Input is XYZ,without RGB
        #     rgb3col =  np.zeros((points.shape[0],3))   # ZZC, + RGB
        #     points_list.append(np.concatenate([points, rgb3col, labels], 1))   # After: Nx7
        # if points.shape[1]==6:   # Input is XYZRGB
        #     points_list.append(np.concatenate([points, labels], 1))     # After: Nx7

    data_label = np.concatenate(points_list, 0)
    print(data_label.shape)
    # xy_min = np.amin(data_label, axis=0)[0:2]  # minimum xy in this office/conference/WC room
    # data_label[:, 0:2] -= xy_min  # shift (X,Y) to min_coord (commented by ZZC) [the most negative point is now at origin.]
    
    if file_format=='txt':     # No!
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % \
                          (data_label[i,0], data_label[i,1], data_label[i,2],
                           data_label[i,3], data_label[i,4], data_label[i,5],
                           data_label[i,6]))
        fout.close()
    elif file_format=='numpy':   # Yes!
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
            (file_format))
        exit()

def point_label_to_obj(input_filename, out_filename, label_color=True, easy_view=False, no_wall=False):
    """ For visualization of a room from data_label file,
	input_filename: each line is X Y Z R G B L
	out_filename: OBJ filename,
        visualize input file by coloring point with label color
        easy_view: only visualize furnitures and floor
        label_color: label_color OR true_color
    """
    data_label = np.loadtxt(input_filename)
    data = data_label[:, 0:6]
    label = data_label[:, -1].astype(int)
    fout = open(out_filename, 'w')
    for i in range(data.shape[0]):
        color = g_label2color[label[i]]
        if easy_view and (label[i] not in g_easy_view_labels):
            continue
        if no_wall and ((label[i] == 2) or (label[i]==0)):
            continue
        if label_color:
            fout.write('v %f %f %f %d %d %d\n' % \
                (data[i,0], data[i,1], data[i,2], color[0], color[1], color[2]))
        else:
            fout.write('v %f %f %f %d %d %d\n' % \
                (data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5]))
    fout.close()


# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]
        # print('range(N)', range(N))
        # print('list(sample): ', list(sample))
        return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)


def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label


def room2blocks(data, label, num_point, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1):
    """ Prepare block training data.
    Args:
        data: N x 9 numpy array, 012 are XYZ in meters, 345 are zeros , 678 are zeros or rgb (0~1)
            Note that the data (X, Y) have been shifted (min point is origin) and aligned
        label: N size uint8 numpy array from 0-3

        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug

    Returns:
        block_datas: K x num_point x 9 np array of XYZxyzrgb
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # Normalize X,Y,Z
    min_room_x = 91260.0
    min_room_y = 438850.0
    data[:, 0] -= min_room_x
    data[:, 1] -= min_room_y

    xyz_raw = data  # make a copy

    # Get blocks
    limit_min = np.amin(data, 0)[0:2]  # min value of every column
    data[:, 0] -= limit_min[0]
    data[:, 1] -= limit_min[1]
    limit = np.amax(data, 0)[0:2]     # max value of every column

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * stride)
                ybeg_list.append(j * stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)):  # Iterate over each block
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < 100:  # discard block if there are less than 100 pts.  # ZZC
            continue

        block_data = data[cond, :]
        block_label = label[cond]
        block_xyz_raw = xyz_raw[cond, :]

        # Normalize (x,y) in this block
        block_data[:, 0] -= xbeg
        block_data[:, 1] -= ybeg
        z_min_block = np.amin(block_data[:, 2])  # min Z in the block
        block_data[:, 2] -= z_min_block

        new_block_data = np.zeros((block_label.shape[0], 9))

        new_block_data[:, 0:3] = block_xyz_raw[:, 0:3]
        new_block_data[:, 3:6] = block_data[:, 0:3]
        new_block_data[:, 6:9] = block_xyz_raw[:, 6:9]

        # randomly subsample data
        block_data_sampled, block_label_sampled = sample_data_label(new_block_data, block_label, num_point)
        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))

    return np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)


# def room2blocks_plus(data_label, num_point, block_size, stride,
#                      random_sample, sample_num, sample_aug):
#     """ room2block with input filename and RGB preprocessing.
#     """
#     data = data_label[:, 0:6]
#     data[:, 3:6] /= 255.0
#     label = data_label[:, -1].astype(np.uint8)
#
#     return room2blocks(data, label, num_point, block_size, stride,
#                        random_sample, sample_num, sample_aug)


# def room2blocks_wrapper(data_label_filename, num_point, block_size=1.0, stride=1.0,
#                         random_sample=False, sample_num=None, sample_aug=1):
#     if data_label_filename[-3:] == 'txt':
#         data_label = np.loadtxt(data_label_filename)
#     elif data_label_filename[-3:] == 'npy':
#         data_label = np.load(data_label_filename)
#     else:
#         print('Unknown file type! exiting.')
#         exit()
#     return room2blocks_plus(data_label, num_point, block_size, stride,
#                             random_sample, sample_num, sample_aug)


def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """

    # Step 1: Download the data and re-save them
    # data = np.zeros(data_label.shape[0], 9)   # To keep input data

    data = np.zeros((data_label.shape[0], 9))

    if data_label.shape[1] == 7:    # (XYZRGBL)
        data[:, 0:3] = data_label[:, 0:3]     # raw XYZ
        data[:, 6:9] = data_label[:, 3:6]     # raw RGB
    if data_label.shape[1] == 4:    # (XYZL)
        data[:, 0:3] = data_label[:, 0:3]  # raw XYZ

    data[:, 6:9] /= 255.0     #  normalize RGB, only ONCE!
    label = data_label[:, -1].astype(np.uint8)

    # Step 2: convert to blocks
    data_batch, label_batch = room2blocks(data, label, num_point, block_size, stride,
                                          random_sample, sample_num, sample_aug)
    # data_batch [B,4096,9] (XYZxyzrgb)
    # label_batch [B,4096]

    return data_batch, label_batch


def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0,
                                   random_sample=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                       random_sample, sample_num, sample_aug)


# def room2samples(data, label, sample_num_point):
#     """ Prepare whole room samples.
#
#     Args:
#         data: N x 3 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
#             assumes the data is shifted (min point is origin) and
#             aligned (aligned with XYZ axis)
#         label: N size uint8 numpy array from 0-12
#         sample_num_point: int, how many points to sample in each sample
#     Returns:
#         sample_datas: K x sample_num_point x 9
#                      numpy array of XYZRGBX'Y'Z', RGB is in [0,1]
#         sample_labels: K x sample_num_point x 1 np array of uint8 labels
#     """
#     N = data.shape[0]
#     order = np.arange(N)
#     np.random.shuffle(order)
#     data = data[order, :]
#     label = label[order]
#
#     batch_num = int(np.ceil(N / float(sample_num_point)))
#     sample_datas = np.zeros((batch_num, sample_num_point, 6))
#     sample_labels = np.zeros((batch_num, sample_num_point, 1))
#
#     for i in range(batch_num):
#         beg_idx = i * sample_num_point
#         end_idx = min((i + 1) * sample_num_point, N)
#         num = end_idx - beg_idx
#         sample_datas[i, 0:num, :] = data[beg_idx:end_idx, :]
#         sample_labels[i, 0:num, 0] = label[beg_idx:end_idx]
#         if num < sample_num_point:
#             makeup_indices = np.random.choice(N, sample_num_point - num)
#             sample_datas[i, num:, :] = data[makeup_indices, :]
#             sample_labels[i, num:, 0] = label[makeup_indices]
#     return sample_datas, sample_labels
#
#
# def room2samples_plus_normalized(data_label, num_point):
#     """ room2sample, with input filename and RGB preprocessing.
#         for each block centralize XYZ, add normalized XYZ as 678 channels
#     """
#     data = data_label[:, 0:3]
#     # data[:, 3:6] /= 255.0
#     label = data_label[:, -1].astype(np.uint8)
#     min_room_x = 91260.0
#     min_room_y = 438850.0
#     min_room_z = -5.0
#     # print(max_room_x, max_room_y, max_room_z)
#
#     data_batch, label_batch = room2samples(data, label, num_point)
#
#     new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
#     for b in range(data_batch.shape[0]):
#         new_data_batch[b, :, 6] = data_batch[b, :, 0] / min_room_x
#         new_data_batch[b, :, 7] = data_batch[b, :, 1] / min_room_y
#         new_data_batch[b, :, 8] = data_batch[b, :, 2] / min_room_z
#         # minx = min(data_batch[b, :, 0])
#         # miny = min(data_batch[b, :, 1])
#         # data_batch[b, :, 0] -= (minx+block_size/2)
#         # data_batch[b, :, 1] -= (miny+block_size/2)
#     new_data_batch[:, :, 0:6] = data_batch
#     return new_data_batch, label_batch
#
#
# def room2samples_wrapper_normalized(data_label_filename, num_point):
#     if data_label_filename[-3:] == 'txt':
#         data_label = np.loadtxt(data_label_filename)
#     elif data_label_filename[-3:] == 'npy':
#         data_label = np.load(data_label_filename)
#     else:
#         print('Unknown file type! exiting.')
#         exit()
#     return room2samples_plus_normalized(data_label, num_point)


# -----------------------------------------------------------------------------
# EXTRACT INSTANCE BBOX FROM ORIGINAL DATA (for detection evaluation)
# -----------------------------------------------------------------------------

def collect_bounding_box(anno_path, out_filename):
    """ Compute bounding boxes from each instance in original dataset files on
        one room. **We assume the bbox is aligned with XYZ coordinate.**

    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save instance bounding boxes for that room.
            each line is x1 y1 z1 x2 y2 z2 label,
            where (x1,y1,z1) is the point on the diagonal closer to origin
    Returns:
        None
    Note:
        room points are shifted, the most negative point is now at origin.
    """
    bbox_label_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes:  # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)
        label = g_class2label[cls]
        # Compute tightest axis aligned bounding box
        xyz_min = np.amin(points[:, 0:3], axis=0)
        xyz_max = np.amax(points[:, 0:3], axis=0)
        ins_bbox_label = np.expand_dims(
            np.concatenate([xyz_min, xyz_max, np.array([label])], 0), 0)
        bbox_label_list.append(ins_bbox_label)

    bbox_label = np.concatenate(bbox_label_list, 0)
    room_xyz_min = np.amin(bbox_label[:, 0:3], axis=0)
    bbox_label[:, 0:3] -= room_xyz_min
    bbox_label[:, 3:6] -= room_xyz_min

    fout = open(out_filename, 'w')
    for i in range(bbox_label.shape[0]):
        fout.write('%f %f %f %f %f %f %d\n' % \
                   (bbox_label[i, 0], bbox_label[i, 1], bbox_label[i, 2],
                    bbox_label[i, 3], bbox_label[i, 4], bbox_label[i, 5],
                    bbox_label[i, 6]))
    fout.close()


def bbox_label_to_obj(input_filename, out_filename_prefix, easy_view=False):
    """ Visualization of bounding boxes.

    Args:
        input_filename: each line is x1 y1 z1 x2 y2 z2 label
        out_filename_prefix: OBJ filename prefix,
            visualize object by g_label2color
        easy_view: if True, only visualize furniture and floor
    Returns:
        output a list of OBJ file and MTL files with the same prefix
    """
    bbox_label = np.loadtxt(input_filename)
    bbox = bbox_label[:, 0:6]
    label = bbox_label[:, -1].astype(int)
    v_cnt = 0  # count vertex
    ins_cnt = 0  # count instance
    for i in range(bbox.shape[0]):
        if easy_view and (label[i] not in g_easy_view_labels):
            continue
        obj_filename = out_filename_prefix + '_' + g_classes[label[i]] + '_' + str(ins_cnt) + '.obj'
        mtl_filename = out_filename_prefix + '_' + g_classes[label[i]] + '_' + str(ins_cnt) + '.mtl'
        fout_obj = open(obj_filename, 'w')
        fout_mtl = open(mtl_filename, 'w')
        fout_obj.write('mtllib %s\n' % (os.path.basename(mtl_filename)))

        length = bbox[i, 3:6] - bbox[i, 0:3]
        a = length[0]
        b = length[1]
        c = length[2]
        x = bbox[i, 0]
        y = bbox[i, 1]
        z = bbox[i, 2]
        color = np.array(g_label2color[label[i]], dtype=float) / 255.0

        material = 'material%d' % (ins_cnt)
        fout_obj.write('usemtl %s\n' % (material))
        fout_obj.write('v %f %f %f\n' % (x, y, z + c))
        fout_obj.write('v %f %f %f\n' % (x, y + b, z + c))
        fout_obj.write('v %f %f %f\n' % (x + a, y + b, z + c))
        fout_obj.write('v %f %f %f\n' % (x + a, y, z + c))
        fout_obj.write('v %f %f %f\n' % (x, y, z))
        fout_obj.write('v %f %f %f\n' % (x, y + b, z))
        fout_obj.write('v %f %f %f\n' % (x + a, y + b, z))
        fout_obj.write('v %f %f %f\n' % (x + a, y, z))
        fout_obj.write('g default\n')
        v_cnt = 0  # for individual box
        fout_obj.write('f %d %d %d %d\n' % (4 + v_cnt, 3 + v_cnt, 2 + v_cnt, 1 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (1 + v_cnt, 2 + v_cnt, 6 + v_cnt, 5 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (7 + v_cnt, 6 + v_cnt, 2 + v_cnt, 3 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (4 + v_cnt, 8 + v_cnt, 7 + v_cnt, 3 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5 + v_cnt, 8 + v_cnt, 4 + v_cnt, 1 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5 + v_cnt, 6 + v_cnt, 7 + v_cnt, 8 + v_cnt))
        fout_obj.write('\n')

        fout_mtl.write('newmtl %s\n' % (material))
        fout_mtl.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout_mtl.write('\n')
        fout_obj.close()
        fout_mtl.close()

        v_cnt += 8
        ins_cnt += 1


def bbox_label_to_obj_room(input_filename, out_filename_prefix, easy_view=False, permute=None, center=False,
                           exclude_table=False):
    """ Visualization of bounding boxes.

    Args:
        input_filename: each line is x1 y1 z1 x2 y2 z2 label
        out_filename_prefix: OBJ filename prefix,
            visualize object by g_label2color
        easy_view: if True, only visualize furniture and floor
        permute: if not None, permute XYZ for rendering, e.g. [0 2 1]
        center: if True, move obj to have zero origin
    Returns:
        output a list of OBJ file and MTL files with the same prefix
    """
    bbox_label = np.loadtxt(input_filename)
    bbox = bbox_label[:, 0:6]
    if permute is not None:
        assert (len(permute) == 3)
        permute = np.array(permute)
        bbox[:, 0:3] = bbox[:, permute]
        bbox[:, 3:6] = bbox[:, permute + 3]
    if center:
        xyz_max = np.amax(bbox[:, 3:6], 0)
        bbox[:, 0:3] -= (xyz_max / 2.0)
        bbox[:, 3:6] -= (xyz_max / 2.0)
        bbox /= np.max(xyz_max / 2.0)
    label = bbox_label[:, -1].astype(int)
    obj_filename = out_filename_prefix + '.obj'
    mtl_filename = out_filename_prefix + '.mtl'

    fout_obj = open(obj_filename, 'w')
    fout_mtl = open(mtl_filename, 'w')
    fout_obj.write('mtllib %s\n' % (os.path.basename(mtl_filename)))
    v_cnt = 0  # count vertex
    ins_cnt = 0  # count instance
    for i in range(bbox.shape[0]):
        if easy_view and (label[i] not in g_easy_view_labels):
            continue
        if exclude_table and label[i] == g_classes.index('table'):
            continue

        length = bbox[i, 3:6] - bbox[i, 0:3]
        a = length[0]
        b = length[1]
        c = length[2]
        x = bbox[i, 0]
        y = bbox[i, 1]
        z = bbox[i, 2]
        color = np.array(g_label2color[label[i]], dtype=float) / 255.0

        material = 'material%d' % (ins_cnt)
        fout_obj.write('usemtl %s\n' % (material))
        fout_obj.write('v %f %f %f\n' % (x, y, z + c))
        fout_obj.write('v %f %f %f\n' % (x, y + b, z + c))
        fout_obj.write('v %f %f %f\n' % (x + a, y + b, z + c))
        fout_obj.write('v %f %f %f\n' % (x + a, y, z + c))
        fout_obj.write('v %f %f %f\n' % (x, y, z))
        fout_obj.write('v %f %f %f\n' % (x, y + b, z))
        fout_obj.write('v %f %f %f\n' % (x + a, y + b, z))
        fout_obj.write('v %f %f %f\n' % (x + a, y, z))
        fout_obj.write('g default\n')
        fout_obj.write('f %d %d %d %d\n' % (4 + v_cnt, 3 + v_cnt, 2 + v_cnt, 1 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (1 + v_cnt, 2 + v_cnt, 6 + v_cnt, 5 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (7 + v_cnt, 6 + v_cnt, 2 + v_cnt, 3 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (4 + v_cnt, 8 + v_cnt, 7 + v_cnt, 3 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5 + v_cnt, 8 + v_cnt, 4 + v_cnt, 1 + v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5 + v_cnt, 6 + v_cnt, 7 + v_cnt, 8 + v_cnt))
        fout_obj.write('\n')

        fout_mtl.write('newmtl %s\n' % (material))
        fout_mtl.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout_mtl.write('\n')

        v_cnt += 8
        ins_cnt += 1

    fout_obj.close()
    fout_mtl.close()


def collect_point_bounding_box(anno_path, out_filename, file_format):
    """ Compute bounding boxes from each instance in original dataset files on
        one room. **We assume the bbox is aligned with XYZ coordinate.**
        Save both the point XYZRGB and the bounding box for the point's
        parent element.

    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save instance bounding boxes for each point,
            plus the point's XYZRGBL
            each line is XYZRGBL offsetX offsetY offsetZ a b c,
            where cx = X+offsetX, cy=X+offsetY, cz=Z+offsetZ
            where (cx,cy,cz) is center of the box, a,b,c are distances from center
            to the surfaces of the box, i.e. x1 = cx-a, x2 = cx+a, y1=cy-b etc.
        file_format: output file format, txt or numpy
    Returns:
        None

    Note:
        room points are shifted, the most negative point is now at origin.
    """
    point_bbox_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes:  # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)  # Nx6
        label = g_class2label[cls]  # N,
        # Compute tightest axis aligned bounding box
        xyz_min = np.amin(points[:, 0:3], axis=0)  # 3,
        xyz_max = np.amax(points[:, 0:3], axis=0)  # 3,
        xyz_center = (xyz_min + xyz_max) / 2
        dimension = (xyz_max - xyz_min) / 2

        xyz_offsets = xyz_center - points[:, 0:3]  # Nx3
        dimensions = np.ones((points.shape[0], 3)) * dimension  # Nx3
        labels = np.ones((points.shape[0], 1)) * label  # N
        point_bbox_list.append(np.concatenate([points, labels,
                                               xyz_offsets, dimensions], 1))  # Nx13

    point_bbox = np.concatenate(point_bbox_list, 0)  # KxNx13
    room_xyz_min = np.amin(point_bbox[:, 0:3], axis=0)
    point_bbox[:, 0:3] -= room_xyz_min

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(point_bbox.shape[0]):
            fout.write('%f %f %f %d %d %d %d %f %f %f %f %f %f\n' % \
                       (point_bbox[i, 0], point_bbox[i, 1], point_bbox[i, 2],
                        point_bbox[i, 3], point_bbox[i, 4], point_bbox[i, 5],
                        point_bbox[i, 6],
                        point_bbox[i, 7], point_bbox[i, 8], point_bbox[i, 9],
                        point_bbox[i, 10], point_bbox[i, 11], point_bbox[i, 12]))

        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, point_bbox)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
              (file_format))
        exit()

