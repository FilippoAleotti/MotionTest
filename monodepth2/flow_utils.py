from __future__ import division
import numpy as np
import tensorflow as tf

def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0., 0., 1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)

    return intrinsics

def get_multi_scale_intrinsics(intrinsics, num_scales):
    intrinsics_mscale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        intrinsics_mscale.append(
            make_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)

    return intrinsics_mscale

def read_intrinsics(intrinsics_file_path):
    cam_paths_queue = tf.train.string_input_producer([intrinsics_file_path], shuffle=False)
    cam_reader = tf.TextLineReader()
    _, raw_cam_contents = cam_reader.read(cam_paths_queue)
    rec_def = []
    for i in range(9):
        rec_def.append([1.])
    raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                record_defaults=rec_def)
    raw_cam_vec = tf.stack(raw_cam_vec)
    intrinsics = tf.reshape(raw_cam_vec, [3, 3])
    intrinsics = tf.expand_dims(intrinsics, axis=0)
    return intrinsics

def read_pose(txt_path):
    ''' 
        Read poses from txt file
        Each line contains 6 values in the following form:
        [[[ 0.00056317 -0.00072765 -0.00018442]]] [[[ 0.000106   -0.00054235 -0.00024723]]]
        First 3 values are T, last 3 are R

    '''
    poses = []
    with open(txt_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip().replace('[','')
        l = l.replace(']','')
        pose = np.fromstring(l, dtype=np.float32, sep=' ')
        poses.append(pose)
    return poses

def flow_to_color(flow, mask=None, max_flow=None):
    '''
    From Unflow by Meister et al
    https://arxiv.org/pdf/1711.07837.pdf
    https://github.com/simonmeister/UnFlow
    
    Converts flow to 3-channel color image.
    Args:
        flow: tensor of shape [num_batch, height, width, 2].
        mask: flow validity mask of shape [num_batch, height, width, 1].
    '''
    n = 8
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))
    mask = tf.ones([num_batch, height, width, 1]) if mask is None else mask
    flow_u, flow_v = tf.unstack(flow, axis=3)
    if max_flow is not None:
        max_flow = tf.maximum(max_flow, 1)
    else:
        max_flow = tf.reduce_max(tf.abs(flow * mask))
    mag = tf.sqrt(tf.reduce_sum(tf.square(flow), 3))
    angle = tf.atan2(flow_v, flow_u)

    im_h = tf.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = tf.clip_by_value(mag * n / max_flow, 0, 1)
    im_v = tf.clip_by_value(n - im_s, 0, 1)
    im_hsv = tf.stack([im_h, im_s, im_v], 3)
    im = tf.image.hsv_to_rgb(im_hsv)
    return im * mask

def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  with tf.variable_scope('euler2mat'):
    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones  = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat

def pose_vec2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  with tf.variable_scope('vec2mat'):
    batch_size, _ = vec.get_shape().as_list()
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.
  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  with tf.variable_scope('pixel2cam'):
    batch, height, width = depth.get_shape().as_list()
    depth = tf.reshape(depth, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
    if is_homogeneous:
      ones = tf.ones([batch, 1, height*width])
      cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.
  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  with tf.variable_scope('cam2pixel'):
    batch, _, height, width = cam_coords.get_shape().as_list()
    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.
  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords

def flow_warp(src_img, flow):
  """ inverse warp a source image to the target image plane based on flow field
  Args:
    src_img: the source  image [batch, height_s, width_s, 3]
    flow: target image to source image flow [batch, height_t, width_t, 2]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t, width_t, 3]
  """
  with tf.variable_scope('flow_warp'):
    batch, height, width, _ = src_img.get_shape().as_list()
    tgt_pixel_coords = tf.transpose(meshgrid(batch, height, width, False),
                      [0, 2, 3, 1])
    src_pixel_coords = tgt_pixel_coords + flow
    output_img = bilinear_sampler(src_img, src_pixel_coords)
    return output_img

def compute_rigid_flow(depth, pose, intrinsics, reverse_pose=False):
  """Compute the rigid flow from target image plane to source image
  Args:
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source (or source to target if reverse_pose=True)
          camera transformation matrix [batch, 6], in the order of
          tx, ty, tz, rx, ry, rz;
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Rigid flow from target image to source image [batch, height_t, width_t, 2]
  """
  with tf.variable_scope('compute_rigid_flow'):
    batch, height, width = depth.get_shape().as_list()
    # Convert pose vector to matrix
    pose = pose_vec2mat(pose)
    if reverse_pose:
      pose = tf.matrix_inverse(pose)
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    tgt_pixel_coords = tf.transpose(pixel_coords[:,:2,:,:], [0, 2, 3, 1])
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    rigid_flow = src_pixel_coords - tgt_pixel_coords
    return rigid_flow

def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.
  Points falling outside the source image boundary have value 0.
  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    #zero = tf.zeros([1], dtype='float32')
    zero = tf.constant(0, dtype=tf.float32)

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output

def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()
  
def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print "Reading %d x %d flow file in .flo format" % (h, w)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    # print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel
  
def get_EPE(flow):
  h,w = flow.shape[:2]
  return np.linalg.norm(flow, axis=-1).sum() / (h*w)