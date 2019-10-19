from __future__ import division, absolute_import
import cv2
import os
import tensorflow as tf
import numpy as np
import sys
from flow_utils import read_intrinsics, read_pose, flow_to_color, compute_rigid_flow, write_flow
import argparse

parser = argparse.ArgumentParser(description='Evaluate rigid flow')
parser.add_argument('--pose',   type=str,   help='path to pose',  required=True)
parser.add_argument('--type',   type=str,   help='path to pose',  required=True)
parser.add_argument('--id',     type=int,   help='index of the frame couple', choices=[0,99], required=True)
args = parser.parse_args()


if __name__ == '__main__':
    disp_path = os.path.abspath(os.path.join('monodepth2', 'disp_' + str(args.id) +'.npy'))

    pose = read_pose(os.path.abspath(args.pose))[args.id]
    pose = np.concatenate([pose[3:], pose[:3]])

    with tf.Session() as session:
        intrinsics = read_intrinsics(os.path.abspath(os.path.join('calib',"calib.txt")))
        # scaling the intrinsics
        mask = tf.constant(np.asarray([[640/1226, 1, 640/1226], [192/370, 192/370 , 1], [1, 1, 1]], dtype=np.float32))
        
        intrinsics = intrinsics * mask

        pose = tf.convert_to_tensor(np.expand_dims(pose, 0), dtype=tf.float32)

        disp = np.load(disp_path)
        disp = np.expand_dims(disp,0)

        depth = tf.convert_to_tensor(1./(disp+1e-7), dtype=tf.float32)

        rigid_flow = compute_rigid_flow(depth, pose, intrinsics, reverse_pose=False)
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord=coordinator)

        rf = session.run(rigid_flow)
        write_flow(rf[0], os.path.join('outputs', "rigid_flow_" + args.type + "_" + str(args.id) +".flo"))
        coordinator.request_stop()
        coordinator.join(threads)