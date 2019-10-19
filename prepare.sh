#!/bin/bash
rm -r outputs
mkdir outputs
python monodepth2/prepare.py --pose monodepth2/real_poses.txt --type real --id 0
python monodepth2/prepare.py --pose monodepth2/fake_poses.txt --type fake --id 0
python monodepth2/prepare.py --pose monodepth2/real_poses.txt --type real --id 99
python monodepth2/prepare.py --pose monodepth2/fake_poses.txt --type fake --id 99
