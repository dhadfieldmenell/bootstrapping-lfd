#! /bin/bash
## script to run several bootstrapping runs on a machine. assumes that directories and tasks.h5 files are already created
python scripts/iros_experiments.py data/actions.h5 data/boot_rot_const_8 --burn_in 50 --tree_sizes 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
python scripts/iros_experiments.py data/actions.h5 data/boot_rot_const_9 --burn_in 50 --tree_sizes 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
python scripts/iros_experiments.py data/actions.h5 data/boot_rot_const_no_cmat_9 --burn_in 50 --tree_sizes 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
python scripts/iros_experiments.py data/actions.h5 data/boot_rot_flow_3 --burn_in 50 --tree_sizes 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
