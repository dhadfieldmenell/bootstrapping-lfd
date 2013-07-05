from collections import defaultdict
import cv2, numpy as np
from rapprentice.plotting_plt import plot_warped_grid_2d
import rapprentice.registration as reg
from time import time
import os, os.path as osp
from glob import glob

def pixel_downsample(xy, s):
    xy = xy[np.isfinite(xy[:,0])]
    d = defaultdict(list)
    for (i,pt) in enumerate(xy):
        x,y = pt
        d[(int(x/s), int(y/s))].append(i)
    return np.array([xy[inds].mean(axis=0) for inds in d.values()])
def voxel_downsample(xyz, s):
    xyz = xyz[np.isfinite(xyz[:,0])]
    d = defaultdict(list)
    for (i,pt) in enumerate(xyz):
        x,y,z = pt
        d[(int(x/s), int(y/s), int(z/s))].append(i)
    return np.array([xyz[inds].mean(axis=0) for inds in d.values()])

SAMPLEDATA_DIR = osp.expanduser("~/Data/sampledata")
assert osp.exists(SAMPLEDATA_DIR)

fnames = glob(osp.join(SAMPLEDATA_DIR, "letter_images", "*.png"))

plotting = False

for fname in fnames:
    im = cv2.imread(fname, 0)
    assert im is not None
    xs, ys = np.nonzero(im==0)
    xys = np.c_[xs, ys]
    R = np.array([[.7, .4], [.3,.9]])
    T = np.array([[.3,.4]])
    fxys = (xys.dot(R) + T)**2 

    xys = np.c_[xys, np.zeros((len(xys)))]
    fxys = np.c_[fxys, np.zeros((len(fxys)))]

    
    scaled_xys, src_params = reg.unit_boxify(xys)
    scaled_fxys, targ_params = reg.unit_boxify(fxys)
    
    scaled_ds_xys = voxel_downsample(scaled_xys, .03)
    scaled_ds_fxys = voxel_downsample(scaled_fxys, .03)
    
    print "downsampled to %i and %i pts"%(len(scaled_ds_xys),len(scaled_ds_fxys))
    tstart = time()
    
    # fest_scaled = reg.tps_rpm(scaled_ds_xys, scaled_ds_fxys, n_iter=10, reg_init = 10, reg_final=.01)
    fest_scaled,_ = reg.tps_rpm_bij(scaled_ds_xys, scaled_ds_fxys, n_iter=10, reg_init = 10, reg_final=.01)
    
    
    
    print "time: %.4f"%(time()-tstart)
    fest = reg.unscale_tps(fest_scaled, src_params, targ_params)
    fxys_est = fest.transform_points(xys)
    print "error:", np.abs(fxys_est - fxys).mean()
    
    if plotting:
        import matplotlib.pyplot as plt
        plt.clf()
        # plt.plot(xys[:,1], xys[:,0],'r.')
        # plt.plot(fxys[:,1], fxys[:,0],'b.')
        # plt.plot(fxys_est[:,1], fxys_est[:,0],'g.')

        scaled_ds_fxys_est = fest_scaled.transform_points(scaled_ds_xys)
        plt.plot(scaled_ds_xys[:,1], scaled_ds_xys[:,0],'r.')
        plt.plot(scaled_ds_fxys[:,1], scaled_ds_fxys[:,0],'b.')
        plt.plot(scaled_ds_fxys_est[:,1], scaled_ds_fxys_est[:,0],'g.')


        # plot_warped_grid_2d(fest.transform_points, [0,0], [1,1])
        plt.draw()
        plt.ginput()
    
    
