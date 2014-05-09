import numpy as np
import cv2, argparse, h5py
import os.path as osp
from pdb import pm, set_trace
import IPython as ipy
from rapprentice import berkeley_pr2, clouds, cloud_proc_funcs
import scipy.spatial.distance as ssd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

#from hd_utils.defaults import demo_files_dir
demo_files_dir = '../data'
resample_size = 72

usage = """
To view and label all demos of a certain task type:
python label_crossings.py --demo_type=DEMO_TYPE
"""
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("--demo_type", type=str)
parser.add_argument("--single_demo", help="View and label a single demo", default=False, type=str)
parser.add_argument("--demo_name", help="Name of demo if single demo.", default="", type=str)
parser.add_argument("--clear", help="Remove crossings info for the given demo_type", default=None, type=str)
parser.add_argument("--label_ends", help="Label the ends of the rope as start and finish for traversal", default=False, type=bool)
parser.add_argument("--label_points", help="Label arbitrary points along the rope in order of traversal", default=False, type=bool)
parser.add_argument("--verify", help="Check existence & content of crossings datasets for the given demo_type.", default=False, type=bool)



"""
Mark the requested point with a circle and store the location in the 
labeled_crossings array passed as param[1].
"""
def mark2(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP: #left-click, normal
        print (x, y)
        cv2.circle(param[0],(x,y),5,(0,200,200),-1)
        if param[1] != []:
            oldy, oldx = tuple(param[1][-1][:2]); pt2 = (x,y)
            pt1 = (oldx, oldy)
            cv2.line(param[0], pt1, pt2, (1,0,0))
        param[1].append([y,x,0])
    elif event == cv2.EVENT_RBUTTONUP: #right-click, undercrossing
        cv2.circle(param[0],(x,y),5,(200,170,50),-1)
        if param[1] != []:
            pt1 = tuple(param[1][-1][:2]); pt2 = (x,y)
            cv2.line(param[0], pt1, pt2, (1,0,0))
        param[1].append([x,y,-1])
    elif event == cv2.EVENT_MBUTTONUP: #middle-click, overcrossing
        cv2.circle(param[0],(x,y),5,(200,20,20),-1)
        if param[1] != []:
            pt1 = tuple(param[1][-1][:2]); pt2 = (x,y)
            cv2.line(param[0], pt1, pt2, (1,0,0))
        param[1].append([x,y,1])

def get_mask(rgb, depth, T_w_k):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    red_mask = ((h<10) | (h>150)) & (s > 100) & (v > 100)
    
    valid_mask = depth > 0
    
    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]
    
    z = xyz_w[:,:,2]   
    z0 = xyz_k[:,:,2]
    # if DEBUG_PLOTS:
    #     cv2.imshow("z0",z0/z0.max())
    #     cv2.imshow("z",z/z.max())
    #     cv2.imshow("rgb", rgb)
    #     cv2.waitKey()
    
    height_mask = xyz_w[:,:,2] > .7 # TODO pass in parameter
    
    
    good_mask = red_mask & height_mask & valid_mask
    return good_mask

"""
Label arbitrary points in the demo
"""
def label_points(hdf, demo_name):
    # set_trace()
    if demo_name != "":
        print "labeling single demo"
        # import IPython;IPython.embed()
        seg_group = hdf[demo_name]
        if "labeled_points" in seg_group.keys():
            del seg_group["labeled_points"]
        ret = label_single_demo_points(seg_group, demo_name)
        while ret == "retry":
            ret = label_single_demo_points(seg_group, demo_name)
        if ret == "quit":
            return
    else:
        for demo in hdf.keys():
            seg_group = hdf[demo]
            if "labeled_points" in seg_group.keys():
                del seg_group["labeled_points"]
            ret = label_single_demo_points(seg_group, demo)
            while ret == "retry":
                ret = label_single_demo_points(seg_group, demo)
            if ret == "quit":
                return

"""
Setup window, draw points, etc.
"""
def label_single_demo_points(seg_group, name):
    if 'old_cloud_xyz' not in seg_group:
        seg_group['old_cloud_xyz'] = seg_group['cloud_xyz'][:]
    print name
    rgb = seg_group['rgb'][:]
    old_cloud_xyz = seg_group['old_cloud_xyz'][:]
    depth = seg_group['depth'][:]
    T_w_k = seg_group['T_w_k'][:]
    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    xyz_w = xyz_k.dot(T_w_k[:3, :3].T) + T_w_k[:3, 3][None, None, :]
    old_cloud_kd = KDTree(old_cloud_xyz)

    image = np.asarray(rgb)
    windowName = name
    ctr = 0

    finished = False
    labeled_points = []
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, mark2, (image, labeled_points))
    while(1):
        cv2.imshow(windowName,image)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'):
            return "retry"
        elif k == ord(' '):
            break
        elif k == ord('p'):
            return "restart"
        elif k == 27:
            return "quit"
    labeled_points = np.asarray(labeled_points)
    if 'labeled_points' in seg_group:
        del seg_group['labeled_points']
    seg_group.create_dataset("labeled_points",shape=(len(labeled_points),3),data=labeled_points)
    # resample the line
    labeled_xyz = []
    for i in range(labeled_points.shape[0]):
        pt_i = labeled_points[i, :2]
        x, y, z = xyz_k[pt_i[0], pt_i[1], :].dot(T_w_k[:3, :3].T) + T_w_k[:3, 3]
        if depth[pt_i[0], pt_i[1]] > 0 and z > .7:
            labeled_xyz.append([x, y, z])
    labeled_xyz = np.asarray(labeled_xyz)
    cloud_xyz = np.zeros((resample_size, 3))
    rope_len = line_length(labeled_xyz)
    for i, t in enumerate(np.linspace(0, rope_len, resample_size)):
        cloud_xyz[i, :] = point_at(labeled_xyz, t)
    plt.scatter(old_cloud_xyz[:, 0], old_cloud_xyz[:, 1], color='r')
    plt.scatter(cloud_xyz[:, 0], cloud_xyz[:, 1], color='b')
    plt.show(block=False)
    k = cv2.waitKey(0) & 0xFF
    plt.close()
    if k == ord('r'):
        return "retry"
    if 'cloud_xyz' in seg_group:
        del seg_group['cloud_xyz']
    seg_group['cloud_xyz'] = cloud_xyz


def line_length(pts):
    """
    @returns : length of the piecewise linear line specified by pts
    """
    return np.trace(ssd.cdist(pts[:-1, :], pts[1:, :]))

def point_at(pts, dist):
    """
    @returns : point at dist into the piecewise linear line specified by pts
    """
    distances = np.r_[0, np.cumsum(np.diag(ssd.cdist(pts[:-1, :], pts[1:, :])))]
    if dist == distances[0]:
        return pts[0, :]
    if dist == distances[-1]:
        return pts[-1, :]
    bigger_mask = distances > dist
    smaller_mask = distances < dist
    start_pt = pts[smaller_mask, :][-1, :]
    end_pt = pts[bigger_mask, :][0, :]
    local_dist = dist - distances[smaller_mask][-1]
    slope  = (end_pt - start_pt)/np.linalg.norm(end_pt - start_pt)
    new_pt = slope*local_dist + start_pt
    print 'new_pt {} start_pt {} end_pt {}'.format(new_pt, start_pt, end_pt)
    return new_pt 

if __name__ == "__main__":
    args = parser.parse_args()
    print args
    demo_type = args.demo_type
    clear = args.clear
    should_label_points = args.label_points
    should_label_ends = args.label_ends
    demo_name = args.demo_name
    verify = args.verify
    h5filename = osp.join(demo_files_dir, demo_type + '.h5')
    hdf = h5py.File(h5filename, 'r+')
    #refactor(hdf)

    if clear != None:
        confirm = raw_input("Really delete all "+ clear +" info for "+demo_type+"? (y/n):")
        if confirm == 'y':
            print "clearing", clear, "data for", demo_type
            remove_data(hdf, clear)
        else:
            print "Canceled."
    elif should_label_points:
        label_points(hdf, demo_name)
    elif should_label_ends:
        print "labeling ends"
        ret = label_ends(hdf, demo_name)
        while ret == "restart":
            ret = label_ends(hdf, demo_name)
    elif verify:
        print "verifying"
        verify_crossings(hdf)
    else:
        print "labeling crossings"
        label_crossings(hdf, demo_name)
