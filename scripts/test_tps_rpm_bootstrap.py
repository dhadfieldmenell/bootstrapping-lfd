#!/usr/bin/env python
###################
"""
Workflow:
1. Fake data + animation only
    --fake_data_segment=xxx --execution=0
2. Fake data + Gazebo. Set Gazebo to initial state of fake data segment so we'll execute the same thing.
    --fake_data_segment=xxx --execution=1
    This is just so we know the robot won't do something stupid that we didn't catch with openrave only mode.
3. Real data + Gazebo
    --execution=1
    The problem is that the gazebo robot is in a different state from the real robot, in particular, the head tilt
    angle. TODO: write a script that       sets gazebo head to real robot head
4. Real data + Real execution.
    --execution=1

The question is, do you update the robot's head transform.
If you're using fake data, don't update it.

"""
from rapprentice import registration, colorize, \
    animate_traj, ros2rave, plotting_openrave, task_execution, \
    planning, func_utils, resampling, ropesim, rope_initialization, clouds
from rapprentice import math_utils as mu

import trajoptpy
import openravepy
import numpy as np
import h5py
from numpy import asarray
import atexit
import time
import sys
import random
import copy
import dhm_utils as dhm_u
import IPython as ipy
from pdb import pm, set_trace

#Don't use args, use globals
#args = None

class Globals:
    robot = None
    env = None
    pr2 = None
    sim = None
    exec_log = None
    viewer = None
    random_seed = None


class RopeState:
    def __init__(self, segment, perturb_radius, perturb_num_points):
        self.segment = segment
        self.perturb_radius = perturb_radius
        self.perturb_num_points = perturb_num_points


class TaskParameters:
    def __init__(self, demofile_name, knot, animate, max_steps_before_failure, choose_segment, log_name):
        self.demofile_name = demofile_name
        self.knot = knot
        self.animate = animate
        self.max_steps_before_failure = max_steps_before_failure
        self.choose_segment = choose_segment
        self.log_name = log_name
        self.random_seed = None
        self.add_to_hdf5 = False
        #only_original_segments, if true, will only register with the original segments
        self.only_original_segments = False


#init_rope_state_segment, perturb_radius, perturb_num_points
def redprint(msg):
    """Print the message to the console in red, bold font."""
    print colorize.colorize(msg, "red", bold=True)


def split_trajectory_by_gripper(seg_info):
    """Split up the trajectory into sections with breaks occuring when the grippers open or close.

    Return: (seg_starts, seg_ends)

    """
    rgrip = asarray(seg_info["r_gripper_joint"])
    lgrip = asarray(seg_info["l_gripper_joint"])

    thresh = .04  # open/close threshold

    n_steps = len(lgrip)

    # indices BEFORE transition occurs
    l_openings = np.flatnonzero((lgrip[1:] >= thresh) & (lgrip[:-1] < thresh))
    r_openings = np.flatnonzero((rgrip[1:] >= thresh) & (rgrip[:-1] < thresh))
    l_closings = np.flatnonzero((lgrip[1:] < thresh) & (lgrip[:-1] >= thresh))
    r_closings = np.flatnonzero((rgrip[1:] < thresh) & (rgrip[:-1] >= thresh))

    before_transitions = np.r_[l_openings, r_openings, l_closings, r_closings]
    after_transitions = before_transitions + 1
    seg_starts = np.unique(np.r_[0, after_transitions])
    seg_ends = np.unique(np.r_[before_transitions, n_steps - 1])

    return seg_starts, seg_ends


def binarize_gripper(angle):
    thresh = .04
    return angle > thresh


def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys() if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return demofile[seg_name]['cloud_xyz']

def rotate_about_median(xyz, theta):
    """                                                                                                                                             
    rotates xyz by theta around the median along the x, y dimensions                                                                                
    """
    median = np.median(xyz, axis=0)
    centered_xyz = xyz - median
    r_mat = np.eye(3)
    r_mat[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    rotated_xyz = centered_xyz.dot(r_mat)
    new_xyz = rotated_xyz + median    
    return new_xyz

FEASIBLE_REGION = [[.3, -.5], [.8, .5]]# bounds on region robot can hope to tie rope in

def place_in_feasible_region(xyz):
    max_xyz = np.max(xyz, axis=0)
    min_xyz = np.min(xyz, axis=0)
    offset = np.zeros(3)
    for i in range(2):
        if min_xyz[i] < FEASIBLE_REGION[0][i]:
            offset[i] = FEASIBLE_REGION[0][i] - min_xyz[i]
        elif max_xyz[i] > FEASIBLE_REGION[1][i]:
            offset[i] = FEASIBLE_REGION[1][i] - max_xyz[i]
    return xyz + offset



def sample_rope_state(demofile, perturb_points=5, min_rad=0, max_rad=.15):
    """
    samples a rope state, by picking a random segment, perturbing, rotating about the median, 
    then setting a random translation such that the rope is essentially within grasp room
    """

    # TODO: pick a random rope initialization
    new_xyz= load_random_start_segment(demofile)
    perturb_radius = random.uniform(min_rad, max_rad)
    rope_nodes = rope_initialization.find_path_through_point_cloud( new_xyz,
                                                                    perturb_peak_dist=perturb_radius,
                                                                    num_perturb_points=perturb_points)
    rand_theta = np.pi*(np.random.rand() - 0.5)
    # rand_theta = np.random.randn()
    rope_nodes = rotate_about_median(rope_nodes, rand_theta)
    r_trans = np.r_[np.random.multivariate_normal([0, 0], np.eye(2)), [0]]
    rope_nodes = rope_nodes + r_trans    
    rope_nodes = place_in_feasible_region(rope_nodes)
    return rope_nodes

def set_gripper_sim(lr, is_open, prev_is_open, animate=True):
    """Opens or closes the gripper. Also steps the simulation.

    Arguments:
        is_open -- boolean that is true if the gripper should be open
        prev_is_open -- boolean that is true if the gripper was open last step
    May send an open command if is_open is true but prev_is_open is false.

    Return False if the simulated gripper failed to grab the rope, eles return True.
    """
    mult = 5
    open_angle = .08 * mult
    closed_angle = .02 * mult

    target_val = open_angle if is_open else closed_angle

    # release constraints if necessary
    if is_open and not prev_is_open:
        Globals.sim.release_rope(lr)

    # execute gripper open/close trajectory
    joint_ind = Globals.robot.GetJoint("%s_gripper_l_finger_joint" % lr).GetDOFIndex()
    start_val = Globals.robot.GetDOFValues([joint_ind])[0]
    #gripper_velocity = 0.2
    #a smaller number makes the gripper move slower.
    #if the gripper moves slower then it will fling the rope less.
    gripper_velocity = 0.005
    joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / gripper_velocity))
    #print "joint_traj after retime =", joint_traj
    for i, val in enumerate(joint_traj):
        Globals.robot.SetDOFValues([val], [joint_ind])
        Globals.sim.step()
        if animate and not i%10:
            Globals.viewer.Step()
            # add constraints if necessary

    if not is_open and prev_is_open:
        if not Globals.sim.grab_rope(lr):
            return False
    return True


def unwrap_arm_traj_in_place(traj):
    assert traj.shape[1] == 7
    for i in [2, 4, 6]:
        traj[:, i] = np.unwrap(traj[:, i])
    return traj


def unwrap_in_place(t):
    # TODO: do something smarter than just checking shape[1]
    if t.shape[1] == 7:
        unwrap_arm_traj_in_place(t)
    elif t.shape[1] == 14:
        unwrap_arm_traj_in_place(t[:, :7])
        unwrap_arm_traj_in_place(t[:, 7:])
    else:
        raise NotImplementedError


def exec_traj_sim(bodypart2traj, animate):
    def sim_callback(i):
        Globals.sim.step()

    dof_inds = []
    trajs = []
    for (part_name, traj) in bodypart2traj.items():
        manip_name = {"larm": "leftarm", "rarm": "rightarm"}[part_name]
        dof_inds.extend(Globals.robot.GetManipulator(manip_name).GetArmIndices())
        trajs.append(traj)
    full_traj = np.concatenate(trajs, axis=1)
    Globals.robot.SetActiveDOFs(dof_inds)

    # make the trajectory slow enough for the simulation
    #orig full_traj = ropesim.retime_traj(Globals.robot, dof_inds, full_traj)
    full_traj = ropesim.retime_traj(Globals.robot, dof_inds, full_traj, max_cart_vel=.01)

    # in simulation mode, we must make sure to gradually move to the new starting position
    curr_vals = Globals.robot.GetActiveDOFValues()
    transition_traj = np.r_[[curr_vals], [full_traj[0]]]
    unwrap_in_place(transition_traj)
    transition_traj = ropesim.retime_traj(Globals.robot, dof_inds, transition_traj, max_cart_vel=.05)
    #transition_traj = ropesim.retime_traj(Globals.robot, dof_inds, transition_traj, max_cart_vel=.005)
    animate_traj.animate_traj(transition_traj, Globals.robot, restore=False, pause=False,
                              callback=sim_callback, step_viewer=animate)
    full_traj[0] = transition_traj[-1]
    unwrap_in_place(full_traj)

    animate_traj.animate_traj(full_traj, Globals.robot, restore=False, pause=False,
                              callback=sim_callback, step_viewer=animate)
    return True

#orig cost_reg_final = 0.01
cost_reg_final = 0.03

def registration_cost(xyz0, xyz1):
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    #TODO: n_iter was 10, reg_final was 0.01
    f, g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, n_iter=10,
                                    reg_init=10, reg_final=cost_reg_final)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost

#TODO: was 0.025
DS_SIZE = .025
#DS_SIZE = .021
#DS_SIZE = .045

#TODO: possibly memoize
#@func_utils.once
def get_downsampled_clouds(values):
    #TODO: actually fix this. It is probably better to upsample the derived segments
    cloud_list = []
    shapes = []
    for seg in values:
        cloud = seg["cloud_xyz"]
        #This eliminates the z dimension
        cloud = cloud[...].copy()
        #cloud[:, 2] = np.mean(cloud[:, 2])
        if cloud.shape[0] > 200:
            down_cloud = clouds.downsample(cloud, DS_SIZE)
        else:
            down_cloud = clouds.downsample(cloud, 0.01 * DS_SIZE)

        #down_cloud[20:, 2] = 0.66
        #down_cloud[:, 2] = np.mean(down_cloud[:, 2])
        cloud_list.append(down_cloud)
        shapes.append(down_cloud.shape)
        #print "cloud_shape = ", down_cloud.shape
    #return [clouds.downsample(seg["cloud_xyz"], DS_SIZE) for seg in demofile.values()]
    return cloud_list, shapes


def downsample(cloud, size):
    print "cloud_size", cloud.size
    return clouds.downsample(cloud, size)


def remove_inds(a, inds):
    return [x for (i, x) in enumerate(a) if i not in inds]


def find_closest_manual(demofile, _new_xyz, original=False):
    """for now, just prompt the user"""
    seg_names = demofile.keys()
    print_string = "choose from the following options (type an integer). Enter a negative number to exit."
    print print_string
    for (i, seg_name) in enumerate(seg_names):
        print "%i: %s" % (i, seg_name)
    print print_string
    choice_ind = task_execution.request_int_in_range(len(seg_names))
    if choice_ind < 0:
        return None
    chosen_seg = seg_names[choice_ind]
    return chosen_seg


def auto_choose(demofile, new_xyz, only_original_segments):
    """
    @param demofile:
    @param new_xyz:
    @param only_original_segments: if true, then only the original_segments will be registered with
    @return:
    """
    import pprint

    """Return the segment with the lowest warping cost. Takes about 2 seconds."""
    parallel = True
    if parallel:
        from joblib import Parallel, delayed
    items = demofile.items()
    if only_original_segments:
        #remove all derived segments from items
        print("Only registering with the original segments")
        items = [item for item in items if not "derived" in item[1].keys()]
    unzipped_items = zip(*items)
    keys = unzipped_items[0]
    values = unzipped_items[1]
    ds_clouds, shapes = get_downsampled_clouds(values)
    ds_new = clouds.downsample(new_xyz, 0.01 * DS_SIZE)
    #print 'ds_new_len shape', ds_new.shape
    if parallel:
        before = time.time()
        #TODO: change back n_jobs=12 ?
        costs = Parallel(n_jobs=8, verbose=0)(delayed(registration_cost)(ds_cloud, ds_new) for ds_cloud in ds_clouds)
        after = time.time()
        print "Parallel registration time in seconds =", after - before
    else:
        costs = []
        for (i, ds_cloud) in enumerate(ds_clouds):
            costs.append(registration_cost(ds_cloud, ds_new))
            print(("completed %i/%i" % (i + 1, len(ds_clouds))))
            #print(("costs\n", costs))
    ibest = np.argmin(costs)
    print "ibest = ", ibest
    #pprint.pprint(zip(keys, costs, shapes))
    #print keys
    print "best key = ", keys[ibest]
    print "best cost = ", costs[ibest]
    return keys[ibest]


def arm_moved(joint_traj):
    if len(joint_traj) < 2:
        return False
    return ((joint_traj[1:] - joint_traj[:-1]).ptp(axis=0) > .01).any()


def tpsrpm_plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f, old_xyz, new_xyz, last_one=False):
    _, src_params = registration.unit_boxify(old_xyz)
    _, targ_params = registration.unit_boxify(new_xyz)
    f = registration.unscale_tps(f, src_params, targ_params)
    #ypred_nd = f.transform_points(x_nd)
    handles = []
    #handles.append(Globals.env.plot3(ypred_nd, 3, (0, 1, 0, 1)))
    ypred_nd = f.transform_points(old_xyz)
    handles.append(Globals.env.plot3(ypred_nd, 3, (0, 1, 0, 1)))
    handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0) - np.r_[0, 0, .1],
                                               old_xyz.max(axis=0) + np.r_[0, 0, .1], xres=.1, yres=.1, zres=.04))

    if Globals.viewer:
        Globals.viewer.Step()
        time.sleep(0.1)
        # Globals.viewer.Idle()


def load_segment(demofile, segment, fake_data_transform=[0, 0, 0, 0, 0, 0]):
    fake_seg = demofile[segment]
    new_xyz = np.squeeze(fake_seg["cloud_xyz"])
    hmat = openravepy.matrixFromAxisAngle(fake_data_transform[3:6])  # @UndefinedVariable
    hmat[:3, 3] = fake_data_transform[0:3]
    new_xyz = new_xyz.dot(hmat[:3, :3].T) + hmat[:3, 3][None, :]
    r2r = ros2rave.RosToRave(Globals.robot, asarray(fake_seg["joint_states"]["name"]))
    return new_xyz, r2r


def unif_resample(traj, max_diff, wt=None):
    """
    Resample a trajectory so steps have same length in joint space
    """
    import scipy.interpolate as si

    tol = .005
    if wt is not None:
        wt = np.atleast_2d(wt)
        traj = traj * wt

    dl = mu.norms(traj[1:] - traj[:-1], 1)
    l = np.cumsum(np.r_[0, dl])
    goodinds = np.r_[True, dl > 1e-8]
    deg = min(3, sum(goodinds) - 1)
    if deg < 1:
        return traj, np.arange(len(traj))

    nsteps = max(int(np.ceil(float(l[-1]) / max_diff)), 2)
    newl = np.linspace(0, l[-1], nsteps)

    ncols = traj.shape[1]
    colstep = 10
    traj_rs = np.empty((nsteps, ncols))
    for istart in xrange(0, traj.shape[1], colstep):
        (tck, _) = si.splprep(traj[goodinds, istart:istart + colstep].T, k=deg, s=tol ** 2 * len(traj), u=l[goodinds])
        traj_rs[:, istart:istart + colstep] = np.array(si.splev(newl, tck)).T
    if wt is not None:
        traj_rs = traj_rs / wt

    newt = np.interp(newl, l, np.arange(len(traj)))
    return traj_rs, newt

#<diffuseColor>.96 .87 .70</diffuseColor>
def make_table_xml(translation, extents):
    xml = """
<Environment>
  <KinBody name="table">
    <Body type="static" name="table_link">
      <Geom type="box">
        <Translation>%f %f %f</Translation>
        <extents>%f %f %f</extents>
        <diffuseColor>.96 .87 .70</diffuseColor>
      </Geom>
    </Body>
  </KinBody>
</Environment>
""" % (translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
    return xml


PR2_L_POSTURES = dict(
    untucked=[0.4, 1.0, 0.0, -2.05, 0.0, -0.1, 0.0],
    tucked=[0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09],
    up=[0.33, -0.35, 2.59, -0.15, 0.59, -1.41, -0.27],
    side=[1.832, -0.332, 1.011, -1.437, 1.1, -2.106, 3.074]
)


def mirror_arm_joints(x):
    "mirror image of joints (r->l or l->r)"
    return np.r_[-x[0], x[1], -x[2], x[3], -x[4], x[5], -x[6]]

###################

#TODO: Change for no body
def move_sim_arms_to_side():
    """Moves the simulated arms to the side."""
    #SetDOFValues sets the joint angles. DOF = degree of feedom
    #Move the arms back ("side" posture)
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]),
                               Globals.robot.GetManipulator("rightarm").GetArmIndices())


def test_bootstrap_tps(task_params):
    """Do one task.

    Arguments:
    task_params -- a task_params object

    If task_parms.max_steps_before failure is -1, then it loops until the knot is detected.

    """
    #Begin: setup local variables from parameters
    filename = task_params.log_name
    demofile_name = task_params.demofile_name
    animate = task_params.animate
    max_steps_before_failure = task_params.max_steps_before_failure
    choose_segment = task_params.choose_segment
    knot = task_params.knot
    #End

    ### Setup ###
    set_random_seed(task_params)
    setup_log(filename)
    demofile = setup_and_return_demofile(demofile_name, 'demo1-seg00', animate=animate)
    print "set up viewer"
    Globals.viewer.Idle()
    new_xyz = Globals.sim.observe_cloud()
    old_xyz = rotate_about_median(new_xyz, -np.pi/3.0)
    int_xyz = rotate_about_median(new_xyz, -np.pi/6.0)
    print 'trying to directly map initial to target'
    handles = []
    handles.append(Globals.env.plot3(new_xyz, 5, (0, 0, 1)))
    handles.append(Globals.env.plot3(old_xyz, 5, (1, 0, 0)))
    scaled_old_xyz, src_params = registration.unit_boxify(old_xyz)
    scaled_new_xyz, targ_params = registration.unit_boxify(new_xyz)
    #TODO: get rid of g
    f, g = registration.tps_rpm_bij(scaled_old_xyz, scaled_new_xyz, plot_cb=tpsrpm_plot_cb,
                                    plotting=5 if animate else 0, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50,
                                    reg_init=10, reg_final=.01, old_xyz=old_xyz, new_xyz=new_xyz)

    redprint('reg_cost:\t'+str(f._cost + g._cost))
    print 'trying with an intermediate state'    
    handles = []
    scaled_old_xyz, src_params = registration.unit_boxify(old_xyz)
    scaled_int_xyz, int_params = registration.unit_boxify(int_xyz)
    handles.append(Globals.env.plot3(int_xyz, 5, (0, 0, 1)))
    handles.append(Globals.env.plot3(old_xyz, 5, (1, 0, 0)))
    #TODO: get rid of g
    _, _, int_corr = registration.tps_rpm_bij(scaled_old_xyz, scaled_int_xyz, plot_cb=tpsrpm_plot_cb,
                                    plotting=5 if animate else 0, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50,
                                    reg_init=10, reg_final=.01, old_xyz=old_xyz, new_xyz=int_xyz, return_corr = True)
    print 'initializing with intermediate correspondences'
    handles = []
    handles.append(Globals.env.plot3(new_xyz, 5, (0, 0, 1)))
    handles.append(Globals.env.plot3(old_xyz, 5, (1, 0, 0)))
    f, g, _ = registration.tps_rpm_bootstrap(scaled_old_xyz, scaled_int_xyz, scaled_new_xyz, int_corr, plot_cb=tpsrpm_plot_cb,
                                    plotting=5 if animate else 0, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50,
                                    reg_init=10, reg_final=.01, old_xyz=old_xyz, new_xyz=new_xyz)
    redprint('reg_cost with intermediate stage:\t'+str(f._cost + g._cost))


def add_loop_results_to_hdf5(demofile, loop_results):
    """Saves the loop_results in the hdf5 file demofile.
    Arguments: demofile is the h5py handle to the already open hdf5 file.
    loop_results is a list of dicts: [{'found_knot': found_knot, 'segment': segment, 'link2eetraj': link2eetraj,
    'new_xyz': new_xyz} ... ]
    """
    #TODO: unit test this function
    #TODO: also append a random number for all the segments (ie. generate one before the for loop)
    for loop_result in loop_results:
        parent_name = loop_result['segment']
        parent = demofile[parent_name]
        child_name = '/' + parent_name + '_' + str(random.randint(0, 10000))
        #Make a copy of the parent
        #TODO: figure out which args are necessary
        parent.copy(parent, child_name, shallow=False, expand_soft=True, expand_external=True, expand_refs=True)
        #parent.copy(parent, child_name)
        child = demofile[child_name]
        #Now update the child with loop_result
        for lr in 'lr':
            #TODO: test this step in ipython
            link_name = "%s_gripper_tool_frame" % lr
            child[link_name]["hmat"][...] = loop_result['link2eetraj'][link_name]
        del child["cloud_xyz"]
        child["cloud_xyz"] = loop_result['new_xyz']
        demofile.flush()
        if not "derived" in child.keys():
            child["derived"] = True
        demofile.flush()


def set_random_seed(task_params):
    if task_params.random_seed:
        Globals.random_seed = task_params.random_seed
        print "Found a random seed"
        print "Random seed is", Globals.random_seed


def setup_log(filename):
    if filename:
        if Globals.exec_log is None:
            redprint("Writing log to file %s" % filename)
            Globals.exec_log = task_execution.ExecutionLog(filename)
            #This will flush to the log when the program closes.
            atexit.register(Globals.exec_log.close)

#TODO  Consider encapsulating these intermedite return values in a class.
def setup_and_return_demofile(demofile_name, init_rope_state_segment,  animate, perturb_radius=0, perturb_num_points=0):
    """For the simulation, this code runs before the main loop. It also sets the numpy random seed"""
    if Globals.random_seed is not None:
        np.random.seed(Globals.random_seed)

    demofile = h5py.File(demofile_name, 'r+')
    Globals.env = openravepy.Environment() # @UndefinedVariable
    Globals.env.StopSimulation()
    Globals.env.Load("robots/pr2-beta-static.zae")
    Globals.robot = Globals.env.GetRobots()[0]

    #Set up the simulation with the table (no rope yet)
    new_xyz, _ = load_segment(demofile, init_rope_state_segment)
    #table_height = new_xyz[:, 2].mean() - .02
    table_height = new_xyz[:, 2].mean() - 0.17
    table_xml = make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
    if animate:
        Globals.viewer = trajoptpy.GetViewer(Globals.env)
    Globals.env.LoadData(table_xml)
    Globals.sim = ropesim.Simulation(Globals.env, Globals.robot)

    # create rope with optional pertubations
    
    move_sim_arms_to_side()
    rope_nodes = rope_initialization.find_path_through_point_cloud(
        new_xyz,
        perturb_peak_dist=perturb_radius,
        num_perturb_points=perturb_num_points)
    rope_nodes = rotate_about_median(rope_nodes, 2*np.pi/3.0)
    ropt_nodes = place_in_feasible_region(rope_nodes)
    Globals.sim.create(rope_nodes)
    return demofile


def loop_body(demofile, choose_segment, knot, animate, task_params, curr_step=None):
    """Do the body of the main task execution loop (ie. do a segment).
    Arguments:
        curr_step is 0 indexed
        choose_segment is a function that returns the key in the demofile to the segment
        knot is the knot the rope is checked against
        new_xyz is the new pointcloud
        task_params is used for the only_original_segments argument

    return None or {'found_knot': found_knot, 'segment': segment, 'link2eetraj': link2eetraj, 'new_xyz': new_xyz}
    """
    #TODO -- Return the new trajectory and state info to be used for bootstrapping (knot_success, new_xyz, link2eetraj,
    #TODO segment)

    #TODO -- End condition
    #TODO -- max_segments logic
    redprint("Acquire point cloud")

    move_sim_arms_to_side()
    #TODO -- Possibly refactor this section to be before the loop

    new_xyz = Globals.sim.observe_cloud()
    new_xyz_upsampled = Globals.sim.observe_cloud(upsample=120)

    print "loop_body only_original_segments", task_params.only_original_segments
    segment = choose_segment(demofile, new_xyz, task_params.only_original_segments)
    if segment is None:
        return None
    seg_info = demofile[segment]

    handles = []
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
    handles.append(Globals.env.plot3(new_xyz, 5, (0, 0, 1)))

    #TODO: test that old_xyz is now bigger if from a derived segment
    #old_xyz = clouds.downsample(old_xyz, DS_SIZE)
    if not "derived" in seg_info.keys():
        old_xyz = downsample(old_xyz, DS_SIZE)
        print "derived, so downsamping"
        #new_xyz = clouds.downsample(new_xyz, DS_SIZE)
    #new_xyz = downsample_if_big(new_xyz, DS_SIZE)
    handles.append(Globals.env.plot3(old_xyz, 5, (1, 0, 0)))

    print "new_xyz cloud size", new_xyz.shape
    print "old_xyz cloud size", old_xyz.shape
    scaled_old_xyz, src_params = registration.unit_boxify(old_xyz)
    scaled_new_xyz, targ_params = registration.unit_boxify(new_xyz)
    #TODO: get rid of g
    f, g = registration.tps_rpm_bij(scaled_old_xyz, scaled_new_xyz, plot_cb=tpsrpm_plot_cb,
                                    plotting=5 if animate else 0, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50,
                                    reg_init=10, reg_final=.01, old_xyz=old_xyz, new_xyz=new_xyz)
    f = registration.unscale_tps(f, src_params, targ_params)
    g = registration.unscale_tps(g, src_params, targ_params)
    #Globals.exec_log(curr_step, "gen_traj.f", f)

    #handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0) - np.r_[0, 0, .1],
    #                                           old_xyz.max(axis=0) + np.r_[0, 0, .1], xres=.1, yres=.1, zres=.04))
    handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0) - np.r_[0, 0, .1],
                                               old_xyz.max(axis=0) + np.r_[0, 0, .02], xres=.01, yres=.01, zres=.04))
    #handles.extend(plotting_openrave.draw_grid(Globals.env, g.transform_points, old_xyz.min(axis=0) - np.r_[0, 0, .1],
    #                                           old_xyz.max(axis=0) + np.r_[0, 0, .02], xres=.01, yres=.01, zres=.04))

    link2eetraj = {}
    #link2eetraj is a hash of gripper fram to new trajectory

    #Transform the gripper trajectory here
    for lr in 'lr':
        link_name = "%s_gripper_tool_frame" % lr
        #old_ee_traj is the old gripper trajectory
        old_ee_traj = asarray(seg_info[link_name]["hmat"])
        #new_ee_traj is the transformed gripper trajectory
        new_ee_traj = f.transform_hmats(old_ee_traj)

        link2eetraj[link_name] = new_ee_traj

        #Draw the old and new gripper trajectories as lines
        handles.append(Globals.env.drawlinestrip(old_ee_traj[:, :3, 3], 2, (1, 0, 0, 1)))
        handles.append(Globals.env.drawlinestrip(new_ee_traj[:, :3, 3], 2, (0, 1, 0, 1)))
        #Globals.exec_log(curr_step, "gen_traj.link2eetraj", link2eetraj)

    miniseg_starts, miniseg_ends = split_trajectory_by_gripper(seg_info)
    success = True
    #print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends

    #TODO: modify for no body
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):
        ################################
        redprint("Generating joint trajectory for segment %s, part %i" % (segment, i_miniseg))

        # figure out how we're gonna resample stuff
        lr2oldtraj = {}
        for lr in 'lr':
            manip_name = {"l": "leftarm", "r": "rightarm"}[lr]
            #old_joint_traj is the old trajectory inside the minisegment
            old_joint_traj = asarray(seg_info[manip_name][i_start: i_end + 1])
            #print (old_joint_traj[1:] - old_joint_traj[:-1]).ptp(axis=0), i_start, i_end
            if arm_moved(old_joint_traj):
                lr2oldtraj[lr] = old_joint_traj
        if len(lr2oldtraj) > 0:
            old_total_traj = np.concatenate(lr2oldtraj.values(), 1)
            JOINT_LENGTH_PER_STEP = .1
            _, timesteps_rs = unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP)
            ####

        ### Generate fullbody traj
        bodypart2traj = {}
        for (lr, old_joint_traj) in lr2oldtraj.items():
            manip_name = {"l": "leftarm", "r": "rightarm"}[lr]

            old_joint_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_joint_traj)), old_joint_traj)

            ee_link_name = "%s_gripper_tool_frame" % lr
            new_ee_traj = link2eetraj[ee_link_name][i_start: i_end + 1]
            new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
            #Call the planner (eg. trajopt)
            with dhm_u.suppress_stdout():
                new_joint_traj = planning.plan_follow_traj(Globals.robot, manip_name,
                                                           Globals.robot.GetLink(ee_link_name), new_ee_traj_rs,
                                                           old_joint_traj_rs)
            part_name = {"l": "larm", "r": "rarm"}[lr]
            bodypart2traj[part_name] = new_joint_traj

        ### Execute the gripper ###
        redprint("Executing joint trajectory for segment %s, part %i using arms '%s'" % (
            segment, i_miniseg, bodypart2traj.keys()))

        for lr in 'lr':
            gripper_open = binarize_gripper(seg_info["%s_gripper_joint" % lr][i_start])
            prev_gripper_open = binarize_gripper(
                seg_info["%s_gripper_joint" % lr][i_start - 1]) if i_start != 0 else False
            if not set_gripper_sim(lr, gripper_open, prev_gripper_open, animate):
                redprint("Grab %s failed" % lr)
                success = False
        if not success:
            break
            # Execute the robot trajectory
        if len(bodypart2traj) > 0:
            success &= exec_traj_sim(bodypart2traj, animate)

        #Globals.exec_log(curr_step, "execute_traj.miniseg_%d.sim_rope_nodes_after_traj" % i_miniseg, Globals.sim.rope.GetNodes())

        if not success:
            break

    Globals.sim.settle(animate=animate)
    if Globals.exec_log:
        Globals.exec_log(curr_step, "execute_traj.sim_rope_nodes_after_full_traj", Globals.sim.rope.GetNodes())

    from rapprentice import knot_identification

    knot_name = knot_identification.identify_knot(Globals.sim.rope.GetControlPoints())
    found_knot = False
    if knot_name is not None:
        if knot_name == knot or knot == "any":
            redprint("Identified knot: %s. Success!" % knot_name)
            #Globals.exec_log(curr_step, "result", True, description="identified knot %s" % knot_name)
            found_knot = True
        else:
            redprint("Identified knot: %s, but expected %s. Continuing." % (knot_name, knot))
    else:
        redprint("Not a knot. Continuing.")

    redprint("Segment %s result: %s" % (segment, success))
    return {'found_knot': found_knot, 'segment': segment, 'link2eetraj': link2eetraj, 'new_xyz': new_xyz_upsampled}


def parse_arguments():
    import argparse

    usage = """
    Run {0} --help for a list of arguments
    Warning: This may write to the hdf5 demofile.
    See https://docs.google.com/document/d/17HmaCcXd5q9QST8P2rJMzuGCd3P0Rb1UdlNZATVWQz4/pub
    """.format(sys.argv[0])

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("h5file", type=str, help="The HDF5 file that contains the recorded demonstration segments.")

    parser.add_argument("--select_manual", action="store_true",
                        help="Select the segments manually. If absent, then the segment will be automatically chosen.")

    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx", "ty", "tz", "rx", "ry", "rz"),
                        default=[0, 0, 0, 0, 0, 0], help="translation=(tx, ty, tz), axis-angle rotation=(rx, ry, rz)")

    parser.add_argument("--sim_init_perturb_radius", type=float, default=None,
                        help="How far to perturb the initial rope state. 0 in none, 0.1 is far.")

    parser.add_argument("--sim_init_perturb_num_points", type=int, default=7,
                        help="Perturb the rope state specified by fake_data_segment at this many points a distance of sim_init_perturb_radius.")

    parser.add_argument("--sim_desired_knot_name", type=str, default='K3a1',
                        help="Which knot the robot should tie. \"K3a1\" is an overhand knot.")

    parser.add_argument("--max_steps_before_failure", type=int, default=-1,
                        help="When not selecting manually (ie. automatic selection) it will declare failure after this many steps if the knot has not been detected.")

    parser.add_argument("--random_seed", type=int, default=None,
                        help="The random seed for the rope perturber. Using the same random seed (and keeping all of the other arguments the same too) allows initial perturbed rope states to be duplicated.")

    parser.add_argument("--log", type=str, default=None, help="Filename for the log file.")
    args = parser.parse_args()
    print "args =", args

    return args


def main():
    args = parse_arguments()
    if args.random_seed is not None:
        Globals.random_seed = args.random_seed
    choose_segment = find_closest_manual if args.select_manual else auto_choose
    params = TaskParameters(args.h5file, args.sim_desired_knot_name, animate=True,
                            max_steps_before_failure=args.max_steps_before_failure, choose_segment=choose_segment,
                            log_name=args.log)
    result = test_bootstrap_tps(params)
    print "Main results are", result


if __name__ == "__main__":
    main()

