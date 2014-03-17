import openravepy as rave
import numpy as np
import bulletsimpy
from rapprentice import ropesim_floating
import IPython as ipy
import trajoptpy

env = rave.Environment()
sim = ropesim_floating.FloatingGripperSimulation(env)

xs = np.linspace(0,1,10)
rope_xyz = np.c_[xs, xs, np.zeros(10)]

sim.create(rope_xyz)
viewer = trajoptpy.GetViewer(env)
print "set viewepoint, then press p"
viewer.Idle()

ipy.embed()



