import argparse
import h5py
import openravepy
from rapprentice import ros2rave, berkeley_pr2, cloud_proc_funcs
import inspect
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("output", type=str)
args = parser.parse_args()

def main():
	in_h5 = h5py.File(args.input, "r")
	out_h5 = h5py.File(args.output, "w")

	env = openravepy.Environment()
	env.Load("robots/pr2-beta-static.zae")
	robot = env.GetRobots()[0]

	seg_names = sorted([s for s in in_h5.keys() if "done" not in s])
	for i, seg_name in enumerate(seg_names):
		in_seg = in_h5[seg_name]
		new_seg_name = "seg%02d" % i
		new_seg = out_h5.create_group(new_seg_name)

		new_seg.create_group("joint_states")
		in_seg["joint_states"].copy("name", new_seg["joint_states"])
		in_seg["joint_states"].copy("position", new_seg["joint_states"])

		in_seg.copy("l_gripper_joint", new_seg)
		new_seg.create_group("l_gripper_tool_frame")
		in_seg["l_gripper_tool_frame"].copy("hmat", new_seg["l_gripper_tool_frame"])
		in_seg.copy("leftarm", new_seg)

		in_seg.copy("r_gripper_joint", new_seg)
		new_seg.create_group("r_gripper_tool_frame")
		in_seg["r_gripper_tool_frame"].copy("hmat", new_seg["r_gripper_tool_frame"])
		in_seg.copy("rightarm", new_seg)

		r2r = ros2rave.RosToRave(robot, new_seg["joint_states"]["name"])
		r2r.set_values(robot, new_seg["joint_states"]["position"][0])
		T_w_h = robot.GetLink("head_plate_frame").GetTransform()
		T_w_k = T_w_h.dot(berkeley_pr2.T_h_k)
		new_seg["T_w_k"] = T_w_k

		new_seg["cloud_proc_code"] = inspect.getsource(cloud_proc_funcs.extract_red)
		new_seg["cloud_proc_func"] = "extract_red"
		new_seg["cloud_proc_mod"] = "rapprentice.cloud_proc_funcs"
		new_seg["cloud_xyz"] = np.asarray(in_seg["cloud_xyz"]).reshape((-1, 3))
		#new_seg["depth"]
		#new_seg["rgb"]

		#new_seg["stamps"]
		new_seg["description"] = ""


	in_h5.close()
	out_h5.close()

if __name__ == "__main__":
	main()
