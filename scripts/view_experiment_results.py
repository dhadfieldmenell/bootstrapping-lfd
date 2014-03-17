import cPickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", type=argparse.FileType("r"))
parser.add_argument("--inspect_single", type=int)
parser.add_argument("--check_knots", action="store_true")
parser.add_argument("--start", type=int, default=0)
args = parser.parse_args()

s = args.input.read()
runs = cPickle.loads(s)
del s

import trajoptpy
import openravepy
env = openravepy.Environment()
viewer = trajoptpy.GetViewer(env)

if args.inspect_single is not None:
	experiment_args, log = runs[args.inspect_single]
	log = runs
	print "runs=", runs
	#print "args:", experiment_args
	print "segs chosen:", ", ".join([entry.data for entry in log if entry.name == "find_closest_demo.seg_name"])
	for rope_state in [entry.data for entry in log if entry.name == "execute_traj.sim_rope_nodes_after_full_traj"]:
		handles = []
		handles.append(env.drawlinestrip(rope_state, 5, [1, 0, 0]))
		viewer.Idle()

elif args.check_knots:
	from rapprentice import knot_identification as ki
	num_failed = 0
	total_runs = 0
	print "runs", runs
	for i_run, run in enumerate(runs):
		log = run
		for entry in log:
			if entry.name == "execute_traj.sim_rope_nodes_after_full_traj" and entry.step == 2:
				total_runs += 1
				last_rope_state = entry.data
		#last_rope_state = [entry.data[1] for entry in log[::-1] if entry.name == "execute_traj.sim_rope_nodes_after_full_traj" and entry.data[0]==2][0]
				result = ki.identify_knot(last_rope_state)
				print "Trial ", i_run
				print "Knot ", result
				handles = []
				handles.append(env.drawlinestrip(last_rope_state, 5, [1, 0, 0]))
				viewer.Idle()
				#print result
				if result == None:
					num_failed += 1
	print "total num failed:", num_failed, "out of", total_runs
	print "Sucess rate =", float(total_runs - num_failed) / total_runs

else:

	if args.start is not None:
		runs = runs[args.start:]

	for i_run, run in enumerate(runs):
		experiment_args, log = run

		print "===== Run # %d =====" % (i_run+args.start)
		print "args:", experiment_args
		print "segs chosen:", ", ".join([entry.data for entry in log if entry.name == "find_closest_demo.seg_name"])
		try:
			last_rope_state = [entry.data for entry in log[::-1] if entry.name == "execute_traj.sim_rope_nodes_after_full_traj"][0]
			print 'result:', [(entry.data, entry.description) for entry in log[::-1] if entry.name == "result"][0]
			#last_rope_state = [entry.data for entry in log[::-1] if entry.name == "acquire_cloud.init_sim_rope_nodes"][0]
			handles = []
			handles.append(env.drawlinestrip(last_rope_state, 5, [1, 0, 0]))
			viewer.Idle()
			print
		except:
			print 'bad'
			pass
