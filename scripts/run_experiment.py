import cPickle
import tempfile
import os
import os.path as osp
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--picloud", action="store_true")
parser.add_argument("--output_file", type=argparse.FileType("w"))
parser.add_argument("--mode", choices=["single_example_no_failures", "multiple_examples", "multiple_examples_no_failures", "multiple_examples_everything", "single_demo_with_own_starting_state"])
parser.add_argument("--num_trials", type=int, default=5)
parser.add_argument("--perturb_radius", type=float)
parser.add_argument("--rand_seed_offset", type=int)
parser.add_argument("--fake", action="store_true")
parser.add_argument("--start_key", type=str)
args = parser.parse_args()

SCRIPTS_DIR = "/home/robbie/ros-stuff/robbie_git/rapprentice/scripts"
DATA_DIR = "/home/henrylu/Data/overhand"
if args.picloud:
    SCRIPTS_DIR = "/home/picloud/rapprentice/scripts"
    DATA_DIR = "/home/picloud/data"

def run_single_experiment(h5file, fake_data_segment, max_steps_before_failure, perturb_num_points, perturb_radius, random_seed, no_failure_examples, only_first_n_examples, only_examples_from_list=None):
    DO_TASK_SCRIPT = osp.join(SCRIPTS_DIR, "do_task.py")
    PARALLEL = True # is this a good idea on picloud?

    # build command
    fd, log_filename = tempfile.mkstemp(); os.close(fd)
    cmd = ["python", DO_TASK_SCRIPT]
    cmd.append(h5file)
    cmd.append("--execution=0")
    cmd.append("--animation=0")
    cmd.append("--simulation=1")
    cmd.append("--fake_data_segment=%s" % fake_data_segment)
    cmd.append("--max_steps_before_failure=%d" % max_steps_before_failure)
    if PARALLEL: cmd.append("--parallel=1")
    cmd.append("--sim_init_perturb_num_points=%d" % perturb_num_points)
    cmd.append("--sim_init_perturb_radius=%f" % perturb_radius)
    cmd.append("--random_seed=%d" % random_seed)
    cmd.append("--no_failure_examples=%d" % no_failure_examples)
    cmd.append("--only_first_n_examples=%d" % only_first_n_examples)
    if only_examples_from_list is not None: cmd.append("--only_examples_from_list=%s" % only_examples_from_list)
    cmd.append("--log=%s" % log_filename)

    # run the command
    print ">>> Running command:", " ".join(cmd)
    subprocess.call(cmd)
    with open(log_filename, "r") as f:
        log_output = cPickle.load(f)
    os.unlink(log_filename)
    return log_output

def run_single_experiment_wrapper(arg_dict):
    return run_single_experiment(**arg_dict)

def make_args_single_example_no_failures():
    out = []
    for i_trial in range(args.num_trials):
        out.append({
            "h5file": osp.join(DATA_DIR, "all.h5"),
            "fake_data_segment": "demo1-seg00",
            "max_steps_before_failure": 20,
            "perturb_num_points": 7,
            "perturb_radius": args.perturb_radius,
            "random_seed": args.rand_seed_offset + i_trial,
            "no_failure_examples": 1,
            "only_first_n_examples": 1,
        })
    return out

def make_args_multiple_examples():
    out = []
    for i_trial in range(args.num_trials):
        out.append({
            "h5file": osp.join(DATA_DIR, "overhand/all_withends.h5"),
            "fake_data_segment": "demo1-seg00",
            "max_steps_before_failure": 20,
            "perturb_num_points": 7,
            "perturb_radius": args.perturb_radius,
            "random_seed": args.rand_seed_offset + i_trial,
            "no_failure_examples": 0,
            "only_first_n_examples": 10,
        })
    return out

def make_args_multiple_examples_everything():
    out = []
    for i_trial in range(args.num_trials):
        out.append({
            "h5file": osp.join(DATA_DIR, "overhand/all_withends.h5"),
            "fake_data_segment": "demo1-seg00",
            "max_steps_before_failure": 20,
            "perturb_num_points": 7,
            "perturb_radius": args.perturb_radius,
            "random_seed": args.rand_seed_offset + i_trial,
            "no_failure_examples": 0,
            "only_first_n_examples": -1,
        })
    return out

def make_args_multiple_examples_no_failures():
    out = []
    for i_trial in range(args.num_trials):
        out.append({
            "h5file": osp.join(DATA_DIR, "overhand/all_withends.h5"),
            "fake_data_segment": "demo1-seg00",
            "max_steps_before_failure": 20,
            "perturb_num_points": 7,
            "perturb_radius": args.perturb_radius,
            "random_seed": args.rand_seed_offset + i_trial,
            "no_failure_examples": 1,
            "only_first_n_examples": 10,
        })
    return out

def make_args_single_demo_with_own_starting_state():
    if args.start_key is not None:
        start_keys = [args.start_key]
    else:
        import h5py
        demofile = h5py.File("/home/jonathan/code/rapprentice/sampledata/overhand/all_withends.h5", "r")
        start_keys = [k for k in demofile.keys() if k.startswith("demo") and k.endswith("seg00")]
        demofile.close()

    out = []
    for start_key in start_keys:
        for i_trial in range(args.num_trials):
            out.append({
                "h5file": osp.join(DATA_DIR, "overhand/all_withends.h5"),
                "fake_data_segment": start_key,
                "max_steps_before_failure": 20,
                "perturb_num_points": 7,
                "perturb_radius": args.perturb_radius,
                "random_seed": args.rand_seed_offset + i_trial,
                "no_failure_examples": 1,
                "only_first_n_examples": -1,
                "only_examples_from_list": start_key[len("demo"):].split("-")[0],
            })
    return out

def run_experiments(experiment_args):
    if args.picloud:
        import cloud
        jids = cloud.map(run_single_experiment_wrapper, experiment_args, _env="test", _type="c2")
        print "Now waiting for results..."
        results = cloud.result(jids)
        return zip(experiment_args, results)
    else:
        return zip(experiment_args, [run_single_experiment(**a) for a in experiment_args])

def main():
    if args.mode == "single_example_no_failures":
        func = make_args_single_example_no_failures
    elif args.mode == "multiple_examples":
        func = make_args_multiple_examples
    elif args.mode == "multiple_examples_no_failures":
        func = make_args_multiple_examples_no_failures
    elif args.mode == "multiple_examples_everything":
        func = make_args_multiple_examples_everything
    elif args.mode == "single_demo_with_own_starting_state":
        func = make_args_single_demo_with_own_starting_state
    else:
        assert False

    experiment_args = func()
    if args.fake:
        for x in experiment_args: print x
        return
    results = run_experiments(experiment_args)
    cPickle.dump(results, args.output_file)

if __name__ == "__main__":
    main()
