import cPickle
import tempfile
import os
import os.path as osp
import subprocess
import argparse
import do_task
import time
import sys

SCRIPTS_DIR = "/home/robbie/ros-stuff/robbie_git/rapprentice/scripts"
DATA_DIR = "/mnt/storage/robbie/hdf5_working"
H5FILE = "all_exp_"


def do_both(iterations, starting_seed, func1, func2):
    f1_results = []
    f2_results = []
    for i in range(iterations):
        f1_results.append(func1(starting_seed + i))
        f2_results.append(func2(starting_seed + i))
        f1_knots = [result[0] for result in f1_results]
        f2_knots = [result[0] for result in f2_results]
        f1_final_results = [result[-1] for result in f1_knots]
        f1_passes = sum(1.0 if result else 0.0 for result in f1_final_results)
        f2_final_results = [result[-1] for result in f2_knots]
        f2_passes = sum(1.0 if result else 0.0 for result in f2_final_results)
        combined_passes = sum(1 if (f1 or f2) else 0 for f1, f2 in zip(f1_final_results, f2_final_results))

        print "On iteration (i+1)=", i + 1
        print "Func1_results =", f1_results
        print "Func2_results =", f2_results
        print "Func1_knots =", f1_knots
        print "Func2_knots =", f2_knots
        print "f1_passes =", f1_passes
        print "f2_passes =", f2_passes
        print "f1_pass_rate =", f1_passes / float(i + 1)
        print "f2_pass_rate =", f2_passes / float(i + 1)
        print "combined_pass_rate =", combined_passes / float(i + 1)
        print "total_trials =", iterations


def do_one(iterations, starting_seed, func1, original_only=False):
    f1_results = []
    print "do_one, original_only", original_only
    for i in range(iterations):
        f1_results.append(func1(starting_seed + i, original_only))
        f1_knots = [result[0] for result in f1_results]
        f1_final_results = [result[-1] for result in f1_knots]
        f1_passes = sum(1.0 if result else 0.0 for result in f1_final_results)

        print "On iteration (i+1)=", i + 1
        print "Func1_results =", f1_results
        print "Func1_knots =", f1_knots
        print "f1_passes / iterations =", f1_passes, "/", i + 1
        print "f1_pass_rate =", f1_passes / float(i + 1)
        print "total_trials =", iterations

def execute_demo1_segments(random_seed=None):
    segments = ["demo1-seg00", "demo1-seg01", "demo1-seg02"]
    #segments = ["demo1-seg00"]
    seg_iter = (x for x in segments)

    def choose_seg(_, _x):
        return seg_iter.next()

    return do_demo1(choose_seg, 3, random_seed)

def execute_demo2_segments(random_seed=None, original_only=False):
    segments = ["demo2-seg00", "demo2-seg01", "demo2-seg02"]
    #segments = ["demo1-seg00"]
    seg_iter = (x for x in segments)

    def choose_seg(_, _x, _y):
        return seg_iter.next()

    return do_demo1(choose_seg, 3, random_seed)


def make_basic_rope_state(demo):
    return do_task.RopeState(demo, 0.1, 7)
    #return do_task.RopeState(demo, 0.14, 7)


def make_basic_task_params(demofile, choose_segment, log_name):
    return do_task.TaskParameters(demofile_name=demofile, knot="K3a1", animate=False, max_steps_before_failure=6,
                                  choose_segment=choose_segment, log_name=log_name)


def do_demo1(choose_segment, max_steps=5, random_seed=None, add_to_hdf5=False, original_only=False):
    print "do_demo1 original_only", original_only
    start = time.time()
    #demo1 = "demo1-seg00"
    demo1 = "demo26-seg00"
    demofile = file_path
    rope_state = make_basic_rope_state(demo1)
    task_params = make_basic_task_params(demofile, choose_segment,
                                         "/mnt/storage/robbie/logs/test_demo1_both_auto_and_demo1-seed841-1.pkl")
    task_params.random_seed = random_seed
    task_params.max_steps_before_failure = max_steps
    task_params.add_to_hdf5 = add_to_hdf5
    task_params.only_original_segments = original_only
    #TODO: remove
    #task_params.animate = True
    return_val = do_task.do_single_random_task(rope_state, task_params)
    print "do_segments return val = ", return_val
    end = time.time()
    print "seconds taken =", end - start
    return return_val


def do_auto(random_seed=None, original_only=False):
    #TODO: change max_steps to 5?
    print "do_auto original_only", original_only
    return do_demo1(do_task.auto_choose, 5, random_seed, add_to_hdf5=to_add, original_only=original_only)


def main(original_only=False):
    #do_both(100, 841, execute_demo1_segments, do_auto)
    #do_both(100, 4215, execute_demo1_segments, do_auto)
    print "main original_only", original_only
    #do_one(100, 841, do_auto, original_only=original_only)
    #do_one(100, 4215, execute_demo2_segments)
    #do_one(20, 841, execute_demo1_segments)
    do_one(60, 100, do_auto, original_only=False)


def parse_arguments():
    import argparse

    usage = """
    Run {0} --help for a list of arguments
    Warning: This may write to the hdf5 demofile.
    See https://docs.google.com/document/d/17HmaCcXd5q9QST8P2rJMzuGCd3P0Rb1UdlNZATVWQz4/pub
    """.format(sys.argv[0])

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--add", action="store_true", help="Will write to the hdf5 file.")
    parser.add_argument("--original", action="store_true", help="Will only register with the original segments.")
    args = parser.parse_args()
    print "args =", args
    return args


if __name__ == "__main__":
    args = parse_arguments()
    global to_add
    to_add = args.add
    original = args.original
    print "original_args", original
    print "to_add = ", to_add
    try:
        print 'Enter the number of the data file you wish to use.\n'
        file_number = int(raw_input('Input:'))
        file_number_string = str(file_number)
    except ValueError:
        print 'Not a number'
    try:
        datafile = H5FILE + file_number_string + ".h5"
        global file_path
        file_path = osp.join(DATA_DIR, datafile)
        if osp.isfile(file_path):
            print 'File already exists would you like to use a copy or not add to hdf5 file?\n'
            print 'Enter 1 to make a copy, 2 to not add to hdf5 file, and -1 to exit'
            input = int(raw_input('Input:'))
            if (input == 1):
                datafile = H5FILE + file_number_string + '_copy.h5'
                file_path = osp.join(DATA_DIR, datafile)
            if (input == 2):
                to_add = False;
                pass
            if (input == -1):
                print 'Exiting'
                sys.exit()
            main(original_only = original)
    except ValueError:
        print 'Not a number'
