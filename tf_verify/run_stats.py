import sys
import os
import time

STATS_RUN_OUT_DIR = "stats_run_output"
PYTHON = "python3"
SCRIPT = "__main2__.py"
MODELS = "models"

def get_running_line(table_line):
    args = table_line.split()
    run_line = PYTHON + " " + SCRIPT
    run_line += " --domain deeppoly --epsilon 0.01 --dataset mnist"
    run_line += " --k1_lst " + args[0]
    run_line += " --failing_origins_num " + args[1]
    run_line += " --delta_sub_k " + args[2]
    run_line += " --samples_per_sub_k " + args[3]
    run_line += " --netname " + MODELS + "/" + args[4] + ".onnx"
    run_line += " --from_test " + args[5]
    run_line += " --num_test " + args[6]
    run_line += " --stats_file " + args[7]
    output_path = STATS_RUN_OUT_DIR + "/" + args[7] + "_" + args[4] + time_str
    run_line += " &> " + output_path
    return run_line, output_path


if __name__ == '__main__':
    time_str = time.strftime("%Y%m%d-%H%M%S")
    file_path = sys.argv[1]
    pid_to_output_file = {}
    with open(file_path, 'r') as file:
        for line in file.readlines():
            run_line, output_file = get_running_line(line)
            pid = os.fork()
            if pid == 0:
                os.system(run_line)
                with open(output_file, 'a') as output:
                    output.write("DONE=====================================\n")
                break
            else:
                print("pid:", pid)
