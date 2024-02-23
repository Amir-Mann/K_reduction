import sys
import os
import time

STATS_RUN_OUT_DIR = "stats_run_output"
# PYTHON = "/root/OUR_WORK/venv/bin/python3"
PYTHON = "python3"
SCRIPT = "__main2__.py"
MODELS = "models"

def get_running_line(table_line):
    args = table_line.split()
    netname = args[4] + ".onnx"
    run_line = PYTHON + " " + SCRIPT
    run_line += " --domain deeppoly --epsilon 0.01 --dataset mnist"
    run_line += " --k1_lst " + args[0]
    run_line += " --failing_origins_num " + args[1]
    run_line += " --delta_sub_k " + args[2]
    run_line += " --samples_per_sub_k " + args[3]
    run_line += " --netname " + MODELS + "/" + netname
    run_line += " --from_test " + args[5]
    run_line += " --num_test " + args[6]
    run_line += " --stats_file " + args[7]
    output_path = STATS_RUN_OUT_DIR + "/" + netname + "/" + args[7] + "_" + time_str
    run_line += " &> " + output_path
    return run_line, output_path

def get_running_args(table_line):
    line_split = table_line.split()
    netname = line_split[4] + ".onnx"
    args = ["__main2__.py", "--domain", "deeppoly", "--dataset", "mnist"]
    args.extend(["--k1_lst " + line_split[0]])
    args.extend(["--failing_origins_num", line_split[1]])
    args.extend(["--delta_sub_k " + line_split[2]])
    args.extend(["--samples_per_sub_k" + line_split[3]])
    args.extend(["--netname", MODELS + "/" + netname])
    args.extend(["--from_test", line_split[5]])
    args.extend(["--num_test" + line_split[6]])
    args.extend(["--stats_file" + line_split[7]])
    output_path = STATS_RUN_OUT_DIR + "/" + netname + "/" + line_split[7] + "_" + time_str
    return args, output_path

if __name__ == '__main__':
    time_str = time.strftime("%Y%m%d-%H%M%S")
    file_path = sys.argv[1]
    pid_to_output_file = {}
    with open(file_path, 'r') as file:
        for line in file.readlines():
            run_line, output_file = get_running_line(line)
            # args, output_file = get_running_args(line)
            pid = os.fork()
            if pid == 0:
                # STDIN_FILENO = 0
                # STDOUT_FILENO = 1
                # STDERR_FILENO = 2
                # # redirect stdout
                # new_stdout = os.open(output_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
                # os.dup2(new_stdout, STDOUT_FILENO)
                # # redirect stderr
                # os.dup2(new_stdout, STDERR_FILENO)
                # print(os.getcwd())
                # os.execv(PYTHON, [PYTHON]+args)
                os.system(run_line)
                time.sleep(60)
                # print("DONE=====================================")
                # new_stdout.close()
                with open(output_file, 'a') as output:
                    output.write("DONE=====================================\n")
                break
            else:
                print(line.split()[-1], "pid:", pid)
