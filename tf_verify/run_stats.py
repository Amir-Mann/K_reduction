import sys
import os


def get_running_line(table_line):
    args = table_line.split()
    run_line = "--k1_list " + args[0]
    run_line += " --failing_origins_num " + args[1]
    run_line += " --delta_sub_k " + args[2]
    run_line += " --samples_per_sub_k " + args[3]
    run_line += " --netname " + args[4]
    run_line += " --from-test " + args[5]
    run_line += " --num-test " + args[6]
    run_line += " --stats_file " + args[7]
    run_line += " &> stats_run_output/" + args[7]
    return run_line, "stats_run_output/" + args[7]


if __name__ == '__main__':
    file_path = sys.argv[1]
    pid_to_output_file = {}
    with open(file_path, 'r') as file:
        for line in file.readlines():
            run_line, output_file = get_running_line(line)
            pid = os.fork()
            if pid > 0:
                os.system(run_line)
                with open(output_file, 'a') as output:
                    output.write("DONE=====================================\n")
