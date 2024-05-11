import json
import os
import pprint


def load_dict(path):
    with open(path) as fd:
        results = json.load(fd)
        return results


def convert_to_more_pleasant_dict(yuval_results):
    converted_results = {}
    # cumulative time is the overall time it took from beginning to end (of each network), not per gpu
    copy_keys = ['netname', 'dataset', 't', 'sampling', 'timeout', 'from_test', 'num_tests', 'cumulative_time', 'images_results_by_index']
    copy_keys.remove('images_results_by_index')
    for key in copy_keys:
        converted_results[key] = yuval_results[key]
    converted_results['images_results_by_index'] = {}
    for key, img_res in yuval_results['images_results_by_index'].items():
        converted_results['images_results_by_index'][key] = img_res['running_time']
    converted_results['calzone_version'] = "??? - Add way to know which calzone ran"
    return converted_results


# def load_all_results(results_dir_path):
#     # TODO go over the results path and extract all the results to be able to compare
#     pass

# def get_worker_and_results_paths_from_dir(dir_path):
#     # return {'worker_paths' : ? , 'result_path' : ?}
#     pass

def get_cheated_time(results_path, worker_paths):
    results = load_dict(results_path)
    condensed_results = convert_to_more_pleasant_dict(results)
    total_time = condensed_results['cumulative_time']

    all_worker_stats = []
    for path in worker_paths:
        all_worker_stats.append(load_dict(path))

    num_workers = len(all_worker_stats)
    for worker_stats in all_worker_stats:
        for depth_stats in worker_stats.values():
            total_time -= depth_stats["sum_time_estimating_p_vector"] / num_workers

    return total_time

def get_paths(results_path):
    results_dir_path = results_path[:results_path.rfind("/") + 1]
    worker_dir_path = results_dir_path + "individual_workers"
    run_info_path = results_dir_path + "run_info.txt"
    all_workers_path = results_dir_path + "stats_collection_all_workers.json"
    return worker_dir_path, run_info_path, all_workers_path


def print_arcsin_info(run_info_path):
    found_info = False
    with open(run_info_path) as f:
        file_lines = f.readlines()
        for line in file_lines:
            if "arcsin" in line:
                print(line, end="")
                found_info = True
    if not found_info:
        print("* NO ARCSIN INFO")


def print_iterative_info(run_info_path):
    found_info = False
    with open(run_info_path) as f:
        file_lines = f.readlines()
        for line in file_lines:
            if "sigmoid_correction" in line:
                print(line, end="")
                found_info = True
    if not found_info:
        print("* NO ITERATIVE INFO")


def print_recursion_info(run_info_path):
    found_info = False
    with open(run_info_path) as f:
        file_lines = f.readlines()
        for line in file_lines:
            is_line_relevant = "l0g_min_k_for_new_strategy" in line or "l0g_max_new_strategies" in line
            if is_line_relevant:
                print(line, end="")
                found_info = True
    if not found_info:
        print("* NO RECURSION INFO")


def print_parameter_info(run_info_path):
    print_arcsin_info(run_info_path)
    print_iterative_info(run_info_path)
    print_recursion_info(run_info_path)


def main(results_path, is_original_calzone):
    if not results_path:
        print("NOTICE: add path for results of run")
        return

    worker_dir_path, run_info_path, all_workers_path = get_paths(results_path)

    # result stuff
    results = load_dict(results_path)
    condensed_results = convert_to_more_pleasant_dict(results)
    print("-------- THE RESULTS OF THE RUN --------")
    pprint.pprint(condensed_results)
    print()

    print("-------- PARAMETER INFO --------")
    print_parameter_info(run_info_path)
    print()

    # worker stuff
    if not is_original_calzone:
        worker_paths = get_list_of_paths_in_dir(worker_dir_path)
        print("-------- WORKER STATS --------")
        for worker_path in worker_paths:
            print(worker_path)
            worker_stats = load_dict(worker_path)
            pprint.pprint(worker_stats)
            print()

        # calculating cheated time
        cheated_time = get_cheated_time(results_path, worker_paths)
        print()
        print("-------- ANALYSIS --------")
        # TODO do per image
        print("non cheated time = ", condensed_results["cumulative_time"])
        print("cheated time = ", cheated_time, " (The original time minus the time estimating p_vector)")


def get_list_of_paths_in_dir(dir_path):
    paths = os.listdir(dir_path)
    paths = [os.path.join(dir_path, path) for path in paths]
    return paths


if __name__ == "__main__":
    results_path = "./results/results_240430_1427/mnist-mnist_relu_3_50.onnx-4.json"
    is_original_calzone = False
    main(results_path, is_original_calzone)
