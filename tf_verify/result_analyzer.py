import json
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


def main(results_path, worker_paths):
    results = load_dict(results_path)
    condensed_results = convert_to_more_pleasant_dict(results)
    pprint.pprint(condensed_results)
    print()

    temp_path = worker_paths[0]
    worker_stats = load_dict(temp_path)
    pprint.pprint(worker_stats)

    cheated_time = get_cheated_time(results_path, worker_paths)
    print("cheated_time = ", cheated_time)


def get_list_of_paths_in_dir(dir_path):
    pass


if __name__ == "__main__":
    results_path = "./results/script_results/original_calzone-amir_script/mnist-MNIST_convSmall_NO_PGD.onnx-3.json"

    # temp_path = "./results/worker_stats/stats_collection-240322_0050-worker1.json"
    # worker_paths = [temp_path]

    dir_path = ""
    worker_paths = get_list_of_paths_in_dir(dir_path)

    main(results_path, worker_paths)  # print results, worker0 stats, and cheated time
