import time
from multiprocessing.connection import Listener
from random import shuffle, sample
import numpy as np
import scipy
import pickle
import os
import json

from sigmoid_probabilty import SigmoidProb

class LZeroGpuWorker:
    def __init__(self, port, config, network, means, stds, is_conv, dataset):
        self.__port = port
        self.__config = config
        self.__network = network
        self.__means = means
        self.__stds = stds
        self.__is_conv = is_conv
        self.__dataset = dataset
        self.__image = None
        self.__label = None
        self.__original_strategy = None
        self.__worker_index = None
        self.__number_of_workers = None
        self.__covering_sizes = None
        self.__w_vector = None
        self.__original_p_vector = None
        self.__k_reduction_statistics = {}
        with open("regressor.pkl", "rb") as f:
            self.__regeressors = pickle.load(f)
        self.__t = None
        self.__normalization_buckets = None
        if dataset == 'cifar10':
            self.__number_of_pixels = 1024
        else:
            self.__number_of_pixels = 784

    def work(self):
        address = ('localhost', self.__port)
        with Listener(address, authkey=b'secret password') as listener:
            print(f"Waiting at port {self.__port}")
            with listener.accept() as conn:
                # Every iteration of this loop is one image
                message = conn.recv()
                while message != 'terminate':
                    # Warmup sampling
                    self.__image, self.__label, sampling_lower_bound, sampling_upper_bound, repetitions = message
                    sampling_successes, sampling_time, sampling_scores = self.__sample(sampling_lower_bound,
                                                                                       sampling_upper_bound,
                                                                                       repetitions)
                    conn.send((sampling_successes, sampling_time, sampling_scores))
                    # verification
                    self.__image, self.__label, self.__original_strategy, self.__worker_index, self.__number_of_workers, \
                    self.__covering_sizes, self.__w_vector, self.__normalization_buckets, self.__t, self.__original_p_vector = conn.recv()
                    
                    self.__original_p_vector = [1] * self.__t + self.__original_p_vector
                    # coverings = self.__load_coverings(strategy)
                    self.__prove(conn)
                    message = conn.recv()

    def __sample(self, sampling_lower_bound, sampling_upper_bound, repetitions):
        population = list(range(0, self.__number_of_pixels))
        sampling_successes = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        sampling_time = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        sampling_scores = []
        for size in range(sampling_lower_bound, sampling_upper_bound + 1):
            for i in range(0, repetitions):
                pixels = sample(population, size)
                start = time.time()
                verified, score = self.verify_group(pixels)
                duration = time.time() - start
                sampling_time[size - sampling_lower_bound] += duration
                if verified:
                    sampling_successes[size - sampling_lower_bound] += 1
                else:
                    sampling_scores.append(score)

        return sampling_successes, sampling_time, sampling_scores

    @staticmethod
    def __load_covering(size, broken_size, t):
        # Load a covering for a set of size {size} using sets of size {broken_size}
        # so that every subset of size {t} is addressed.
        covering = []
        with open(f'coverings/({size},{broken_size},{t}).txt', 'r') as coverings_file:
            for line in coverings_file:
                block = tuple(int(item) for item in line.split(','))
                covering.append(block)
        return covering

    @staticmethod
    def __load_coverings(strategy):
        # load all coverings for a given strategy
        t = strategy[-1]
        coverings = dict()
        for size, broken_size in zip(strategy, strategy[1:]):
            covering = []
            with open(f'coverings/({size},{broken_size},{t}).txt',
                      'r') as coverings_file:
                for line in coverings_file:
                    block = tuple(int(item) for item in line.split(','))
                    covering.append(block)
                coverings[size] = covering
        return coverings

    def __depth_enter(self, depth):
        if depth not in self.__k_reduction_statistics:
            self.__k_reduction_statistics[depth] = {
                "enters": 0,
                "count_subgroups": 0,
                "sum_estimated_prob_of_next_k": 0,
                "sum_sr": 0,
                "sum_time_spent_choosing_strategy": 0,
                "sum_time_estimating_p_vector": 0,
                "sum_time_loading_coverings": 0,
            }
        self.__k_reduction_statistics[depth]["enters"] += 1

    def __prove_by_strategy(self, conn, groups_to_verify, strategy, coverings, depth):
        while len(groups_to_verify) > 0:
            if self.__is_stop_signal(conn):
                return
            group_to_verify = groups_to_verify.pop(0)
            start = time.time()
            verified, score = self.verify_group(group_to_verify)
            duration = time.time() - start
            if len(strategy) > 1 and len(group_to_verify) == strategy[1]:
                self.__k_reduction_statistics[depth]["count_subgroups"] += 1
            if verified:
                if len(strategy) > 1 and len(group_to_verify) == strategy[1]:
                    self.__k_reduction_statistics[depth]["sum_sr"] += 1
                conn.send((True, len(group_to_verify), duration))
            else:
                conn.send((False, len(group_to_verify), duration))
                if len(group_to_verify) in coverings:
                    groups_to_verify = self.__break_failed_group(group_to_verify, coverings[
                        len(group_to_verify)]) + groups_to_verify
                else:
                    conn.send('adversarial-example-suspect')
                    conn.send(group_to_verify)

    def __prove_recursive(self, conn, group_to_verify, depth=0, min_k=None, max_recursion_depth=None):
        # verify group of pixels, create new strategy after each fail and continue recursively util
        # size of the group is smaller than min_k or recursion_depth = 0, will stop creating new strategies
        # if min_k and recursion_depth are None, will not stop creating new strategies
        if self.__is_stop_signal(conn):
            return
        start = time.time()
        verified, score = self.verify_group(group_to_verify)
        duration = time.time() - start
        self.__k_reduction_statistics[depth]["count_subgroups"] += 1

        if verified:
            self.__k_reduction_statistics[depth]["sum_sr"] += 1
            conn.send((True, len(group_to_verify), duration))
        else:
            conn.send((False, len(group_to_verify), duration))
            if len(group_to_verify) == self.__t:
                # We got to the end of the covering, should be checked with a complete verifier
                conn.send('adversarial-example-suspect')
                conn.send(group_to_verify)
            else:
                depth = depth + 1
                self.__depth_enter(depth)
                strategy = self.__generate_new_strategy(group_to_verify, score, depth)
                start = time.time()
                coverings = self.__load_coverings(strategy)
                self.__k_reduction_statistics[depth]["sum_time_loading_coverings"] += time.time() - start
                continue_recursion = (min_k is None or strategy[1] >= min_k) and (max_recursion_depth is None or max_recursion_depth > depth)
                if continue_recursion:
                    covering = self.__load_covering(len(group_to_verify), strategy[1], self.__t)
                    groups_to_verify = self.__break_failed_group(group_to_verify, covering)
                    for group in groups_to_verify:
                        self.__prove_recursive(conn, group, depth, min_k, max_recursion_depth)
                else:
                    coverings = self.__load_coverings(strategy)
                    groups_to_verify = self.__break_failed_group(group_to_verify, coverings[len(group_to_verify)])
                    self.__prove_by_strategy(conn, groups_to_verify, strategy, coverings, depth)

    def __prove(self, conn):
        self.__depth_enter(0)
        with open(f'coverings/({self.__number_of_pixels},{self.__original_strategy[0]},{self.__t}).txt',
                  'r') as shared_covering:
            for line_number, line in enumerate(shared_covering):
                if self.__is_stop_signal(conn):
                    return
                if line_number % self.__number_of_workers == self.__worker_index:
                    pixels = tuple(int(item) for item in line.split(','))
                    self.__prove_recursive(conn, pixels)
        self.__write_stats_to_json()
        conn.send("done")
        message = conn.recv()
        if message != 'stop':
            raise Exception('This should not happen')
        conn.send('stopped')

    def __is_stop_signal(self, conn):
        if conn.poll() and conn.recv() == 'stop':
            self.__write_stats_to_json()
            conn.send('stopped')
            return True
        return False

    def __write_stats_to_json(self):
        path_name = f"stats_collection_worker{self.__worker_index}.json"
        with open(os.path.join(self.__config.l0_results_dir, "individual_workers", path_name), "w") as res_file:
            json.dump(self.__k_reduction_statistics, res_file)

    @staticmethod
    def __break_failed_group(pixels, covering):
        permutation = list(pixels)
        shuffle(permutation)
        return [tuple(sorted(permutation[item] for item in block)) for block in covering]

    def verify_group(self, pixels_group):
        if self.__config.normalized_region == True:
            specLB = np.copy(self.__image)
            specUB = np.copy(self.__image)
            for pixel_index in self.get_indexes_from_pixels(pixels_group):
                specLB[pixel_index] = 0
                specUB[pixel_index] = 1
            self.normalize(specLB)
            self.normalize(specUB)
        else:
            pass

        if self.__config.quant_step:
            specLB = np.round(specLB / self.__config.quant_step)
            specUB = np.round(specUB / self.__config.quant_step)

        if self.__config.target == None:
            prop = -1
        else:
            pass
            # prop = int(target[i])

        is_correctly_classified, bounds = self.__network.test(specLB, specUB, self.__label)
        last_layer_bounds = bounds[-1]
        score = self.__calculate_score(last_layer_bounds, self.__label) if not is_correctly_classified else None
        return is_correctly_classified, score

    @staticmethod
    def __calculate_score(last_layer_bounds, label):
        # not implementing any different scoring methods for now
        power = 6
        label_l = last_layer_bounds[label][0]
        upper_bounds = [bounds[1] for bounds in last_layer_bounds]
        v = [(u - label_l) ** power for i, u in enumerate(upper_bounds) if i != label and u > label_l]
        return sum(v) ** (1 / power)

    def normalize(self, image):
        # normalization taken out of the network
        if len(self.__means) == len(image):
            for i in range(len(image)):
                image[i] -= self.__means[i]
                if self.__stds != None:
                    image[i] /= self.__stds[i]
        elif self.__config.dataset == 'mnist' or self.__config.dataset == 'fashion':
            for i in range(len(image)):
                image[i] = (image[i] - self.__means[0]) / self.__stds[0]
        elif (self.__config.dataset == 'cifar10'):
            count = 0
            tmp = np.zeros(3072)
            for i in range(1024):
                tmp[count] = (image[count] - self.__means[0]) / self.__stds[0]
                count = count + 1
                tmp[count] = (image[count] - self.__means[1]) / self.__stds[1]
                count = count + 1
                tmp[count] = (image[count] - self.__means[2]) / self.__stds[2]
                count = count + 1

            is_gpupoly = (self.__config.domain == 'gpupoly' or self.__config.domain == 'refinegpupoly')
            if self.__is_conv and not is_gpupoly:
                for i in range(3072):
                    image[i] = tmp[i]
                # for i in range(1024):
                #    image[i*3] = tmp[i]
                #    image[i*3+1] = tmp[i+1024]
                #    image[i*3+2] = tmp[i+2048]
            else:
                count = 0
                for i in range(1024):
                    image[i] = tmp[count]
                    count = count + 1
                    image[i + 1024] = tmp[count]
                    count = count + 1
                    image[i + 2048] = tmp[count]
                    count = count + 1

    def get_indexes_from_pixels(self, pixels_group):
        if self.__dataset != 'cifar10':
            return pixels_group
        indexes = []
        for pixel in pixels_group:
            indexes.append(pixel * 3)
            indexes.append(pixel * 3 + 1)
            indexes.append(pixel * 3 + 2)
        return indexes

    def __get_bucket(self, score, buckets=None):
        if buckets is None:
            buckets = self.__normalization_buckets

        index_above = self.__binary_search_first_above(score, buckets)
        if index_above == 0:
            return 0
        if index_above == len(buckets):
            return len(buckets)

        index_below = index_above - 1
        value_above, value_below = buckets[index_above], buckets[index_below]

        relative_pos = (score - value_below) / (value_above - value_below)
        relative_index = index_below + (relative_pos * (index_above - index_below))
        return relative_index

    def __binary_search_first_above(self, value, sorted_list):
        """
        returns the index of the first element in the list that is bigger than value
        NOTE: can return len(sorted_list) if value is bigger than all the elements in the list
        """
        left = 0
        right = len(sorted_list)
        while left < right:
            mid = (left + right) // 2
            if sorted_list[mid] < value:
                left = mid + 1
            else:
                right = mid
        return left

    def __get_fnr(self, p_vector, v, k):
        if isinstance(p_vector, SigmoidProb):
            return p_vector.smart_fnr(v, k)
        if 1 - p_vector[v] < 1e-9:
            return 0
        return (1 - p_vector[k]) / (1 - p_vector[v])

    def __choose_strategy(self, p_vector, number_of_pixels, depth):
        # Dynamic programming to choose the best strategy
        assert number_of_pixels < 100
        A = dict()
        A[self.__t] = (0, None)
        for v in range(self.__t + 1, number_of_pixels + 1):
            best_k = None
            best_k_value = None
            for k in range(self.__t, v):
                if (v, k) not in self.__covering_sizes:
                    continue
                k_value = self.__covering_sizes[(v, k)] * (
                            self.__w_vector[k - self.__t] + self.__get_fnr(p_vector, v, k) * A[k][0])
                if best_k_value is None or k_value < best_k_value:
                    best_k = k
                    best_k_value = k_value
            A[v] = (best_k_value, best_k)
        strategy = [number_of_pixels]
        move_to = A[number_of_pixels][1]
        if move_to is not None:
            self.__k_reduction_statistics[depth]["sum_estimated_prob_of_next_k"] += p_vector[move_to]
        while move_to is not None:
            strategy.append(move_to)
            move_to = A[move_to][1]
        return strategy, A

    def __get_p_vector(self, score, pixels, n_to_sample):
        fnr = lambda k: self.__get_fnr(self.__original_p_vector, len(pixels), k)
        before_midpoint = 0
        for k in range(len(pixels)):
            if fnr(k) < 0.5:
                before_midpoint = k
        # We want: t * fnr(before_midpoint) + (1 - t) * fnr(before_midpoint + 1) = 0.5
        t = (fnr(before_midpoint + 1) - 0.5) / (fnr(before_midpoint + 1) - fnr(before_midpoint))
        alpha_over_beta = before_midpoint + t
        beta = self.__image_beta * self.__original_strategy[0] / len(pixels)
        alpha = alpha_over_beta * beta
        
        def sample_func(k, n):
            return len([i for i in range(n) if self.verify_group(sample(pixels, k))[0]])

        alpha, beta = self.correct_sigmoid_itertive(alpha, beta, sample_func, n_to_sample, len(pixels))

        p_vector = SigmoidProb(alpha=alpha, beta=beta, start=self.__t, k=len(pixels))
        return p_vector

    def correct_sigmoid_itertive(self, alpha, beta, sample_func, num_samples, k, v=3.36):
        """"
        Corrects a sigmoid using sampeling, and the assumption that the error is distributing T(v=v)
        alpha (float): any real number
        beta (float): smaller then 0
        sample_func (func(int, int)->(int)): a function which takes a k, num_samples and return the amount of successes,
                                            it samples the real probabilty distribution at that k, num_samples times.
        num_samples (int, optional): num_times to sample real distribution.
        return (tuple[float, float]): corrected (alpha, beta)
        """
        if num_samples == 0:
            return (alpha, beta)
        success_ks, fail_ks = [], []

        k_to_sample = max(self.__t, min(round(- alpha / beta), k - 1))  # Iterative_sampeling
        for i in range(num_samples):
            d = round((3 + num_samples) / (3 + i))
            if d == 0:
                d = 1
            if sample_func(k_to_sample, 1) == 1:
                success_ks.append(k_to_sample)
                k_to_sample = min(k_to_sample + d, k - 1)
            else:
                fail_ks.append(k_to_sample)
                k_to_sample = max(k_to_sample - d, self.__t)

        success_ks = np.array(success_ks)
        fail_ks = np.array(fail_ks + [k])

        def func_to_minimize(s):
            return (v + 1) / 2 * np.log((1 + s ** 2 / v)) \
                   + np.sum(np.log(1 + np.exp(- (alpha + s * beta + beta * success_ks)))) \
                   + np.sum(np.log(1 + np.exp(+ (alpha + s * beta + beta * fail_ks))))

        result = scipy.optimize.minimize_scalar(func_to_minimize)
        if result.success:
            alpha = alpha + beta * result.x
        else:
            print("\nFailed to minimize scalar (to find the best s) in correct_sigmoid_itertive\n")
        return (alpha, beta)

    def __generate_new_strategy(self, pixels, score, depth):
        start = time.time()
        p_vector = self.__get_p_vector(score, pixels, n_to_sample=0)
        mid = time.time()
        self.__k_reduction_statistics[depth]["sum_time_estimating_p_vector"] += mid - start
        strategy, A = self.__choose_strategy(p_vector, number_of_pixels=len(pixels), depth=depth)
        self.__k_reduction_statistics[depth]["sum_time_spent_choosing_strategy"] += time.time() - mid
        estimated_verification_time = A[len(pixels)][0]
        bucket_of_score = self.__get_bucket(score)
        print(
            f'Worker {self.__worker_index}, Score: {bucket_of_score:.2f} | est. verif. time: {estimated_verification_time:.3f} sec | Chosen strategy: {strategy}')
        print(f'Chosen strategy is {strategy}, estimated verification time for worker {self.__worker_index} is {estimated_verification_time:.3f} sec')
        return strategy
