import time
from multiprocessing.connection import Listener
from random import shuffle, sample
import numpy as np


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
        self.__strategy = None
        self.__worker_index = None
        self.__number_of_workers = None
        self.__covering_sizes = None
        self.__w_vector = None
        self.__buckets = None
        self.__t = None
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
                    image, label, sampling_lower_bound, sampling_upper_bound, repetitions = message
                    sampling_successes, sampling_time = self.__sample(image, label, sampling_lower_bound, sampling_upper_bound, repetitions)
                    conn.send((sampling_successes, sampling_time))  # TODO: send d
                    # verification
                    self.__image, self.__label, self.__strategy, self.__worker_index, self.__number_of_workers, \
                        self.__covering_sizes, self.__w_vector, self.__buckets, self.__t = conn.recv()
                    # coverings = self.__load_coverings(strategy)
                    self.__prove(conn)
                    message = conn.recv()

    def __sample(self, image, label, sampling_lower_bound, sampling_upper_bound, repetitions):
        # TODO: also return d
        population = list(range(0, self.__number_of_pixels))
        sampling_successes = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        sampling_time = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        for size in range(sampling_lower_bound, sampling_upper_bound + 1):
            for i in range(0, repetitions):
                pixels = sample(population, size)
                start = time.time()
                verified = self.verify_group(image, label, pixels)
                duration = time.time() - start
                sampling_time[size - sampling_lower_bound] += duration
                if verified:
                    sampling_successes[size - sampling_lower_bound] += 1

        return sampling_successes, sampling_time

    def __load_covering(self, size, broken_size, t):
        covering = []
        with open(f'coverings/({size},{broken_size},{t}).txt', 'r') as coverings_file:
            for line in coverings_file:
                # TODO: ignore last line of file
                block = tuple(int(item) for item in line.split(','))
                covering.append(block)
        return covering

    def __load_coverings(self, strategy):
        t = strategy[-1]
        coverings = dict()
        for size, broken_size in zip(strategy, strategy[1:]):
            covering = []
            with open(f'coverings/({size},{broken_size},{t}).txt',
                      'r') as coverings_file:
                for line in coverings_file:
                    # TODO: ignore last line of file
                    block = tuple(int(item) for item in line.split(','))
                    covering.append(block)
                coverings[size] = covering
        return coverings

    def __prove(self, conn):
        with open(f'coverings/({self.__number_of_pixels},{self.__strategy[0]},{self.__t}).txt',
                  'r') as shared_covering:
            for line_number, line in enumerate(shared_covering):
                if conn.poll() and conn.recv() == 'stop':
                    conn.send('stopped')
                    return
                if line_number % self.__number_of_workers == self.__worker_index:
                    pixels = tuple(int(item) for item in line.split(','))
                    start = time.time()
                    verified, score = self.verify_group(self.__image, self.__label, pixels)
                    duration = time.time() - start
                    if verified:
                        conn.send((True, len(pixels), duration))
                    else:
                        conn.send((False, len(pixels), duration))
                        if len(pixels) == self.__t:
                            # We got to the end of the covering, should be checked with a complete verifier
                            conn.send('adversarial-example-suspect')
                            conn.send(pixels)
                        else:
                            self.__generate_new_strategy(self, pixels, score)
                            coverings = self.__load_coverings(self, self.__strategy)
                            groups_to_verify = self.__break_failed_group(pixels, coverings[len(pixels)])
                            while len(groups_to_verify) > 0:
                                if conn.poll() and conn.recv() == 'stop':
                                    conn.send('stopped')
                                    return
                                group_to_verify = groups_to_verify.pop(0)
                                start = time.time()
                                verified, score = self.verify_group(self.__image, self.__label, group_to_verify)
                                duration = time.time() - start
                                if verified:
                                    conn.send((True, len(group_to_verify), duration))
                                else:
                                    conn.send((False, len(group_to_verify), duration))
                                    if len(group_to_verify) in coverings:
                                        groups_to_verify = self.__break_failed_group(group_to_verify, coverings[len(group_to_verify)]) + groups_to_verify
                                    else:
                                        conn.send('adversarial-example-suspect')
                                        conn.send(group_to_verify)
                    conn.send('next')
        conn.send("done")
        message = conn.recv()
        if message != 'stop':
            raise Exception('This should not happen')
        conn.send('stopped')

    def __break_failed_group(self, pixels, covering):
        permutation = list(pixels)
        shuffle(permutation)
        return [tuple(sorted(permutation[item] for item in block)) for block in covering]

    def verify_group(self, image, label, pixels_group):
        if self.__config.normalized_region == True:
            specLB = np.copy(image)
            specUB = np.copy(image)
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
        is_correctly_classified, bounds = self.__network.test(specLB, specUB, label)
        return is_correctly_classified, self.get_score(bounds[-1], label)

    def get_score(self, last_layer_bounds, label):
        pass# TODO: write function to return score(d)

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

    def __get_bucket(self, buckets, score):
        #TODO: OMER
        bucket = None
        return bucket

    def __get_fnr(self, p_vector, v, k):
        return (1 - p_vector[k]) / (1 - p_vector[v])

    def __choose_strategy(self, p_vector, number_of_pixels):
        # Dynamic programming to choose the best strategy
        assert number_of_pixels < 100
        A = dict()
        A[self.__t] = (0, None)
        for v in range(self.__t + 1, number_of_pixels+1):
            best_k = None
            best_k_value = None
            for k in range(self.__t, number_of_pixels):
                if (v, k) not in self.__covering_sizes:
                    continue
                k_value = self.__covering_sizes[(v, k)] * (self.__w_vector[k - self.__t] + self.__get_fnr(p_vector, v, k) * A[k][0])
                if best_k_value is None or k_value < best_k_value:
                    best_k = k
                    best_k_value = k_value
            A[v] = (best_k_value, best_k)
        strategy = []
        move_to = A[self.__number_of_pixels][1]
        while move_to is not None:
            strategy.append(move_to)
            move_to = A[move_to][1]
        return strategy, A

    def __get_p_vector(self, score, pixels, n_to_sample):
        return []

    def __generate_new_strategy(self, pixels, score):
        p_vector = self.__get_p_vector(score, pixels, n_to_sample=10)
        self.__strategy, _ = self.__choose_strategy(p_vector, number_of_pixels=len(pixels))