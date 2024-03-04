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
                    image, label, strategy, worker_index, number_of_workers = conn.recv()
                    coverings = self.__load_coverings(strategy)
                    self.__prove(conn, image, label, strategy, worker_index, number_of_workers, coverings)
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

    def __load_coverings(self, strategy):
        # TODO: only load one covering file at a time according to out next k
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

    def __prove(self, conn, image, label, strategy, worker_index, number_of_workers, coverings):
        # TODO: replace strategy with single k
        t = strategy[-1]
        with open(f'coverings/({self.__number_of_pixels},{strategy[0]},{t}).txt',
                  'r') as shared_covering:
            for line_number, line in enumerate(shared_covering):
                if conn.poll() and conn.recv() == 'stop':
                    conn.send('stopped')
                    return
                if line_number % number_of_workers == worker_index:
                    pixels = tuple(int(item) for item in line.split(','))
                    start = time.time()
                    verified = self.verify_group(image, label, pixels) # TODO: self.verify_group will return more date
                    duration = time.time() - start
                    if verified:
                        conn.send((True, len(pixels), duration))
                    else:
                        conn.send((False, len(pixels), duration))
                        if len(pixels) not in coverings:
                            # We got to the end of the covering, should be checked with a complete verifier
                            conn.send('adversarial-example-suspect')
                            conn.send(pixels)
                        else:
                            groups_to_verify = self.__break_failed_group(pixels, coverings[len(pixels)])
                            while len(groups_to_verify) > 0:
                                if conn.poll() and conn.recv() == 'stop':
                                    conn.send('stopped')
                                    return
                                group_to_verify = groups_to_verify.pop(0)
                                start = time.time()
                                verified = self.verify_group(image, label, group_to_verify) # TODO: self.verify_group will return more date
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
        # TODO: ask Anan
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

        # TODO: ask Anan
        if self.__config.target == None:
            prop = -1
        else:
            pass
            # prop = int(target[i])
        # TODO: analyze and return score(d) from __network.test(after Anan's change return is_correctly_classified, bounds)
        return self.__network.test(specLB, specUB, label)

    def get_score(self, last_layer_bounds, label):
        # TODO: write function to return score(d)

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