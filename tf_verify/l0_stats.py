import random
import numpy as np
import __main__ as main
import json
import os

networks = ["models/mnist_relu_3_50.onnx", "models/128_0.005_94_92_0.5_0.1.onnx", "models/MNIST_convSmall_128_0.004_91_89_0.5_0.1.onnx",
            "models/MNIST_convSmall_NO_PGD.onnx", "models/MNIST_6x200_128_0.004_97_97_0.5_0.1.onnx"]

class SubKStats():
    def __init__(self, sub_k, sample_size, success_rate):
        self.sub_k = sub_k
        self.samples = sample_size
        self.success = success_rate
        # super().__init__(self, sub_k=sub_k, sample_size=sample_size, success_rate=success_rate)

    def is_same_size(self, other):
        return self.sub_k == other.sub_k

    def combine_stats(self, other):
        assert self.is_same_size(other)
        self.success = (self.success * self.samples + other.success * other.samples) / (self.samples + other.samples)
        self.samples = self.samples + other.samples

class FailingOriginStats():
    def __init__(self, dataset, network, image, k, pixels_index, label, Lbounds, Ubounds):
        self.dataset = dataset
        self.network = network
        self.image = image
        self.k = k
        self.pixels_index = pixels_index
        self.label = label
        self.Lbounds = Lbounds #nlb[-1]
        self.Ubounds = Ubounds #nub[-1]
        self.statistics = []
        # super().__init__(self, dataset = dataset, network = network, image = image,k = k,pixels_index = pixels_index,
        #               label = label,Lbounds = Lbounds,Ubounds = Ubounds,statistics = [])

    def update_stats(self, sub_k, sample_size, success_rate):
        new_stats = SubKStats(sub_k, sample_size, success_rate)
        for stat in self.statistics:
            if stat.sub_k == sub_k:
                stat.combine_stats(new_stats)
                return
        self.statistics.append(new_stats)

    def get_dict(self):
        dct = self.__dict__
        for i in range(len(dct["statistics"])):
            dct["statistics"][i] = dct["statistics"][i].__dict__
        return dct

    """
    def get_dict_2(self):  # for original format
        dct = self.__dict__.copy()
        dct.pop("statistics")
        statistics = [stat.__dict__ for stat in self.statistics]
        return [dct, statistics]
    """

def create_json(data_lst, path):
    with open(path, 'w') as f:
        data_lst_dicts = [data.get_dict() for data in data_lst]
        json.dump(data_lst_dicts, f, indent=4)

def append_json(new_data_lst, path):
    assert len(new_data_lst) != 0
    if not os.path.isfile(path):
        create_json(new_data_lst, path)
        return

    with open(path, 'rb+') as file:
        new_data_lst_dict = [data.get_dict() for data in new_data_lst]
        json_str = "," + json.dumps(new_data_lst_dict, indent=4)[1:]
        file.seek(-1, os.SEEK_END)
        file.write(json_str.encode())


def get_rnd_sample(image, k, indices=None):
    if indices is None:
        chosen_pixels = random.sample(range(len(image)), k)
    else:
        chosen_pixels = random.sample(indices, k)
    chosen_pixels.sort()
    specLB = np.copy(image)
    specUB = np.copy(image)
    for pixel in chosen_pixels:
        specLB[pixel] = 0
        specUB[pixel] = 1
    assert main.config.netname in networks
    if main.config.netname != "models/mnist_relu_3_50.onnx":
        means = [0.1307]
        stds = [0.30810001]
    main.normalize(specLB, main.means, main.stds, main.dataset)
    main.normalize(specUB, main.means, main.stds, main.dataset)
    return chosen_pixels, specLB, specUB

# start with k = 50 n = 100
# go over rnd samples of size k and find n failing origins
# for each failing origin:
#   create new FailingOrigin object
#   sub_k_diff = 1 sample_size = 10
#   for sub_k in range(1,k, step = sub_k_diff):
#       generate (sample size) random subsets samples of size sub_k and check if robust
#       update stats
#   add to failing origin list
# add list to json