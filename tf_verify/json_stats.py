import json
import os


class Network:
    def __init__(self, name):
        self.name = name
        self.pics = {}
        self.total = 0

    def add_fo(self, pic, k, num_of_fo=1):
        if (pic, k) in self.pics:
            self.pics[pic,k] += num_of_fo
        else:
            self.pics[pic,k] = num_of_fo
        self.total += num_of_fo

    def print(self):
        if len(self.pics) > 0:
            print(self.name, self.pics, "total:", self.total, "avg:", self.total/(len(self.pics)))
        else:
            print(self.name, self.pics, "total:", self.total)


# count number of failing origins for each network
# (only counts files in the correct sub folder for network)
def get_fo_number(folder_path):
    networks = {}
    for sub_folder in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if os.path.isdir(sub_folder_path):
            networks[sub_folder] = Network(sub_folder)
            for file_name in os.listdir(sub_folder_path):
                file_path = os.path.join(sub_folder_path, file_name)
                try:
                    with open(file_path, 'r') as file:
                        failing_origins = json.load(file)
                        for fo in failing_origins:
                            assert fo["network"] == "models/"+sub_folder
                            networks[sub_folder].add_fo(fo["image"],fo["k"])
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return networks.values()


def get_biggest_k_for_success_rate(json_path, success):
    images = {}
    with open(json_path, 'r') as file:
        failing_origins = json.load(file)
        for fo in failing_origins:
            for stat in fo["statistics"]:

                if stat["success"] > success and (fo["image"] not in images or images[fo["image"]][0] < stat["sub_k"]):
                    images[fo["image"]] = [stat["sub_k"], stat["success"]]
    return images

if __name__ == '__main__':
    nets = get_fo_number("json_stats")
    for network in nets:
        network.print()

    # ps = [0.8, 0.9, 0.99]
    # # print("MNIST_convSmall_128_0.004_91_89_0.5_0.1.onnx")
    # for p in ps:
    #     images = get_biggest_k_for_success_rate("json_stats/MNIST_convSmall_128_0.004_91_89_0.5_0.1.onnx/all_image.json", p)
    #     print("largest k with success rate", p, images)
    #     # for image, (k, success) in zip(images.keys(), images.values()):
    #     #     print("last k with success rate", p, "for image ", image, "")