import json
import os
import re

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

# unite every file.json with the file(i).json for i 1 until cant find such file
def unite_files(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            i = 1
            file_name_split = os.path.splitext(file_name)
            file_to_add_path = os.path.join(folder_path, file_name_split[0] + "(" + str(i) + ")" + file_name_split[1])
            while os.path.isfile(file_to_add_path):
                add_lst = []
                with open(file_to_add_path, 'r') as file:
                    add_lst = json.load(file)
                with open(file_path, 'rb+') as file:
                    json_str = "," + json.dumps(add_lst, indent=4)[1:]
                    file.seek(-1, os.SEEK_END)
                    file.write(json_str.encode())
                print(file_to_add_path, "added to", file_path)
                with open(file_to_add_path, "a") as file:
                    file.write("content added to " + file_path)
                i += 1
                file_to_add_path = os.path.join(folder_path, file_name_split[0] + "(" + str(i) + ")" + file_name_split[1])

def split_image_bounds_stats(file_path):
    images = [[],[],[],[]]
    with open(file_path, 'r') as file:
        lst = json.load(file)
    for obj in lst:
        print(type(obj))
        print(obj["image"])
        images[obj["image"]].append(obj)

    for i,img in enumerate(images):
        if len(img) != 0:
            new_path = re.sub("IMG[0-9]+(-[0-9]+)?","IMG"+i, file_path)
            with open(new_path, 'w') as f:
                json.dump(img, f, indent=4)


if __name__ == '__main__':
    # split_image_bounds_stats("image_bounds_stats/MNIST_convSmall_128_0.004_91_89_0.5_0.1.onnx/IMG0-2_K1-250_SAMPALES50.json")
    # unite_files("json_stats/MNIST_convSmall_NO_PGD.onnx")
    # nets = get_fo_number("json_stats")
    # for network in nets:
    #     network.print()

    ps = [0.8, 0.9, 0.99]
    # print("mnist_relu_3_50.onnx")
    for p in ps:
        images = get_biggest_k_for_success_rate("json_stats/MNIST_convSmall_128_0.004_91_89_0.5_0.1.onnx/all_image_4.json", p)
        print("largest k with success rate", p, images)
        # for image, (k, success) in zip(images.keys(), images.values()):
        #     print("last k with success rate", p, "for image ", image, "")