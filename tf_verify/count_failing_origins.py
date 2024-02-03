import json
import os


class Network:
    def __init__(self, name):
        self.name = name
        self.pics = {}
        self.total = 0

    def add_fo(self, pic, num_of_fo=1):
        if pic in self.pics:
            self.pics[pic] += num_of_fo
        else:
            self.pics[pic] = num_of_fo
        self.total += num_of_fo

    def print(self):
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
                            networks[sub_folder].add_fo(fo["image"])
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return networks.values()


if __name__ == '__main__':
    nets = get_fo_number("json_stats")
    for network in nets:
        network.print()

