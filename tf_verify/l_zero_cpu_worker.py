import contextlib
from os import devnull
from multiprocessing.connection import Listener
from constraint_utils import get_constraints_for_dominant_label
from ai_milp import verify_network_with_milp
import numpy as np


class LZeroCpuWorker:
    def __init__(self, port, config, eran, means, stds, is_conv, dataset):
        self.__port = port
        self.__config = config
        self.__eran = eran
        self.__means = means
        self.__stds = stds
        self.__is_conv = is_conv
        self.__dataset = dataset

    def work(self):
        address = ('localhost', self.__port)
        with Listener(address, authkey=b'secret password') as listener:
            print(f"Waiting at port {self.__port}")
            with listener.accept() as conn:
                # Every iteration of this loop is one image
                message = conn.recv()
                while message != 'terminate':
                    image, label = message
                    self.__handle_image(conn, image, label)
                    conn.send('stopped')
                    message = conn.recv()

    def __handle_image(self, conn, image, label):
        jobs = []
        while True:
            while conn.poll() or len(jobs) == 0:
                message = conn.recv()
                if message != 'stop':
                    jobs.append(message)
                else:
                    return
            pixels = jobs.pop(0)
            with contextlib.redirect_stdout(open(devnull, 'w')):
                verified, adv_image, adv_label, timeout = self.verify_group(image, label, pixels)
            conn.send((pixels, verified, adv_image, adv_label, timeout))

    # TODO: code duplication
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

        verified, adv_image, adv_label, timeout = self.use_milp(specLB, specUB, label, prop)
        if verified or timeout:
            return verified, adv_image, adv_label, timeout

        real_adv_image = np.copy(image)
        for index in self.get_indexes_from_pixels(pixels_group):
            real_adv_image[index] = adv_image[index]
        spec_LB_adv = np.copy(real_adv_image)
        spec_UB_adv = np.copy(real_adv_image)
        self.normalize(spec_LB_adv)
        self.normalize(spec_UB_adv)
        perturbed_label, nn, nlb, nub, failed_labels, x = self.__eran.analyze_box(spec_LB_adv, spec_UB_adv, "deeppoly",
                                                                                  self.__config.timeout_lp,
                                                                                  self.__config.timeout_milp,
                                                                                  self.__config.use_default_heuristic,
                                                                                  label=adv_label, prop=prop, K=0, s=0,
                                                                                  timeout_final_lp=self.__config.timeout_final_lp,
                                                                                  timeout_final_milp=self.__config.timeout_final_milp,
                                                                                  use_milp=False,
                                                                                  complete=False,
                                                                                  terminate_on_failure=not self.__config.complete,
                                                                                  partial_milp=0,
                                                                                  max_milp_neurons=0,
                                                                                  approx_k=0)
        assert perturbed_label == adv_label, 'Adv example denormalize problem'
        return verified, real_adv_image.tolist(), adv_label, False



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

    def use_milp(self, specLB, specUB, label, prop):
        # TODO: ask Anan
        perturbed_label, nn, nlb, nub, failed_labels, x = self.__eran.analyze_box(specLB, specUB, "deeppoly",
                                                                                      self.__config.timeout_lp,
                                                                                      self.__config.timeout_milp,
                                                                                      self.__config.use_default_heuristic,
                                                                                      label=label, prop=prop, K=0, s=0,
                                                                                      timeout_final_lp=self.__config.timeout_final_lp,
                                                                                      timeout_final_milp=self.__config.timeout_final_milp,
                                                                                      use_milp=False,
                                                                                      complete=False,
                                                                                      terminate_on_failure=not self.__config.complete,
                                                                                      partial_milp=0,
                                                                                      max_milp_neurons=0,
                                                                                      approx_k=0)

        if (perturbed_label == label):
            return True, None, None, False

        if failed_labels is not None:
            failed_labels = list(set(failed_labels))
            constraints = get_constraints_for_dominant_label(label, failed_labels)
            verified_flag, adv_image, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
            if (verified_flag == True):
                return True, None, None, False
            else:
                if adv_image != None:
                    cex_label, _, _, _, _, _ = self.__eran.analyze_box(adv_image[0], adv_image[0], 'deepzono',
                                                                self.__config.timeout_lp, self.__config.timeout_milp,
                                                                self.__config.use_default_heuristic, approx_k=self.__config.approx_k)
                    if (cex_label != label):
                        self.denormalize(adv_image[0], self.__means, self.__stds, self.__config.dataset)
                        return False, adv_image[0], cex_label, False
                    else:
                        # TODO : ask Anan, can we get here?
                        assert False, 'This should not happen'
                        return False, None, None, False
                else:
                    # Timeout
                    return False, None, None, True

    def denormalize(self, image, means, stds, dataset):
        if dataset == 'mnist' or dataset == 'fashion':
            for i in range(len(image)):
                image[i] = image[i] * stds[0] + means[0]
        elif (dataset == 'cifar10'):
            count = 0
            tmp = np.zeros(3072)
            for i in range(1024):
                tmp[count] = image[count] * stds[0] + means[0]
                count = count + 1
                tmp[count] = image[count] * stds[1] + means[1]
                count = count + 1
                tmp[count] = image[count] * stds[2] + means[2]
                count = count + 1

            for i in range(3072):
                image[i] = tmp[i]

    def get_indexes_from_pixels(self, pixels_group):
        if self.__dataset != 'cifar10':
            return pixels_group
        indexes = []
        for pixel in pixels_group:
            indexes.append(pixel * 3)
            indexes.append(pixel * 3 + 1)
            indexes.append(pixel * 3 + 2)
        return indexes

