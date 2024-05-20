from typing import overload
import numpy as np
import matplotlib.pyplot as plt
import scipy

class SigmoidProb:
    def __init__(self, alpha=None, beta=None, start=0, k=784):
        self.__start = start
        self.__current = self.__start
        self.__end = k
        self.alpha = alpha
        self.beta = beta
    
    def __iter__(self): # Allows to do "for probabilty in pvector: do_something()"
        self.__current = self.__start
        return self

    def __next__(self): # One step in __iter__
        self.__current += 1
        if self.__current <= self.__end and self.__current > self.__start:
            return self[self.__current - 1]
        raise StopIteration
    
    def __len__(self):
        return self.__end - self.__start
    
    def __check_index(self, index):
        if not isinstance(index, (int, np.integer)):
            raise IndexError(f"None int or any kind of np.integer index provided to SigmoidProb index={index}, type={type(index)}")
        if index >= self.__end:
            raise IndexError(f"Out of range index ({index}) to SigmoidProb ending at ({self.__end})")
        if index < self.__start:
            raise IndexError(f"Out of range idnex ({index}) to SigmoidProb starting at ({self.__start})")
        
    
    def __getitem__(self, index):
        
        if isinstance(index, slice): # For pvector[2:10]
            if not index.step is None and index.step != 1:
                raise IndexError(f"Slicing of none 1 index is not allowed for SigmoidProb, given {index.step}")
            if index.start is not None:
                self.__check_index(index.start)
            if index.stop is not None:
                self.__check_index(index.stop)
            return self.__getitem_slice(index)

        if isinstance(index, np.ndarray): # For pvector[np.arange(5, 10)]
            for i in index:
                self.__check_index(i)
        else: # For pvector[7]
            self.__check_index(index)
        return self.__getitem_index(index)
    
    def __getitem_index(self, index):
        return 1/(1 + np.exp(- self.alpha - self.beta * index))
    
    def __getitem_slice(self, __key: slice):
        start=__key.start
        k=__key.stop
        if start is None:
            start=self.__start
        if k is None:
            k=self.__end
        return SigmoidProb(alpha=self.alpha, beta=self.beta, start=start, k=k)
    
    def smart_fnr(self, large_k, sub_k): # Math at https://www.overleaf.com/read/htcqxnyjrbsz#20ed50
        assert(large_k > sub_k)
        numerator = (1 + np.exp(-(self.alpha + self.beta * large_k)))
        divisor = (1 + np.exp(-(self.alpha + self.beta * sub_k)))
        return float(np.exp(self.beta * (large_k - sub_k)) * numerator / divisor)
    
    def plot(self, ks_list=None):
        if ks_list is None:
            ks_list = np.arange(self.__start, self.__end)
        else:
            ks_list = np.array(ks_list)
        
        probs = self[ks_list]
        plt.plot(ks_list, probs)
    
    def __repr__(self):
        return f"SigmoidProbabilty(alpha={self.alpha},beta={self.beta},start={self.__start},k={self.__end})"
    
    def plot_smart_fnr(self, k):
        self.__check_index(k)
        ks_list = np.arange(self.__start, k)
        probs = [self.smart_fnr(k, sub_k) for sub_k in ks_list]
        plt.plot(ks_list, probs)
    
    def plot_stupid_fnr(self, k):
        self.__check_index(k)
        ks_list = np.arange(self.__start, k)
        probs = [(1 - self[sub_k])/(1 - self[k]) for sub_k in ks_list]
        plt.plot(ks_list, probs)

    @staticmethod
    def create_sigmoid_from_sampling(sample_func, k):
        baseline = [sample_func(sub_k) for sub_k in range(k + 1)]
        fail_ks = [i for i, sample_result in enumerate(baseline) if sample_result == 0]
        success_ks = [i for i, sample_result in enumerate(baseline) if sample_result == 1]
        alpha = -k
        beta = -1
        k_to_sample = success_ks[-1]
        while True:
            for i in range(15):
                if sample_func(k_to_sample) == 1:
                    success_ks.append(k_to_sample)
                    k_to_sample -= 1
                else:
                    fail_ks.append(k_to_sample)
                    k_to_sample += 1
            np_success_ks = np.array(success_ks)
            np_fail_ks = np.array(fail_ks)

            def func_to_minimize1(vars):
                a, b = vars
                return np.sum(np.log(1 + np.exp(- (a + b * success_ks)))) \
                    + np.sum(np.log(1 + np.exp(+ (a + b * fail_ks))))

            result = scipy.optimize.minimize(func_to_minimize1, [alpha, beta])
            if result.success:
                alpha, beta = result.x
            else:
                print(f"\nFailed to minimize scalars for double correction in correct_sigmoid_itertive\n", end=", ")
                return self
            prv_alpha = alpha
            prv_beta = beta
        return SigmoidProb(corrected_alpha, corrected_beta, self.__start, self.__end)



    def correct_sigmoid_itertive(self, sample_func, num_samples, v=3.36, var_beta=1):
        """"
        Corrects a sigmoid using sampeling, and the assumption that the error is distributing T(v=v)
        sample_func (func(int, int)->(int)): a function which takes a k, num_samples and return the amount of successes,
                                            it samples the real probabilty distribution at that k, num_samples times.
        num_samples (int, optional): num_times to sample real distribution.
        return (SigmoidProb): corrected
        """
        if num_samples == 0:
            return self
        success_ks, fail_ks = [], []


        k_to_sample = max(self.__start, min(round(- self.alpha / self.beta), self.__end - 1))  # Iterative_sampeling
        ks_to_sample = f"{k_to_sample}|"
        for i in range(num_samples):
            d = round((3 + num_samples ** 0.5) / (3 + i ** 0.5))
            if d == 0:
                d = 1
            if sample_func(k_to_sample) == 1:
                success_ks.append(k_to_sample)
                k_to_sample = min(k_to_sample + d, self.__end)
            else:
                fail_ks.append(k_to_sample)
                k_to_sample = max(k_to_sample - d, self.__start)
            ks_to_sample += f"{k_to_sample}|"
        success_ks = np.array(success_ks)
        fail_ks = np.array(fail_ks + [self.__end])
        if num_samples < 69:
            def func_to_minimize(s):
                return (v + 1) / 2 * np.log((1 + s ** 2 / v)) \
                    + np.sum(np.log(1 + np.exp(- (self.alpha + s * self.beta + self.beta * success_ks)))) \
                    + np.sum(np.log(1 + np.exp(+ (self.alpha + s * self.beta + self.beta * fail_ks))))

            result = scipy.optimize.minimize_scalar(func_to_minimize)
            if result.success:
                corrected_alpha = self.alpha + self.beta * result.x
                corrected_beta = self.beta
            else:
                print("\nFailed to minimize scalar (to find the best s) in correct_sigmoid_itertive\n")
                return self
        else: # Double correction (correct both beta and alpha)
            def func_to_minimize1(vars):
                return np.sum(np.log(1 + np.exp(- (vars[0] + vars[1] * success_ks)))) \
                     + np.sum(np.log(1 + np.exp(+ (vars[0] + vars[1] * fail_ks))))
            
            result = scipy.optimize.minimize(func_to_minimize1, [0, self.beta])
            if result.success:
                a, b = result.x
                corrected_alpha = a
                corrected_beta = b
            else:
                print(f"\nFailed to minimize scalars for double correction in correct_sigmoid_itertive\n", end=", ")
                return self
        if True:#corrected_beta > -0.001 or corrected_beta < -1000 or -corrected_alpha / corrected_beta > self.__end + 1 or -corrected_alpha / corrected_beta < self.__start:
            midpoint = -corrected_alpha / corrected_beta
            beta = corrected_beta
            t = self.__start
            k = self.__end
            self.plot()
            ks = set(success_ks)
            ks.update(set(fail_ks))
            success_per_k = {k: len([k_ for k_ in success_ks if k_ == k]) for k in ks}
            fail_per_k = {k: len([k_ for k_ in fail_ks if k_ == k]) for k in ks}
            plt.plot(sorted(ks), [success_per_k[k] / (success_per_k[k] + fail_per_k[k]) for k in ks])
            SigmoidProb(corrected_alpha, corrected_beta, self.__start, self.__end).plot()
            plt.legend(["original", "sampled", "'corrected'"], ncol=1, loc='center right',bbox_to_anchor=[1, 1])
            plt.title(f"{t=} {k=} {beta=} {midpoint=}")
            print(ks_to_sample)
            plt.show()

        return SigmoidProb(corrected_alpha, corrected_beta, self.__start, self.__end)



if __name__ == "__main__":
    print("Sum example uses / test cases")
    a = SigmoidProb(alpha=20, beta=-0.5, start=3 ,k=50)
    b = SigmoidProb(alpha=10, beta=-0.25, start=3 ,k=50)
    c = SigmoidProb(alpha=15, beta=-0.33, start=3 ,k=50)
    print(f"{a=}\n{b=}\n{c=}")
    print(f"{a[42]=}\n{b[42]=}\n{c[42]=}")
    print(f"{a[np.array([40, 41, 42, 43])]=}\n{b[np.array([40, 41, 42, 43])]=}\n{c[np.array([40, 41, 42, 43])]=}")
    a.plot()
    b.plot()
    c.plot()
    plt.legend([str(a), str(b), str(c)], ncol=1, loc='center right',
                        bbox_to_anchor=[1, 1], fontsize=6,
                        columnspacing=1.0, labelspacing=0.0,
                        handletextpad=0.0, handlelength=1.5,
                        fancybox=True, shadow=True)
    plt.show()
    print("Splicing, d=a[30:45]")
    a[30:45].plot()
    b.plot()
    c.plot()
    plt.legend([str(a) + " sliced [30:45]", str(b), str(c)], ncol=1, loc='center right',
                        bbox_to_anchor=[1, 1], fontsize=6,
                        columnspacing=1.0, labelspacing=0.0,
                        handletextpad=0.0, handlelength=1.5,
                        fancybox=True, shadow=True)
    plt.show()
    
    print(f"{a.smart_fnr(39, 37)=}")
    print(f"{a.smart_fnr(39, 30)=}")
    a.plot_smart_fnr(25)
    a.plot_stupid_fnr(25)
    plt.legend(["a.plot_smart_fnr(25)", "a.plot_stupid_fnr(25)"], ncol=1, loc='center right',
                        bbox_to_anchor=[1, 1], fontsize=6,
                        columnspacing=1.0, labelspacing=0.0,
                        handletextpad=0.0, handlelength=1.5,
                        fancybox=True, shadow=True)
    plt.show()
