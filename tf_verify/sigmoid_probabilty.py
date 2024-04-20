
import numpy as np
import matplotlib.pyplot as plt
import scipy


class SigmoidProb:
    def __init__(self, alpha=None, beta=None, start=0, k=784):
        self.__start = start
        self.__current = self.__start
        self.__len = k
        self.alpha = alpha
        self.beta = beta
    
    def __iter__(self): # Allows to do "for probabilty in pvector: do_something()"
        self.__current = self.__start
        return self

    def __next__(self): # One step in __iter__
        self.__current += 1
        if self.__current < self.__len or self.__current == self.__start:
            return self[self.__current - 1]
        raise StopIteration
    
    def __len__(self):
        return self.__len
    
    def __check_index(self, index):
        if not isinstance(index, (int, np.integer)):
            raise IndexError(f"None int or any kind of np.integer index provided to SigmoidProb index={index}, type={type(index)}")
        if index >= self.__len:
            raise IndexError(f"Out of range index ({index}) to SigmoidProb of length ({self.__len})")
        if index < self.__start:
            raise IndexError(f"Out of range idnex ({index}) SigmoidProb of starting at ({self.__start})")
        
    
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
            k=self.__len
        return SigmoidProb(alpha=self.alpha, beta=self.beta, start=start, k=k)
    
    def smart_fnr(self, large_k, sub_k): # Math at https://www.overleaf.com/read/htcqxnyjrbsz#20ed50
        assert(large_k > sub_k)
        numerator = (1 + np.exp(-(self.alpha + self.beta * large_k)))
        divisor = (1 + np.exp(-(self.alpha + self.beta * sub_k)))
        return float(np.exp(self.beta * (large_k - sub_k)) * numerator / divisor)
    
    def plot(self, ks_list=None):
        if ks_list is None:
            ks_list = np.arange(self.__start, self.__len)
        else:
            ks_list = np.array(ks_list)
        
        probs = self[ks_list]
        plt.plot(ks_list, probs)
    
    def __repr__(self):
        return f"SigmoidProbabilty(alpha={self.alpha},beta={self.beta},start={self.__start},k={self.__len})"
    
    def plot_smart_fnr(self, k, plot_s=False):
        self.__check_index(k)
        ks_list = np.arange(self.__start, k)
        if not plot_s:
            probs = [self.smart_fnr(k, sub_k) for sub_k in ks_list]
        else:
            probs = [1 - self.smart_fnr(k, sub_k) for sub_k in ks_list]
        plt.plot(ks_list, probs)
    
    def plot_stupid_fnr(self, k, plot_s=False):
        self.__check_index(k)
        ks_list = np.arange(self.__start, k)
        if not plot_s:
            probs = [(1 - self[sub_k])/(1 - self[k]) for sub_k in ks_list]
        else:
            probs = [1 - (1 - self[sub_k]) / (1 - self[k]) for sub_k in ks_list]
        plt.plot(ks_list, probs)

    @staticmethod
    def classic_ls_problem(vars, x, y, weights):
        inside_sigmoid = vars[0] + vars[1] * x
        exponent = np.exp(- inside_sigmoid)
        return ((1 / (1 + exponent)) - y) * weights

    @staticmethod
    def calc_weight_for_probabilty(p, lower_bound=0, upper_bound=1, weight_important_points=5):
        return weight_important_points if p < upper_bound and p > lower_bound else 1

    @staticmethod
    def sigmoid_weighted_least_squares(p_vector, **kwargs_for_weights_calc):
        ks = np.arange(len(p_vector))  # sorted([subk for subk in fo["statistics"]], key= lambda subk: subk["sub_k"])
        weights = np.array([SigmoidProb.calc_weight_for_probabilty(p, **kwargs_for_weights_calc) for p in p_vector])

        vars = np.array([20, -0.5])
        bbound = (500 * 4 * 2) ** 2
        result = scipy.optimize.least_squares(SigmoidProb.classic_ls_problem, vars, args=(ks, np.array(p_vector), weights),
                                            bounds=((-bbound, -bbound), (bbound, bbound)))
        return SigmoidProb(*result["x"])


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