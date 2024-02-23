import matplotlib.pyplot as plt
import scipy
import numpy as np

## -------- Functions for d ---------------- ##
def d_power(sample, power):
    l = sample["Lbounds"][sample["label"]]
    v = [(u - l) ** power for i, u in enumerate(sample["Ubounds"]) if i != sample["label"] and u > l]
    return sum(v) ** (1 / power)


# print([u - l for u, l in zip(sample["Ubounds"], sample["Lbounds"])])
# d_power(sample, 1)

def d_sum_of_mistakes(sample):
    # for every wrong labeling > correct label, adds the difference between it and the correct label
    # d = sum({Ubound[i]-Lbound["label"] | Ubound[i] > i, i!="label"})
    total_sum = 0
    correct_label_lbound = sample["Lbounds"][sample["label"]]
    for i, ubound in enumerate(sample["Ubounds"]):
        if i == sample["label"]:
            continue
        if ubound > correct_label_lbound:
            total_sum += ubound - correct_label_lbound
    return total_sum


def d_avg_of_mistakes(sample):
    # d_sum_of_mistakes / num_mistakes
    total_sum = 0
    num_mistakes = 0
    correct_label_lbound = sample["Lbounds"][sample["label"]]
    for i, ubound in enumerate(sample["Ubounds"]):
        if i == sample["label"]:
            continue
        if ubound > correct_label_lbound:
            total_sum += ubound - correct_label_lbound
            num_mistakes += 1
    return total_sum / num_mistakes


def d_mean_of_nonlabels(sample):
    # d = Lbound["label"] - avg({Ubound["non_label"]})
    correct_label_lbound = sample["Lbounds"][sample["label"]]
    average_of_other_ubounds = 0
    for i, ubound in enumerate(sample["Ubounds"]):
        if i == sample["label"]:
            continue
        average_of_other_ubounds += ubound
    if len(sample["Ubounds"]) <= 1:
        # should not get here
        return 0
    average_of_other_ubounds /= (len(sample["Ubounds"]) - 1)
    return average_of_other_ubounds


## -------- functions for y axis ---------------- ##

from copy import Error


def get_k_of_specified_percentile(sample, percentile):
    # returns the last k who's success rate is bigger or equal (>=) to the percentile
    if percentile > 1 or percentile < 0:
        raise ValueError("need to enter a percentile between 0 and 0.99")
    k = 0
    for sub_k_stats in sample["statistics"]:
        if sub_k_stats["sub_k"] > k and sub_k_stats["success"] >= percentile:
            k = sub_k_stats["sub_k"]
    return k


def get_k_of_80_precntile(sample):
    k = 0
    for sub_k_stats in sample["statistics"]:
        if sub_k_stats["sub_k"] > k and sub_k_stats["success"] >= 0.79:  # should this be 0.8?
            k = sub_k_stats["sub_k"]
    return k


def get_average_success_rate(sample):
    summation = 0
    last_subk_success_rate = 1
    statistics_as_dict = {subk["sub_k"]: subk["success"] for subk in sample["statistics"]}
    for i in range(sample["k"]):
        if i in statistics_as_dict:
            last_subk_success_rate = statistics_as_dict[i]
        summation += last_subk_success_rate
    summation /= sample["k"]
    return summation


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def classic_ls_problem(vars, x, y, weights):
    # I want to change what's inside sigmoid to np.float128 to avoid , but it's not working
    # https://bobbyhadz.com/blog/runtime-warning-overflow-encountered-in-exp-in-python#:~:text=The%20NumPy%20%22RuntimeWarning%3A%20overflow%20encountered,float128%20before%20calling%20exp()%20.
    # returns inf, which actually seems to be fine
    inside_sigmoid = - vars[0] - vars[1] * x
    exponent = np.exp(inside_sigmoid)
    value_to_return = ((1 / (1 + exponent)) - y) * weights
    # original_value = ((1 / (1 + np.exp(- vars[0] - vars[1] * x))) - y) * weights
    return value_to_return


def calc_weight_for_s(s, lower_bound=0, upper_bound=1, weight_important_points=5):
    return weight_important_points if s < upper_bound and s > lower_bound else 1


sigmoid_weighted_least_squares_dict = {}
def sigmoid_weighted_least_squares(failing_origin, **kwargs_for_weights_calc):
    run_info = f"{failing_origin=} {kwargs_for_weights_calc=}"
    if run_info not in sigmoid_weighted_least_squares_dict:
        sigmoid_weighted_least_squares_dict[run_info] = sigmoid_weighted_least_squares_aux(failing_origin, **kwargs_for_weights_calc)
    return sigmoid_weighted_least_squares_dict[run_info]


# you can now run sigmoid_weighted_least_squares(fo, lower_bound=0.5) and it will only give weight to above 0.5 points
def sigmoid_weighted_least_squares_aux(failing_origin, **kwargs_for_weights_calc):
    subks = failing_origin["statistics"]  # sorted([subk for subk in fo["statistics"]], key= lambda subk: subk["sub_k"])
    weights = np.array([calc_weight_for_s(subk["success"], **kwargs_for_weights_calc) for subk in subks])
    x = np.array([subk["sub_k"] for subk in subks])
    y = np.array([subk["success"] for subk in subks])
    if sum(y) == len(y):
        print(y)
        np.append(y, 0)
        np.append(x, failing_origin["k"])
    vars = np.array([20, -0.5])
    bbound = (500 * 4 * 4) ** 2 
    result = scipy.optimize.least_squares(classic_ls_problem, vars, args=(x, y, weights), 
                                          bounds=((-bbound, -bbound), (bbound, bbound)))
    return result["x"]


plot=False
def successrate_scores(failing_origin, estimated_alpha, estimated_beta, score_type=None, **kwargs_for_weights_calc):
    """
    Returns scores which are based on calculating the diffrences between the estimated sucess rate un best fit sucess rate.
    """
    alpha, beta = sigmoid_weighted_least_squares(failing_origin, **kwargs_for_weights_calc)
    subks = failing_origin["statistics"]  # sorted([subk for subk in fo["statistics"]], key= lambda subk: subk["sub_k"])
    weights = np.array([calc_weight_for_s(subk["success"], **kwargs_for_weights_calc) for subk in subks])
    x = np.array(sorted([subk["sub_k"] for subk in subks]))
    ground_trueth = sigmoid_array(alpha + beta * x)
    estimated = sigmoid_array(estimated_alpha + estimated_beta * x)
    if plot:
        plt.plot(x, ground_trueth, color="green")
        plt.plot(x, estimated, color="purple")
        plt.show()
    residuals = ground_trueth - estimated
    scores = {
        "max": max(np.abs(residuals)),
        "l1_residuals": sum(np.abs(residuals)) / len(residuals),
        "l2_resiuals": (sum(residuals ** 2) ** 0.5) / len(residuals),
        "weigted_l1": sum(np.abs(residuals) * weights) / sum(weights),
        "weigted_l2": (sum(residuals ** 2 * weights) ** 0.5) / sum(weights)
    }
    if score_type != None:
        if score_type not in scores:
            print("unkown score type")
            return
        return {score_type: scores[score_type]}
    return scores


def avg_successrate_scores(fo_samples, estimated_alphas, estimated_betas, score_type=None, **kwargs_for_weights_calc):
    scores_sum = {}
    for fo, a, b in zip(fo_samples, estimated_alphas, estimated_betas):
        scores = successrate_scores(fo, a, b, score_type, **kwargs_for_weights_calc)
        for score_name, score in scores.items():
            if score_name not in scores_sum:
                scores_sum[score_name] = score
            else:
                scores_sum[score_name] += score
    scores_avg = {}
    for score_name in scores_sum.keys():
        scores_avg[score_name] = scores_sum[score_name] / len(fo_samples)
    return scores_avg


def get_fnr_sigmoid(img_sample, k, **kwargs_for_weights_calc):
    fnr_as_fo = get_fnr_as_fo(img_sample, k)
    alpha, beta = sigmoid_weighted_least_squares(fnr_as_fo, **kwargs_for_weights_calc)
    return alpha, beta


def get_fnr_as_fo(img_sample, k):
    fnr = get_all_fnr_for_k(img_sample, k)
    return {"dataset": img_sample["dataset"], "k": k,
            "network": img_sample["network"], "image": img_sample["image"],
            "statistics": [{"sub_k": sub_k, "success": 1-fnr[sub_k]} for sub_k in fnr.keys()]}


def get_all_fnr_for_k(img_sample, k):
    samples = img_sample["statistics"]
    sr = {sample["sub_k"]: sample["success"] for sample in samples if sample["sub_k"] <= k}
    k_sr = sr[k]
    fnr = {}
    for sub_k in sorted(sr.keys()):
        if sub_k == k:
            continue
        fnr[sub_k] = (1 - sr[sub_k]) / (1 - k_sr)
    return fnr


def get_fnr(origin_sample, k, sub_k):
    samples = origin_sample["statistics"]
    for sample in samples:
        if sample["sub_k"] == sub_k:
            sr_sub_k = sample["success"]
        if sample["sub_k"] == k:
            sr_k = sample["success"]
    return (1 - sr_sub_k) / (1 - sr_k)
