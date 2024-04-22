from analasist import *
import argparse
import pickle
import random

WARMPUP_MAX_K = 100


def get_normelized_ds_for_regressor(img_bound_data, data, use_arcsin_norm=False):
    feature_data = []
    
    func_for_d = lambda fo, label=None: d_power(fo, 6, label=label) / fo["k"] / fo["k"]
    NUM_OF_BUCKETS = 100
    list_of_warmup_ds_per_img = get_list_of_ds_per_k_and_image(img_bound_data, func_for_d, None, True, data)
    list_of_warmup_d_buckets_per_img = get_buckets_for_d(list_of_warmup_ds_per_img, NUM_OF_BUCKETS)

    def get_normilized_d(fo):
        string_for_img = get_string_for_image(fo)
        warmup_d_buckets_for_image = list_of_warmup_d_buckets_per_img[string_for_img]
        bucket_index = get_estimated_bucket_index(func_for_d(fo), warmup_d_buckets_for_image)
        if use_arcsin_norm:
            return math.asin(math.asin(bucket_index / 50 - 1) / math.pi * 2)
        return bucket_index
    
    for fname, fos in data.items():
        for fo in fos:
            feature_data.append((fo["k"], fo["k"] * get_normilized_d(fo)))
    
    return np.array(feature_data)
            

def get_ys_for_regressor(data):
    alphas_over_betas, one_over_betas = [], []
    for fname, fos in data.items():
        for fo in fos:
            alpha, beta = sigmoid_weighted_least_squares(fo)
            alphas_over_betas.append(alpha / beta)
            one_over_betas.append(1 / beta)
    return np.array(alphas_over_betas), np.array(one_over_betas)


def simple_evaluate_regressors(feature_data, regressors_dict, alphas_over_betas, one_over_betas, count_sigmoids_to_draw=0):
    predicted_betas = 1 / regressors_dict["one_over_beta"].predict(feature_data)
    predicted_alphas = regressors_dict["alpha_over_beta"].predict(feature_data) * predicted_betas
    true_betas = 1 / one_over_betas
    true_alphas = alphas_over_betas * true_betas
    sigmoids_to_draw = random.sample(range(len(feature_data)), count_sigmoids_to_draw)
    max_values = []
    mean_values = []
    buckets = []
    for i, datapoint, preda, predb, truea, trueb in zip(range(len(feature_data)), feature_data, predicted_alphas, predicted_betas, true_alphas, true_betas):
        k = int(datapoint[0])
        buckets.append(datapoint[1] / k)
        d = datapoint[1]
        x = np.array(range(1, k))
        ground_trueth = sigmoid_array(truea + trueb * x)
        estimated = sigmoid_array(preda + predb * x)
        residuals = ground_trueth - estimated
        if i in sigmoids_to_draw:
            plt.plot(x, estimated)
            plt.plot(x, ground_trueth)
            plt.legend(["estimated", "ground_trueth"], ncol=1, loc='center right', bbox_to_anchor=[1, 1], fontsize=6)
            plt.title(f"{k=}, {d/k=}")
            plt.show()
        max_values.append(max(np.abs(residuals)))
        mean_values.append(sum(np.abs(residuals)) / len(residuals))
    plt.hist(buckets, 60, density=True)
    plt.title("D's after normalization distribution")
    plt.show()
    return {"max residual per fo mean":sum(max_values) / len(max_values),
            "mean residual per fo mean":sum(mean_values) / len(mean_values)}


def main():
    global data
    parser = argparse.ArgumentParser(description="Generates a single regressor and stores it.")
    parser.add_argument("--path_to_regressor", type=str, default="../tf_verify/regressor.pkl",
                        help="The path to where to save the regressor. default is ../tf_verify/regressor.pkl")
    parser.add_argument("--path_to_data", type=str, default="../tf_verify/json_stats",
                        help="The path to the data. default is ../tf_verify/json_stats")
    parser.add_argument("--path_to_ds_data", type=str, default="../tf_verify/image_bounds_stats",
                        help="The path to the data. default is ../tf_verify/image_bounds_stats")
    parser.add_argument("--test_substr", type=str, default=None,
                        help="sub string of filenames that should be skipped for training. default is None (no skip)")
    parser.add_argument("--overide_regressor", action="store_true",
                        help="Overide the regressor if the file already exists."),
    parser.add_argument("--count_sigmoids_to_draw", type=int, default=0,
                        help="Count of estimated sigmoids to draw in evaluation, random.")
    parser.add_argument("--use_arcsin_norm", action="store_true",
                        help="Use arcsin norm on bucketed d's to get a more normal looking distribution rather then uniform."),
    
    
    args = parser.parse_args()
    
    print("Loading data\n")
    full_image_data, data = get_data(args.path_to_data, args.test_substr)
    _, ds_for_normalization = get_data(args.path_to_ds_data, None)
    for network_img_string, network_img_data in ds_for_normalization.items():
        for k_data in sorted(network_img_data[:], key=lambda k_data: k_data["k"]):
            if k_data["k"] > WARMPUP_MAX_K:
                network_img_data.remove(k_data)
    feature_data = get_normelized_ds_for_regressor(ds_for_normalization, data, use_arcsin_norm=args.use_arcsin_norm)
    
    print("Calculating alphas and betas\n")
    alphas_over_betas, one_over_betas = get_ys_for_regressor(data)
    
    print("Fitting and evaluating\n")
    regressor_midpoint = LinearRegression(fit_intercept=False)
    regressor_slope = LinearRegression(fit_intercept=False)
    regressor_midpoint.fit(feature_data, alphas_over_betas)
    regressor_slope.fit(feature_data, one_over_betas)
    regressors_dict = {"alpha_over_beta" : regressor_midpoint, "one_over_beta" : regressor_slope}
    
    print(simple_evaluate_regressors(feature_data, regressors_dict, alphas_over_betas, one_over_betas, 
                                     count_sigmoids_to_draw=args.count_sigmoids_to_draw))
    
    def store_regressor():
        with open(args.path_to_regressor, "wb") as regressor_file:
            pickle.dump(regressors_dict, regressor_file)
        print(f"Saved the regressors at {args.path_to_regressor}")
    
    if args.overide_regressor or not os.path.isfile(args.path_to_regressor):
        store_regressor()
    else:
        answer = input(f"Do you wish to over write regressor at {args.path_to_regressor} [y/(n)]? ").lower()
        if answer != "" and answer[0] == "y":
            store_regressor()
    
    
if __name__ == "__main__":
    main()

