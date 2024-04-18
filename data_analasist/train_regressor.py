from analasist import *
import argparse
import pickle


def get_normelized_ds_for_regressor(img_bound_data, data):
    feature_data = []
    
    func_for_d = lambda fo, label=None: d_power(fo, 6, label=label) / fo["k"]
    NUM_OF_BUCKETS = 100
    list_of_warmup_ds_per_img = get_list_of_ds_per_k_and_image(img_bound_data, func_for_d, None, True, data)
    list_of_warmup_d_buckets_per_img = get_buckets_for_d(list_of_warmup_ds_per_img, NUM_OF_BUCKETS)

    def get_normilized_d(fo):
        string_for_img = get_string_for_image(fo)
        warmup_d_buckets_for_image = list_of_warmup_d_buckets_per_img[string_for_img]
        return get_estimated_bucket_index(func_for_d(fo), warmup_d_buckets_for_image) * fo["k"]
    
    for fname, fos in data.items():
        for fo in fos:
            feature_data.append((fo["k"], get_normilized_d(fo)))
            #print(feature_data[-1], feature_data[-1][-1] / fo["k"])
    
    return np.array(feature_data)
            

def get_ys_for_regressor(data):
    alphas_over_betas, one_over_betas = [], []
    for fname, fos in data.items():
        for fo in fos:
            alpha, beta = sigmoid_weighted_least_squares(fo)
            alphas_over_betas.append(alpha / beta)
            one_over_betas.append(1 / beta)
    return np.array(alphas_over_betas), np.array(one_over_betas)


def simple_evaluate_regressors(feature_data, regressors_dict, alphas_over_betas, one_over_betas, plot_sigmoids_p):
    predicted_betas = 1 / regressors_dict["one_over_beta"].predict(feature_data)
    predicted_alphas = regressors_dict["alpha_over_beta"].predict(feature_data) * predicted_betas
    true_betas = 1 / one_over_betas
    true_alphas = alphas_over_betas * true_betas
    values = []
    for datapoint, preda, predb, truea, trueb in zip(feature_data, predicted_alphas, predicted_betas, true_alphas, true_betas):
        k = int(datapoint[0])
        x = np.array(range(1, k))
        ground_trueth = sigmoid_array(truea + trueb * x)
        estimated = sigmoid_array(preda + predb * x)
        residuals = ground_trueth - estimated
        if np.random() < plot_sigmoids_p:
            plot_sigmoids([preda, truea], [predb, trueb])
            plt.legend(["estimated", "goal"], ncol=1, loc='center right', bbox_to_anchor=[1, 1], fontsize=6)
            plt.title(f"Sigmoids for {k=}")
            plt.show()
        values.append(max(np.abs(residuals)))
    return {"max residual per fo mean":sum(values) / len(values), "max residual per fo midian": sorted(values)[len(values) // 2]}


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
                        help="Overide the regressor if the file already exists.")
    parser.add_argument("--plot_sigmoids_p", type=int, default=0,
                        help="Probabilty to plot each sigmoid. defualt is 0 (dont plot)")
    
    
    args = parser.parse_args()
    full_image_data, data = get_data(args.path_to_data, args.test_substr)
    _, ds_for_normalization = get_data(args.path_to_ds_data, None)
        
    feature_data = get_normelized_ds_for_regressor(ds_for_normalization, data)
    alphas_over_betas, one_over_betas = get_ys_for_regressor(data)
    
    regressor_midpoint = LinearRegression(fit_intercept=False)
    regressor_slope = LinearRegression(fit_intercept=False)
    regressor_midpoint.fit(feature_data, alphas_over_betas)
    regressor_slope.fit(feature_data, one_over_betas)
    regressors_dict = {"alpha_over_beta" : regressor_midpoint, "one_over_beta" : regressor_slope}
    
    print(simple_evaluate_regressors(feature_data, regressors_dict, alphas_over_betas, one_over_betas, args.plot_sigmoids_p))
    
    if args.overide_regressor or not os.path.isfile(args.path_to_regressor) or input("Do you wish to over write [y/(n)]? ").lower()[0] == "y":
        with open(args.path_to_regressor, "wb") as regressor_file:
            pickle.dump(regressors_dict, regressor_file)
        print(f"Saved the regressors at {args.path_to_regressor}")
    
    
if __name__ == "__main__":
    main()

