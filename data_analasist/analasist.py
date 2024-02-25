import os
import json
from json.decoder import JSONDecodeError
from fo_funcs import *
from utils import *
from sklearn.linear_model import LinearRegression, Ridge

stats_folder = "../tf_verify/json_stats"

filter_out_perfect_data = False
# filter_out_all_image = True

full_image_data = {}
data = {}
for root, dirs, files in os.walk(stats_folder):
    for fname in files:
        # if filter_out_all_image and "all_image" in fname:
        # continue
        fpath = os.path.join(root, fname)
        with open(fpath, "r") as f:
            try:
                new_data = json.load(f)
            except JSONDecodeError:
                print(f"--- Encountered json decode error with file : {fpath}, skipping.")
        net_name = new_data[0]["network"][len("models/MNIST_"):-len(".onnx")]
        if "all_image" in fname:
            dataset_to_add_to = full_image_data
        else:
            dataset_to_add_to = data
        if filter_out_perfect_data:
            dataset_to_add_to[net_name + "_" + fname[:-5]] = [fo for fo in new_data if
                                                              len([subk for subk in fo["statistics"] if
                                                                   subk["success"] != 1]) > 0]
        else:
            dataset_to_add_to[net_name + "_" + fname[:-5]] = new_data


# functions for the d
functions_for_d = [
    ("L_inf", lambda sample: max(sample["Ubounds"][:sample["label"]] + sample["Ubounds"][sample["label"] + 1:]) -
                             sample["Lbounds"][sample["label"]]),
    # ("L_2", lambda sample: d_power(sample, 2)),
    ("L_10", lambda sample: d_power(sample, 10)),
    # ("L_8", lambda sample: d_power(sample, 8)),
    ("L_6", lambda sample: d_power(sample, 6)),
    # ("L_4", lambda sample: d_power(sample, 4)),
    # ("d_sum_of_mistakes", d_sum_of_mistakes),
    # ("nonlabl_mean", d_mean_of_nonlabels),
    # ("avg_of_mistakes", d_avg_of_mistakes)
]

# functions for the y
functions_for_y = [
    ("50th percentile", lambda sample: get_k_of_specified_percentile(sample, 0.50)),
    ("80th percentile", get_k_of_80_precntile),
    ("90th percentile", lambda sample: get_k_of_specified_percentile(sample, 0.90)),
    ("99th percentile", lambda sample: get_k_of_specified_percentile(sample, 0.99)),
    # ("Avg of 1st and 99th %ile", lambda sample: (get_k_of_specified_percentile(sample, 0.99)+get_k_of_specified_percentile(sample, 0.01)) / 2),
    # ("Avg of 90th, 95th, 100th %ile", lambda sample: (
    # get_k_of_specified_percentile(sample, 1)+
    # get_k_of_specified_percentile(sample, 0.9)+
    # get_k_of_specified_percentile(sample, 0.95)
    # ) / 3),
    ("Avg of 70th and 90th %ile",
     lambda sample: (get_k_of_specified_percentile(sample, 0.9) + get_k_of_specified_percentile(sample, 0.7)) / 2),
    ("Average success rate", get_average_success_rate),
]


# ---------- helper functions ----------------
def get_full_image_data_from_FO(sample):
    last_dot = sample["network"].rfind(".")
    key_name = sample["network"][13:last_dot]
    key_name = f"{key_name}_all_image"
    for fo in full_image_data[key_name]:
        if (fo["image"] == sample["image"]):
            return fo
    print(full_image_data.keys())
    return None

# Returns a string that's unique for each (dataset, network, image) 3-tuple
def get_string_for_image(sample):
    return f'{sample["dataset"]} -- {sample["network"]} -- {sample["image"]}'


# Returns a string that's unique for each (dataset, network, image, k) 4-tuple
def get_string_for_k(sample):
    return f'{sample["dataset"]} -- {sample["network"]} -- {sample["image"]} -- {sample["k"]}'


# def get_bucket_index(d, d_buckets):
#     # for i, bucket in enumerate(d_buckets):
#     #     if d < bucket:
#     #         return i
#     # return len(d_buckets) - 1



def binary_search_first_above(value, sorted_list):
    """
    returns the index of the first element in the list that is bigger than value
    NOTE: can return len(sorted_list) if value is bigger than all the elements in the list
    """
    left = 0
    right = len(sorted_list)
    while left < right:
        mid = (left + right) // 2
        if sorted_list[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


def get_rounded_bucket_index(d, sorted_d_buckets):
    # binary search for the bucket
    return binary_search_first_above(d, sorted_d_buckets)


def get_estimated_bucket_index(d, sorted_d_buckets):
    index_above = binary_search_first_above(d, sorted_d_buckets)
    if index_above == 0:
        return 0
    if index_above == len(sorted_d_buckets):
        return len(sorted_d_buckets)

    index_below = index_above - 1
    value_above, value_below = sorted_d_buckets[index_above], sorted_d_buckets[index_below]
    relative_pos = (d - value_below) / (value_above - value_below)

    relative_index = index_below + (relative_pos * (index_above - index_below))
    return relative_index



def generate_feature_info(func_for_d, file_names):
    """
    parameters:
        fuc_for_d: the function used to calculate d, of the format "func(sample) -> number"
        file_names: the names of the files to use

    returns:
        (feature_datas, feature_names, ys, y_names)
    """

    # ------------ configuration -------------------
    # file_names = [file_name for file_name in data.keys() if "relu" not in file_name]

    samples_in_use = []
    for file_name in file_names:
        samples_in_use += data[file_name]

    feature_datas = []  # list of feature_data
    feature_data_names = []  #

    ys = []
    y_names = []

    # --------------- pre-calculations ; currently calculating normalization for d -------------

    list_of_ds_per_k = {}
    list_of_ds_per_img = {}
    for file_name in file_names:
        file_name_per_k = get_string_for_k(data[file_name][0])
        file_name_per_img = get_string_for_image(data[file_name][0])
        if file_name_per_k not in list_of_ds_per_k:
            list_of_ds_per_k[file_name_per_k] = []
        if file_name_per_img not in list_of_ds_per_img:
            list_of_ds_per_img[file_name_per_img] = []

        ds_in_current_file = [func_for_d(sample) for sample in data[file_name]]

        list_of_ds_per_k[file_name_per_k] += ds_in_current_file
        list_of_ds_per_img[file_name_per_img] += ds_in_current_file

    mean_and_variance_per_k = {}
    mean_and_variance_per_img = {}
    for file_name in list_of_ds_per_k:
        mean_and_variance_per_k[file_name] = (np.mean(list_of_ds_per_k[file_name]), np.var(list_of_ds_per_k[file_name]))
    for file_name in list_of_ds_per_img:
        mean_and_variance_per_img[file_name] = (np.mean(list_of_ds_per_img[file_name]), np.var(list_of_ds_per_img[file_name]))


    NUM_OF_BUCKETS = 100
    list_of_d_buckets_per_img = {}
    for file_name_of_img in list_of_ds_per_img:
        ds_of_image = list_of_ds_per_img[file_name_of_img]
        sorted_list = sorted(ds_of_image)

        if len(ds_of_image) <= NUM_OF_BUCKETS:
            list_of_d_buckets_per_img[file_name_of_img] = sorted_list
            continue

        bucket_list = []
        for i in range(NUM_OF_BUCKETS):
            bucket_list.append(sorted_list[int(i * len(sorted_list) / NUM_OF_BUCKETS)])

        list_of_d_buckets_per_img[file_name_of_img] = bucket_list

    # -------------- Creating the datas for the features, and the ys --------------------
    first = True
    for sample in samples_in_use:
        # initializing variables that will be used
        full_image_fo = get_full_image_data_from_FO(sample)

        img_a, img_b = sigmoid_weighted_least_squares(full_image_fo)
        fo_k = sample["k"]
        fo_d = func_for_d(sample)

        # ----------normalization -------------
        # initializing variables that will be used
        string_for_img = get_string_for_image(sample)
        ds_of_image = list_of_ds_per_img[string_for_img]

        # getting per image stats
        d_mean_per_image, d_variance_per_image = mean_and_variance_per_img[string_for_img]
        d_min_per_image = min(ds_of_image)
        d_max_per_image = max(ds_of_image)
        d_buckets_for_image = list_of_d_buckets_per_img[string_for_img]

        # setting the important variables, can be changed to per_K instead of per_image
        d_mean, d_variance = d_mean_per_image, d_variance_per_image
        d_min, d_max = d_min_per_image, d_max_per_image

        # getting the normalized values
        img_d_normalized = {
            "standard": (fo_d - d_mean) / d_variance,
            "div_by_mean": fo_d / d_mean,
            "min_max": (fo_d - d_min) / (d_max - d_min),
            "rounded_bucket": get_rounded_bucket_index(fo_d, d_buckets_for_image),
            "estimated_bucket": get_estimated_bucket_index(fo_d, d_buckets_for_image),
        }

        # creating list of datapoints and features to add to the feature list ----------- ADD HERE
        datapoints = [
            ([img_a, img_b, fo_k, fo_d],                  "a_img, b_img, k, d"),
            ([fo_k],                                       "k"),
            ([fo_k, fo_d],                                "k , d"),
            ([fo_k, img_d_normalized["standard"]],         "k, d_normalized[\"standard\"]"),
            ([fo_k, img_d_normalized["standard"], 1/img_d_normalized["standard"]],  "k, |d|s, 1/|d|s"),
            ([img_a, img_b, fo_k, img_d_normalized["standard"]],         "a_img, b_img, k, |d|s"),
            ([fo_k, img_a/img_b, 1/img_b, img_d_normalized["standard"], 1/img_d_normalized["standard"]],  "k, |d|s, 1/|d|s, a/b, 1/b"),
            ([img_a, img_b, fo_k, img_d_normalized["div_by_mean"]],      "a_img, b_img, k, d_normalized[\"div_by_mean\"]"),
            ([img_a, img_b, fo_k, img_d_normalized["min_max"]],          "a_img, b_img, k, d_normalized[\"min_max\"]"),
            ([fo_k, img_d_normalized["div_by_mean"]],      "k, d_normalized[\"div_by_mean\"]"),
            ([fo_k, img_d_normalized["min_max"]],          "k, d_normalized[\"min_max\"]"),
            ([fo_k, img_d_normalized["rounded_bucket"]], "k, d_normalized[\"rounded_bucket\"]"),
            ([fo_k, img_d_normalized["estimated_bucket"]], "k, d_normalized[\"estimated_bucket\"]"),

            #([b1**i * b2**j for i in range(50) for j in range(20) for b1 in [img_d] for b2 in [img_a, img_b, fo_k]], "Overfit"),
            # ([k / img_vars[1]], "k / img_b"),
            # ([img_vars[1]], "img_b"),
            # ([1 / img_vars[1]], "1 / img_b"),
            # ([np.log(img_vars[1])], "ln(img_b)")
            # example datapoint : ([alpha_img, FO_k], "alpha_img, FO_K") ---- Dont forget to add a feature name!
        ]

        sample_a, sample_b = sigmoid_weighted_least_squares(sample)

        # y predictors
        y_predictors = [
            (sample_a, "a"),
            (sample_b, "b"),
            (sample_a / sample_b, "a/b"),
            (-(sample_a + 4) / sample_b, "-(a+4)/b"),
            (1 / sample_b, "1/b"),
        ]
        #for i in range(2, 7, 2):
        #    y_predictors.append((-(sample_a + i) / sample_b, f"-(a+{i})/b"))

        # adding the datapoints to the feature_data and the y_predictors to the ys
        adding_lists = [(datapoints, feature_datas, feature_data_names), (y_predictors, ys, y_names)]
        for given_data, list_to_add_to, names in adding_lists:
            if first:
                for t in range(len(given_data)):
                    list_to_add_to.append([])
                    names.append(given_data[t][1])
            for t, element in enumerate(given_data):
                list_to_add_to[t].append(element[0])

        first = False
    return feature_datas, feature_data_names, ys, y_names, samples_in_use


def get_all_fnr_sigmoid(fo_samples, **kwargs_for_weights_calc):
    alphas = []
    betas = []
    for fo in fo_samples:
        img_fo = get_full_image_data_from_FO(fo)
        alpha, beta = get_fnr_sigmoid(img_fo, fo["k"], **kwargs_for_weights_calc)
        alphas.append(alpha)
        betas.append(beta)
    return alphas, betas


def fit_regressor_to_data(func_for_d=None):
    # Configuration
    if func_for_d is None:
        func_for_d = lambda sample: d_power(sample, 6)
        # func_for_d = lambda sample: max(sample["Ubounds"][:sample["label"]] + sample["Ubounds"][sample["label"]+1:]) - sample["Lbounds"][sample["label"]]

    regressors = [LinearRegression()]
    regressor_names = ["Linear"]

    test_images_str = "IMG4"
    print(f"Using all images with '{test_images_str}' in their name as test")
    file_names = [file_name for file_name in data.keys() if test_images_str not in file_name]
    print(f"Amount of train files in use: {len(file_names)}")
    rest_of_file_names = [file_name for file_name in data.keys() if file_name not in file_names]
    print(f"Amount of test files in use: {len(rest_of_file_names)}")
    # rest_of_file_names = [file_name for file_name in data.keys() if "relu" not in file_name]

    # --- Generating the feature info
    train_data = generate_feature_info(func_for_d, file_names)
    test_data = generate_feature_info(func_for_d, rest_of_file_names)

    feature_datas, feature_data_names, ys, y_names, fo_samples = train_data
    test_feature_datas, _, test_ys, _, _ = test_data

    # ------------------ Scaling the features -------------------
    # feature_data = pd.DataFrame(feature_data)
    # scaler = StandardScaler()
    # scaler.fit(feature_data)
    # feature_data = scaler.transform(feature_data)

    # Making the polynomial feature
    # poly = PolynomialFeatures(4)
    # poly_feature_data = pd.DataFrame(poly.fit_transform(feature_data))

    # --------------- Fitting and scoring the regressors ----------------------

    scores = []
    train_predictions = []
    test_predictions = []
    test_scores = []
    for i, regressor in enumerate(regressors):
        scores_per_feature_data = []
        train_predictions_per_feature_data = []
        test_predictions_per_feature_data = []
        test_scores_per_feature_data = []
        for j, feature_data in enumerate(feature_datas):
            scores_per_y = []
            train_predictions_per_y = []
            test_predictions_per_y = []
            test_scores_per_y = []
            for k, y in enumerate(ys):
                if len(y) == 0:
                    continue
                regressor.fit(feature_data, y)

                # Calculating train scores
                ROUNDING_PRECISION = 5
                prediction = regressor.predict(feature_data)
                score1 = regressor.score(feature_data, y)  # R^2
                score2 = np.sum(np.abs(prediction - y)) / len(y)  # L_1 / len(y)
                score3 = (np.sum((prediction - y) ** 2) ** (1 / 2)) / len(y)  # L_2 / len(y)

                # test scores!
                test_feature_data = test_feature_datas[j]
                test_y = test_ys[k]
                test_prediction = regressor.predict(test_feature_data)
                test_score1 = regressor.score(test_feature_data, test_y)
                test_score2 = np.sum(np.abs(test_prediction - test_y)) / len(test_y)
                test_score3 = (np.sum((test_prediction - test_y) ** 2) ** (1 / 2)) / len(test_y)

                # adding and rounding scores
                temp_scores = [score1, score2, score3]
                temp_scores = list(np.array(temp_scores).round(ROUNDING_PRECISION))
                scores_per_y.append(temp_scores)

                temp_test_scores = [test_score1, test_score2, test_score3]
                temp_test_scores = list(np.array(temp_test_scores).round(ROUNDING_PRECISION))
                test_scores_per_y.append(temp_test_scores)

                train_predictions_per_y.append(prediction)
                test_predictions_per_y.append(test_prediction)

            scores_per_feature_data.append(scores_per_y)
            train_predictions_per_feature_data.append(train_predictions_per_y)
            test_predictions_per_feature_data.append(test_predictions_per_y)
            test_scores_per_feature_data.append(test_scores_per_y)

        scores.append(scores_per_feature_data)
        train_predictions.append(train_predictions_per_feature_data)
        test_predictions.append(test_predictions_per_feature_data)
        test_scores.append(test_scores_per_feature_data)

        # printing out the results
        print("--- Fitting model to (alpha_img, beta_img, K, d) ---")
        print("Training data scores:")
        for i in range(len(regressors)):
            regressor_name = regressor_names[i] if len(regressor_names) > i else "REGRESSOR NOT NAMED"
            for j in range(len(feature_datas)):
                feature_names = feature_data_names[j] if len(feature_data_names) > j else "FEATURE NOT NAMED"
                print(f"-------------------------------------------")
                print(f"Features:\t[{feature_names}]")
                print(f"Regressor:\t{regressor_names[i]}")
                print(f"Scores:\t\t[R^2, L_1, L_2]\n")
                matrix = [[f"-> {y_names[r]}", scores[i][j][r], test_scores[i][j][r]] for r in range(len(ys))]
                matrix = [["__ Y __", "__ Train __", "__ Test __"]] + matrix
                pretty_print(matrix)

    # ------------------ getting a, b -------------------
    ab_formulas = [
        {"x1_name": "a", "x2_name": "b", "comment": "a, b from regressor"},
        {"x1_name": "a/b", "x2_name": "-(a+4)/b",
         "a_func": lambda aDivb, a4Divb: (-4 * aDivb) / (aDivb + a4Divb),
         "b_func": lambda aDivb, a4Divb: (-4) / (aDivb + a4Divb)},
        # {"x1_name": "-(a+6)/b", "x2_name": "-(a+4)/b",
        #  "a_func": lambda x1, x2: (4*x1 - 6*x2)/(x2 - x1),
        #  "b_func": lambda x1, x2: -((4*x1 - 6*x2)/(x2 - x1) + 4) / x2},
        {"x1_name": "a/b", "x2_name": "b",
         "a_func": lambda x1, x2: x1 * x2},
        #{"x1_name": "a/b", "x2_name": "1/b",
        # "a_func": lambda x1, x2: x1 / x2, "b_func": lambda x1, x2: 1 / x2},
    ]

    ab_scores = get_ab_train_and_test_scores(regressors, regressor_names, feature_datas, feature_data_names, train_predictions, test_predictions, y_names, ab_formulas, fo_samples)
    print_ab_scores(ab_scores, True)


def print_ab_scores(ab_scores, is_test_score=False):
    # ------------------ printing out ab results -------------------
    print("evaluating a, b")
    # Assuming ab_train_scores and ab_test_scores are the same length and same format
    for regressor in ab_scores.keys():
        for feature in ab_scores[regressor].keys():
            print("---------------------------------------------------")
            print("Regressor: " + regressor + ", Feature: " + feature)


            if is_test_score:
                # both train and test scores
                matrix = [["formula", "--train scores--","", "", "", "", "--test scores--", "", "", "", ""]]
                matrix += [["", *ab_scores[regressor][feature][0]["scores"]["train"].keys(), *ab_scores[regressor][feature][0]["scores"]["test"].keys()]]
                matrix += [[ab_formula["name"], *[format(score, ".4f") for score in ab_formula["scores"]["train"].values()],
                            *[format(score, ".4f") for score in ab_formula["scores"]["test"].values()]]
                           for ab_formula in ab_scores[regressor][feature]]
            else:
                matrix = [["", *ab_scores[regressor][feature][0]["scores"].keys()]]
                matrix += [[ab_formula["name"], *[format(score, ".4f") for score in ab_formula["scores"].values()]] for
                           ab_formula in ab_scores[regressor][feature]]
                matrix += [["best: ", *[
                    format(min([ab_formula["scores"][name] for ab_formula in ab_scores[regressor][feature]]), ".4f") for
                    name in ab_scores[regressor][feature][0]["scores"].keys()]]]
            pretty_print(matrix)


def get_ab_train_and_test_scores(regressors, regressor_names, feature_datas, feature_data_names, train_predictions, test_predictions, y_names, ab_formulas, fo_samples):
    ab_train_scores = get_ab_scores(regressors, regressor_names, feature_datas, feature_data_names, train_predictions, y_names, ab_formulas, fo_samples)
    ab_test_scores = get_ab_scores(regressors, regressor_names, feature_datas, feature_data_names, test_predictions, y_names, ab_formulas, fo_samples)

    # uniting the train and test scores
    ab_scores = {}
    for regressor in ab_train_scores.keys():
        ab_scores[regressor] = {}
        for feature in ab_train_scores[regressor].keys():
            ab_scores[regressor][feature] = []
            for train_score, test_score in zip(ab_train_scores[regressor][feature], ab_test_scores[regressor][feature]):
                ab_scores[regressor][feature].append({"name": train_score["name"], "scores": {"train": train_score["scores"], "test": test_score["scores"]}})
    return ab_scores


def get_ab_scores(regressors, regressor_names, feature_datas, feature_data_names, predictions, y_names, ab_formulas, fo_samples):
    ab = {}
    for i in range(len(regressors)):
        regressor_name = regressor_names[i] if len(regressor_names) > i else "REGRESSOR NOT NAMED"
        ab[regressor_name] = {}
        for j in range(len(feature_datas)):
            feature_names = feature_data_names[j] if len(feature_data_names) > j else "FEATURE NOT NAMED"
            ab[regressor_name][feature_names] = []
            for formula in ab_formulas:
                x1 = predictions[i][j][y_names.index(formula["x1_name"])] if "x1_name" in formula else None
                x2 = predictions[i][j][y_names.index(formula["x2_name"])] if "x2_name" in formula else None
                if "a_func" in formula:
                    a = get_y(x1, x2, formula["a_func"])
                else:
                    a = predictions[i][j][y_names.index(formula["x1_name"])]
                if "b_func" in formula:
                    b = get_y(x1, x2, formula["b_func"])
                else:
                    b = predictions[i][j][y_names.index(formula["x2_name"])]
                scores = avg_successrate_scores(fo_samples, a, b)
                name = formula["x1_name"] + ", " + formula["x2_name"]
                if "comment" in formula:
                    name += " (" + formula["comment"] + ")"
                ab[regressor_name][feature_names].append({"name": name, "scores": scores})

            a, b = get_all_fnr_sigmoid(fo_samples)
            scores = avg_successrate_scores(fo_samples, a, b)
            ab[regressor_name][feature_names].append({"name": "fnr", "scores": scores})
    return ab

if __name__ == "__main__":
    fit_regressor_to_data()


