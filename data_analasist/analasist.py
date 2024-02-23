import os
import json
from json.decoder import JSONDecodeError
from fo_funcs import *
from utils import *
from sklearn.linear_model import LinearRegression, Ridge

stats_folder = "../tf_verify/fixed_json_stats"

filter_out_perfect_data = False
filter_out_all_image = True

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

# print(len(data))
# for value in data.values():
#     print(len(value), end=", ")
# print("\b\b")
#
# print(full_image_data)

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


def check_correlation(functions_for_d, functions_for_y, file_names):
    FILE_NAMES = file_names
    best_rsquare_per_file = []

    r_squared_per_xyfunc = []

    for k, file_name in enumerate(FILE_NAMES):
        samples_in_use = data[file_name]

        # calculating the points
        ys_for_plot = [[y_func(sample) for sample in samples_in_use] for name, y_func in functions_for_y]
        xs_for_plot = [[x_func(sample) for sample in samples_in_use] for name, x_func in functions_for_d]

        table_r_squared = []
        for i, y_func_tup in enumerate(functions_for_y):
            y = ys_for_plot[i]
            lst_r_squared_xfunc = []
            for j, x_func_tup in enumerate(functions_for_d):
                x = xs_for_plot[j]
                r_squared, = gather_statistics(x, y)
                lst_r_squared_xfunc.append(r_squared)
            table_r_squared.append(lst_r_squared_xfunc)
        r_squared_per_xyfunc.append(table_r_squared)

        # print stats on interpretaions
        max_r_squared = table_r_squared[0][0]
        for row in table_r_squared:
            for entry in row:
                if entry > max_r_squared:
                    max_r_squared = entry

        # updating overall max and min
        # print(f"MAX R^2 = {max_r_squared}")
        best_rsquare_per_file.append(max_r_squared)

    # finding the best square per file
    index_of_best_file = np.argmax(best_rsquare_per_file)
    print(f"OVERALL MAX R^2 = {best_rsquare_per_file[index_of_best_file]}, file {index_of_best_file}")

    # calculating the mean r^2 of each (x_func, y_func) pair
    yx_means = []
    for i in range(len(functions_for_y)):
        x_means = []
        for j in range(len(functions_for_d)):
            mean = 0
            assert (len(FILE_NAMES) > 0)
            for k in range(len(FILE_NAMES)):
                mean += r_squared_per_xyfunc[k][i][j]
            mean /= len(FILE_NAMES)
            x_means.append(mean)
        yx_means.append(x_means)

    # printing results
    print("\nMeans of R^2 per yfunc (how to judge the 'dropdown' given an FO), xfunc (how to calculate d):")
    pretty_yx_means = [["Y \ X"] + [name for name, func in functions_for_d]] + yx_means
    for i in range(1, len(yx_means) + 1):
        pretty_yx_means[i] = [functions_for_y[i - 1][0]] + yx_means[i - 1]
    pretty_print(pretty_yx_means)

    best_mean_index = max_index_of_matrix(yx_means)
    best_mean = yx_means[best_mean_index[0]][best_mean_index[1]]

    print(
        f"\nBest mean functions: Y = {functions_for_y[best_mean_index[0]][0]}, X = {functions_for_d[best_mean_index[1]][0]}, with mean R^2 {best_mean}")


# for key, value in data.items():
#     print(key, len(value))

## ------ Visualization of data ------------- ##

# ------ Configuration ----------

# Which data we're using
# list_of_file_names_in_RELU = [str for str in list(data.keys()) if "relu" in str]
# FILE_NAME = list_of_file_names_in_RELU[3]
# print(f"NUM OF FILES IN \"RELU\": {len(list_of_file_names_in_RELU)}")


if False:
    for key in data.keys():
        visualize_d_and_y_funcs(key)


## helper functions
def get_full_image_data_from_FO(sample):
    last_dot = sample["network"].rfind(".")
    key_name = sample["network"][13:last_dot]
    key_name = f"{key_name}_all_image"
    for fo in full_image_data[key_name]:
        if (fo["image"] == sample["image"]):
            return fo
    return None


def get_dataset_network_image_string(sample):
    return f'{sample["dataset"]} -- {sample["network"]} -- {sample["image"]}'


def get_dataset_network_image_string_K(sample):
    return f'{sample["dataset"]} -- {sample["network"]} -- {sample["image"]} -- {sample["k"]}'


def generate_feature_info(func_for_d):
    """
    paramters:
        fuc_for_d: the function used to calculate d, of the format "func(sample) -> number"

    returns:
        (feature_datas, feature_names, ys, y_names)
    """

    # ------------ configuration -------------------
    file_names = [file_name for file_name in data.keys() if "relu" not in file_name]

    samples_in_use = []
    for file_name in file_names:
        samples_in_use += data[file_name]

    feature_datas = []  # list of feature_data
    feature_data_names = []  #

    ys = []
    y_names = []

    # --------------- pre-calculations ; currently calculating normalization for d -------------

    # TODO calculate the proper normalization
    # caluculating the average D in each (network, img, K) 3-tuple
    average_d_per_K = {}
    for file_name in file_names:
        current_sum = 0
        for sample in data[file_name]:
            current_sum += func_for_d(sample)
        current_sum /= len(data[file_name])
        file_name_in_sample = get_dataset_network_image_string_K(data[file_name][0])
        average_d_per_K[file_name_in_sample] = current_sum

    # caluculating the average D in each (network, img) 2-tuple
    average_d_per_img = {}
    for file_name in file_names:
        file_name_in_sample = get_dataset_network_image_string(data[file_name][0])
        current_sum = 0
        for sample in data[file_name]:
            current_sum += func_for_d(sample)
        if file_name_in_sample not in average_d_per_img:
            average_d_per_img[file_name_in_sample] = (current_sum, len(data[file_name]))
        else:
            tup = average_d_per_img[file_name_in_sample]
            average_d_per_img[file_name_in_sample] = (tup[0] + current_sum, tup[1] + len(data[file_name]))
    for key in average_d_per_img:
        average_d_per_img[key] = (average_d_per_img[key][0] / average_d_per_img[key][1])

    # -------------- Creating the datas for the features, and the ys --------------------
    first = True
    for sample in samples_in_use:
        # initializing variables that will be used
        full_image_fo = get_full_image_data_from_FO(sample)

        img_vars = sigmoid_weighted_least_squares(full_image_fo)
        k = sample["k"]
        img_d = func_for_d(sample)
        img_d_normalized_per_K = img_d / average_d_per_K[get_dataset_network_image_string_K(sample)]
        img_d_normalized_per_img = img_d / average_d_per_img[get_dataset_network_image_string(sample)]

        current_feature_index = 0
        img_b = img_vars[1]
        # creating list of datapoints and features to add to the feature list ----------- ADD HERE
        datapoints = [
            ([img_vars[0], img_vars[1], k, img_d], "a_img, b_img, k, d"),
            #([k], "k"),
            # ([img_d_normalized_per_img],                "d/sum(ds_in_img)") ,
            # ([img_d_normalized_per_K],                  "d/sum(ds_in_K)"),
            ([k, img_d_normalized_per_img],         "k, d/sum(ds_in_img)"),
            #([k, img_d_normalized_per_K],           "k, d/sum(ds_in_K)"),
            # ([img_b*784/k],                          "img_b*784/k"),
            ([img_b/k, k, img_d_normalized_per_img], "img_b/k, k, img_d_normalized_per_img"),
            #([img_b/k, k, img_d_normalized_per_img], "img_b/k, k, d/sum(ds_in_img)"),
            ([b1**i * b2**j for i in range(50) for j in range(20) for b1 in [img_d] for b2 in img_vars + [k]], "Overfit"),
            # ([k * img_vars[1]], "k * img_b"),
            # ([img_vars[1] / k], "img_b / k"),
            # ([img_vars[1] / k], "img_b / k"),
            # ([k / img_vars[1]], "k / img_b"),
            # ([img_vars[1]], "img_b"),
            # ([1 / img_vars[1]], "1 / img_b"),
            # ([np.log(img_vars[1])], "ln(img_b)")
            ## example datapoint : ([alpha_img, FO_k], "alpha_img, FO_K") ---- Dont forget to add a feature name!
        ]

        # adding the data to feature_datas
        if first:
            for k in range(len(datapoints)):
                feature_datas.append([])
                feature_data_names.append(datapoints[k][1])
        for datapoint in datapoints:
            feature_datas[current_feature_index].append(datapoint[0])
            current_feature_index += 1

        # adding the alpha to the ys
        sample_vars = sigmoid_weighted_least_squares(sample)
        a = sample_vars[0]
        b = sample_vars[1]
        img_a = img_vars[0]
        img_b = img_vars[1]
        s = 0
        if first:
            ys += [[], [], []]
            y_names += ["a", "b", "a/b"]
        ys[s].append(a)
        s += 1
        ys[s].append(b)
        s += 1
        ys[s].append(a / b)
        s += 1
        values = [4]
        for i in values:
            if first:
                ys += [[]]
                y_names += [f"-(a+{i})/b"]
            ys[s].append(-(a + i) / b);
            s += 1

        first = False
    return (feature_datas, feature_data_names, ys, y_names, samples_in_use)


# def fit_regressor_to_data(feature_info=None, func_for_d=None):
#     ## Configuration
#     if func_for_d == None:
#         func_for_d = lambda sample: d_power(sample, 6)
#         # func_for_d = lambda sample: max(sample["Ubounds"][:sample["label"]] + sample["Ubounds"][sample["label"]+1:]) - sample["Lbounds"][sample["label"]]
#
#     regressors = [LinearRegression()]
#     regressor_names = ["Linear"]
#
#     if feature_info == None:
#         feature_datas, feature_data_names, ys, y_names, fo_samples = generate_feature_info(func_for_d)
#     else:
#         feature_datas, feature_data_names, ys, y_names, fo_samples = feature_info
#
#     ## ------------------ Scaling the features -------------------
#     # feature_data = pd.DataFrame(feature_data)
#     # scaler = StandardScaler()
#     # scaler.fit(feature_data)
#     # feature_data = scaler.transform(feature_data)
#
#     ## Making the polynomial feature
#     # poly = PolynomialFeatures(4)
#     # poly_feature_data = pd.DataFrame(poly.fit_transform(feature_data))
#
#     ## --------------- Fitting and scoring the regressors ----------------------
#
#     scores = []
#     predictions = []
#     for i, regressor in enumerate(regressors):
#         scores_per_feature_data = []
#         predictions_per_feature_data = []
#         for j, feature_data in enumerate(feature_datas):
#             scores_per_y = []
#             predictions_per_y = []
#             for k, y in enumerate(ys):
#                 if len(y) == 0:
#                     continue
#                 regressor.fit(feature_data, y)
#
#                 # Calculating scores
#                 ROUNDING_PRECISION = 5
#                 prediction = regressor.predict(feature_data)
#                 score1 = regressor.score(feature_data, y)  # R^2
#                 score2 = np.sum(np.abs(prediction - y)) / len(y)  # L_1 / len(y)
#                 score3 = (np.sum((prediction - y) ** 2) ** (1 / 2)) / len(y)  # L_2 / len(y)
#
#                 temp_scores = np.array([score1, score2, score3]).round(ROUNDING_PRECISION)
#                 scores_per_y.append(list(temp_scores))
#                 predictions_per_y.append(prediction)
#             scores_per_feature_data.append(scores_per_y)
#             predictions_per_feature_data.append(predictions_per_y)
#         scores.append(scores_per_feature_data)
#         predictions.append(predictions_per_feature_data)
#
#     # calc a and b
#     y_names.append("a2")
#     y_names.append("b2")
#     ys.append(ys[y_names.index("a")])
#     ys.append(ys[y_names.index("b")])
#     for i, regressor in enumerate(regressors):
#         for j, feature in enumerate(feature_datas):
#             aDivb = predictions[i][j][y_names.index("a/b")]
#             a4Divb = predictions[i][j][y_names.index("-(a+4)/b")]
#             b = np.divide(-4, np.add(aDivb, a4Divb))
#             a = np.multiply(aDivb, b)
#             score1 = -1  # R^2
#             score2 = np.sum(np.abs(a - ys[y_names.index("a")])) / len(y)  # L_1 / len(y)
#             score3 = (np.sum((a - ys[y_names.index("a")]) ** 2) ** (1 / 2)) / len(y)  # L_2 / len(y)
#             temp_scores = np.array([score1, score2, score3]).round(ROUNDING_PRECISION)
#             scores[i][j].append(temp_scores)
#             score1 = -1  # R^2
#             score2 = np.sum(np.abs(b - ys[y_names.index("b")])) / len(y)  # L_1 / len(y)
#             score3 = (np.sum((b - ys[y_names.index("b")]) ** 2) ** (1 / 2)) / len(y)  # L_2 / len(y)
#             temp_scores = np.array([score1, score2, score3]).round(ROUNDING_PRECISION)
#             scores[i][j].append(temp_scores)
#
#     # printing out the results
#     print("--- Fitting model to (alpha_img, beta_img, K, d) ---")
#     print("Training data scores:")
#     for i in range(len(regressors)):
#         regressor_name = regressor_names[i] if len(regressor_names) > i else "REGRESSOR NOT NAMED"
#         for j in range(len(feature_datas)):
#             feature_names = feature_data_names[j] if len(feature_data_names) > j else "FEATURE NOT NAMED"
#             print(f"______________________________________________\n")
#             print(f"Features:\t[{feature_names}]")
#             print(f"Regressor:\t{regressor_names[i]}")
#             print(f"Scores:\t\t[R^2, L_1, L_2]\n")
#             matrix = [[f"-> {y_names[r]}", scores[i][j][r]] for r in range(len(ys))]
#             pretty_print(matrix)

def get_all_fnr_sigmoid(fo_samples, **kwargs_for_weights_calc):
    alphas = []
    betas = []
    for fo in fo_samples:
        img_fo = get_full_image_data_from_FO(fo)
        alpha, beta = get_fnr_sigmoid(img_fo, fo["k"], **kwargs_for_weights_calc)
        alphas.append(alpha)
        betas.append(beta)
    return alphas, betas


def fit_regressor_to_data(feature_info=None, func_for_d=None):
    # Configuration
    if func_for_d is None:
        func_for_d = lambda sample: d_power(sample, 6)
        # func_for_d = lambda sample: max(sample["Ubounds"][:sample["label"]] + sample["Ubounds"][sample["label"]+1:]) - sample["Lbounds"][sample["label"]]

    regressors = [LinearRegression()]
    regressor_names = ["Linear"]

    if feature_info is None:
        feature_datas, feature_data_names, ys, y_names, fo_samples = generate_feature_info(func_for_d)
    else:
        feature_datas, feature_data_names, ys, y_names, fo_samples = feature_info

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
    predictions = []
    for i, regressor in enumerate(regressors):
        scores_per_feature_data = []
        predictions_per_feature_data = []
        for j, feature_data in enumerate(feature_datas):
            scores_per_y = []
            predictions_per_y = []
            for k, y in enumerate(ys):
                if len(y) == 0:
                    continue
                regressor.fit(feature_data, y)

                # Calculating scores
                ROUNDING_PRECISION = 5
                prediction = regressor.predict(feature_data)
                score1 = regressor.score(feature_data, y)  # R^2
                score2 = np.sum(np.abs(prediction - y)) / len(y)  # L_1 / len(y)
                score3 = (np.sum((prediction - y) ** 2) ** (1 / 2)) / len(y)  # L_2 / len(y)

                temp_scores = np.array([score1, score2, score3]).round(ROUNDING_PRECISION)
                scores_per_y.append(list(temp_scores))
                predictions_per_y.append(prediction)
            scores_per_feature_data.append(scores_per_y)
            predictions_per_feature_data.append(predictions_per_y)
        scores.append(scores_per_feature_data)
        predictions.append(predictions_per_feature_data)

        # printing out the results
        print("--- Fitting model to (alpha_img, beta_img, K, d) ---")
        print("Training data scores:")
        for i in range(len(regressors)):
            regressor_name = regressor_names[i] if len(regressor_names) > i else "REGRESSOR NOT NAMED"
            for j in range(len(feature_datas)):
                feature_names = feature_data_names[j] if len(feature_data_names) > j else "FEATURE NOT NAMED"
                print(f"______________________________________________\n")
                print(f"Features:\t[{feature_names}]")
                print(f"Regressor:\t{regressor_names[i]}")
                print(f"Scores:\t\t[R^2, L_1, L_2]\n")
                matrix = [[f"-> {y_names[r]}", scores[i][j][r]] for r in range(len(ys))]
                pretty_print(matrix)

    # ------------------ getting a, b -------------------
    ab_formulas = [{"x1_name": "a", "x2_name": "b", "comment": "a, b from regressor"},
                   {"x1_name": "a/b", "x2_name": "-(a+4)/b",
                    "a_func": lambda aDivb, a4Divb: (-4 * aDivb) / (aDivb + a4Divb),
                    "b_func": lambda aDivb, a4Divb: (-4) / (aDivb + a4Divb)},
                   # {"x1_name": "-(a+6)/b", "x2_name": "-(a+4)/b",
                   #  "a_func": lambda x1, x2: (4*x1 - 6*x2)/(x2 - x1),
                   #  "b_func": lambda x1, x2: -((4*x1 - 6*x2)/(x2 - x1) + 4) / x2},
                   {"x1_name": "a/b", "x2_name": "b",
                    "a_func": lambda x1, x2: x1 * x2},
                   ]
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

    # ------------------ printing out ab results -------------------
    print("evaluating a, b")
    for regressor in ab.keys():
        for feature in ab[regressor].keys():
            print("---------------------------------------------------")
            print("Regressor: " + regressor + ", Feature: " + feature)
            matrix = [["", *ab[regressor][feature][0]["scores"].keys()]]
            matrix += [[ab_formula["name"], *[format(score, ".4f") for score in ab_formula["scores"].values()]] for ab_formula in ab[regressor][feature]]
            matrix += [["best: ", *[format(min([ab_formula["scores"][name] for ab_formula in ab[regressor][feature]]), ".4f") for name in ab[regressor][feature][0]["scores"].keys()]]]
            pretty_print(matrix)


fit_regressor_to_data()

# file_names = data.keys()
# file_names = [file_name for file_name in data.keys() if ("" in file_name and "" in file_name)]
# SIG_functions_for_y = [
#     ("alpha_of_FO", lambda sample: (sigmoid_weighted_least_squares(sample))[0]),
#     ("beta_of_FO", lambda sample: (sigmoid_weighted_least_squares(sample))[1]),
#     ("alpha_over_beta",
#      lambda sample: (sigmoid_weighted_least_squares(sample))[0] / (sigmoid_weighted_least_squares(sample))[1]),
#     ("alpha+1_over_beta",
#      lambda sample: ((sigmoid_weighted_least_squares(sample))[0] + 1) / (sigmoid_weighted_least_squares(sample))[1]),
#     ("alpha+2_over_beta",
#      lambda sample: ((sigmoid_weighted_least_squares(sample))[0] + 2) / (sigmoid_weighted_least_squares(sample))[1]),
#     ("alpha+5_over_beta",
#      lambda sample: ((sigmoid_weighted_least_squares(sample))[0] + 5) / (sigmoid_weighted_least_squares(sample))[1]),
# ]
# visualize_K_and_y_funcs(SIG_functions_for_y, file_names)
#
# for file_name in file_names:
#     visualize_d_and_y_funcs(functions_for_d, SIG_functions_for_y, [file_name])
#
# check_correlation(functions_for_d, SIG_functions_for_y, file_names)
