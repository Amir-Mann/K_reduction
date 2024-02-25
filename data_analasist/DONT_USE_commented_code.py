# def fit_regressor_to_data(feature_info=None, func_for_d=None):
#     ## Configuration
#     if func_for_d == None:
#         func_for_d = lambda sample: d_power(sample, 6)
#         # func_for_d = lambda sample: max(sample["Ubounds"][:sample["label"]] + sample["Ubounds"][sample["label"]+1:])
#                       - sample["Lbounds"][sample["label"]]
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

    print(f"\nBest mean functions: Y = {functions_for_y[best_mean_index[0]][0]}, X = {functions_for_d[best_mean_index[1]][0]}, with mean R^2 {best_mean}")
