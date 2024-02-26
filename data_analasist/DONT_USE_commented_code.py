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


