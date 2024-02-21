import matplotlib.pyplot as plt
import numpy as np


def position_breakdown(index, columns):
    return (int(np.floor(index / columns)), int(index % columns))


# ------ Visualization ----------
def visualize_d_and_y_funcs(x_funcs, y_funcs, file_names):
    GRID_PRESENTATION = True
    COLUMNS = 3

    if GRID_PRESENTATION:
        num_rows = int(np.ceil(len(y_funcs) / COLUMNS))
        fig, axs = plt.subplots(num_rows, COLUMNS, figsize=(15, 5 * num_rows))
        # plt.subplots_adjust(wspace=1)

    # the data
    samples_in_use = []
    for file_name in file_names:
        samples_in_use += data[file_name]
    ys_for_plot = [[y_func(sample) for sample in samples_in_use] for name, y_func in y_funcs]
    xs_for_plot = [[x_func(sample) for sample in samples_in_use] for name, x_func in x_funcs]
    # print(f"{FILE_NAME=}")
    # print(f"NUMBER OF SAMPLES = {len(samples_in_use)}")

    # plotting the graphs
    table_r_squared = []
    for i, func_tuple_for_y in enumerate(y_funcs):
        y_func_name, y_func = func_tuple_for_y

        # making "my_plotter" the thing we plot on
        if GRID_PRESENTATION:
            if num_rows > 1:
                my_plotter = axs[
                    position_breakdown(i, COLUMNS)[0], position_breakdown(i, COLUMNS)[1]]  # yes I tried unpacking
            else:
                my_plotter = axs[position_breakdown(i, COLUMNS)[1]]
            my_plotter.set_title(f"{y_func_name}")
        else:
            fig, my_plotter = plt.subplots()

        labels = []
        lst_r_squared_xfunc = []

        # calculating the ys
        regular_y = ys_for_plot[i]
        for j, color, x_func_tup in zip(range(1000), mcolors.BASE_COLORS, x_funcs):
            # finding the x and the ys
            x_func_name, x_func = x_func_tup
            # x = [x_func(sample) for sample in SAMPLES_IN_USE]
            x = xs_for_plot[j]
            # y_for_visualization = [y+j*0.1 for y in regular_y]
            y_for_visualization = regular_y

            # gather statistics
            r_squared, = gather_statistics(x, regular_y)  # currently just calculates r^2
            r_squared = round(r_squared, 4)
            lst_r_squared_xfunc.append(
                (r_squared, f"X: {x_func_name}, Y: {y_func_name}"))  # TODO !! CHANGE TO ONLY NUMBER

            # visualize
            my_plotter.scatter(x, y_for_visualization, color=color, s=15)
            labels.append(f"R^2 = {r_squared} | {x_func_name}")

        # updating the r^2 table
        table_r_squared.append(lst_r_squared_xfunc)

        # set up plot
        my_plotter.set_xlabel('d')
        my_plotter.set_ylabel(f'y_func = {y_func_name}')
        my_plotter.legend(labels, ncol=1, loc='center left',
                          bbox_to_anchor=[0.73, 0.9],
                          columnspacing=1.0, labelspacing=0.0,
                          handletextpad=0.0, handlelength=1.5,
                          fancybox=True, shadow=True,
                          fontsize=10)

        if not GRID_PRESENTATION:
            plt.show()

    fig.suptitle(f"File: {file_name}", fontsize=30)
    plt.show()
