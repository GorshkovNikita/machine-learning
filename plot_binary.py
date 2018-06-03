import matplotlib.pyplot as plt


def plot_binary(inputs, results):
    x_val = [v[0] for v in inputs]
    y_val = [v[1] for v in inputs]
    colors = ['r' if r == 1 else 'b' for r in results]
    print(colors)
    plt.scatter(x_val, y_val, c=colors)
    # plt.show()


def plot_binary_on_plt(plt, inputs, results):
    x_val = [v[0] for v in inputs]
    y_val = [v[1] for v in inputs]
    colors = ['r' if r == 1 else 'b' for r in results]
    print(colors)
    plt.scatter(x_val, y_val, c=colors)


if __name__ == '__main__':
    plot_binary([[1, 2], [10, 14]], [0, 1])