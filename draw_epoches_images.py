import argparse

import matplotlib.pyplot as plt


def get_data_list(model_name):
    count = 0
    psnr = loss = list()
    for line in open("results/results_{}.txt".format(model_name), "r"):
        if count == 0:
            psnr = line.rstrip(" \n").split(" ")
            psnr = [float(num) for num in psnr]
        else:
            loss = line.rstrip(" \n").split(" ")
            loss = [float(num) for num in loss]
        count += 1
    return psnr, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help="The datapoints will be printed if True.")
    parser.add_argument('--save-image',
                        default=True,
                        help="Graphs will be saved if True.")

    args = parser.parse_args()

    psnr_srcnn, loss_srcnn = get_data_list("SRCNN")
    psnr_simple, loss_simple = get_data_list("SIMPLENET")
    psnr_vdsr, loss_vdsr = get_data_list("VDSR")

    x1 = x2 = [0, 50, 100, 150, 200, 250, 300]
    y1 = [psnr_srcnn[0]]
    y2 = [psnr_simple[0]]
    y3 = [psnr_vdsr[0]]
    z1 = [loss_srcnn[0]]
    z2 = [loss_simple[0]]
    z3 = [loss_vdsr[0]]

    for i in range(1, 300):
        if (i + 1) % 50 == 0:
            y1.append(psnr_srcnn[i])
            y2.append(psnr_simple[i])
            y3.append(psnr_vdsr[i])
            z1.append(loss_srcnn[i])
            z2.append(loss_simple[i])
            z3.append(loss_vdsr[i])

    if args.save_image:
        print(x1)
        print(y1)
        print(y2)
        print(y3)
        print(z1)
        print(z2)
        print(z3)

    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.plot(x1, y3)
    plt.legend(["SRCNN", "SIMPLENET", "VDSR"])
    plt.title('PSNR value of the best results of three models after 300 epoches')
    plt.ylabel('psnr')
    plt.xlabel('epochs')

    if args.save_image:
        plt.savefig("results/graph_PSNR.png")

    plt.show()

    plt.plot(x1, z1)
    plt.plot(x1, z2)
    plt.plot(x1, z3)
    plt.legend(["SRCNN", "SIMPLENET", "VDSR"])
    plt.title('loss of three models after 300 epoches (MSE)')
    plt.xlabel('epochs')

    if args.save_image:
        plt.savefig("results/graph_loss.png")

    plt.show()
