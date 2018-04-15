import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="This script shows training graph from history file.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input history h5 file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input

    df = pd.read_hdf(input_path, "history")
    print(df)

    # plot loss function
    input_dir = os.path.dirname(input_path)
    plt.plot(df["loss"], label="loss")
    plt.plot(df["val_loss"], label="val_loss")
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "loss.png"))
    plt.cla()

    # plot accuracy
    plt.plot(df["acc"], label="accuracy")
    plt.plot(df["val_acc"], label="val_accuracy")
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "accuracy.png"))

main()