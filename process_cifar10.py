import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--df_path', type=str, default="./")
    parser.add_argument('--batch_files_path', type=str, default='./cifar-10-batches-py')
    parser.add_argument('--export_path', type=str, default='./cifar10')
    return parser

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifarTo_png(export_path, batch_files_path, df, classes_name, test=False, percent=100):
    
    if test:
        print("Starting test")
        d = unpickle(batch_files_path+"/test_batch")
        names = d[b'filenames']
        labels = d[b'labels']
        data = d[b'data']
        for name, label, dat in (zip(names, labels, data)):
            img = dat.reshape(3,32,32).transpose(1,2,0)
            name = name.decode("utf-8")
            plt.imsave(f"{export_path}/test/{classes_name[label]}/{name}", img)
        print("Done")
    else:
        print("Starting train")
        for j in range(1,6):
            d = unpickle(f"{batch_files_path}/data_batch_{j}")
            print(f"Batch {j}, number of images : {len(d[b'labels'])}")
            for i in range(len(d[b'labels'])):
                if df["name"].str.contains(d[b'filenames'][i].decode("utf-8")).any():
                    img = d[b'data'][i].reshape(3,32,32).transpose(1,2,0)
                    name = d[b'filenames'][i].decode("utf-8")
                    plt.imsave(f"{export_path}/train_{percent}/{classes_name[d[b'labels'][i]]}/{name}", img)
        print("Done")

def main(args):
    
    df_train_100 = pd.read_csv(args.df_path + "/Cifar10_train_split_100.csv")
    df_train_10 = pd.read_csv(args.df_path + "/Cifar10_train_split_10.csv")
    df_train_1 = pd.read_csv(args.df_path + "/Cifar10_train_split_1.csv")
    df_name_test = pd.read_csv(args.df_path + "/Cifar10_test.csv")


    classes_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #Create the folder of classes_name in args.export_path
    for i in range(len(classes_name)):
        os.makedirs(args.export_path + "/train_100/" + classes_name[i], exist_ok=True)
        os.makedirs(args.export_path + "/test/" + classes_name[i], exist_ok=True)
        os.makedirs(args.export_path + "/train_10/" + classes_name[i], exist_ok=True)
        os.makedirs(args.export_path + "/train_1/" + classes_name[i], exist_ok=True)



    cifarTo_png(args.export_path,args.batch_files_path, df_name_test, classes_name,test=True)
    cifarTo_png(args.export_path,args.batch_files_path, df_train_100, classes_name,percent=100)
    cifarTo_png(args.export_path,args.batch_files_path, df_train_10, classes_name,percent=10)
    cifarTo_png(args.export_path,args.batch_files_path, df_train_1, classes_name,percent=1)


if __name__=="__main__":
    parser = argparse.ArgumentParser('CIFAR10 processing script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
