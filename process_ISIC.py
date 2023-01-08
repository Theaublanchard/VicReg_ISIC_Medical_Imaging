import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm
import zipfile


def get_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--export_path', type=str, default='./ISIC')
    parser.add_argument('--zip_path', type=str, default='./ISIC_2019_Training_Input.zip')
    parser.add_argument('--csv_path', type=str, default='./')
    parser.add_argument('--split', type=int, default=100,choices=[1,10,100])
    parser.add_argument('--test', type=bool, default=False)

    return parser

def export_zip(zip_path, export_path,df):
    # for c in classes_to_idx:
    #     os.makedirs(os.path.join(export_path, c), exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        name_file = zip_ref.namelist()[2:-1]

        for i,_ in tqdm(enumerate(df["image"]), total=len(df)):
            export_file_path = os.path.join(export_path, df["class"][i])
            zip_ref.extract(name_file[i], export_file_path)

def main(args):
    train_100 = pd.read_csv(args.csv_path+"ISIC_2019_train_split_100.csv")
    train_1 = pd.read_csv(args.csv_path+"ISIC_2019_train_split_1.csv")
    train_10 = pd.read_csv(args.csv_path+"ISIC_2019_train_split_10.csv")
    test = pd.read_csv(args.csv_path+"ISIC_2019_test_split.csv")

    if args.test:
        export_zip(args.zip_path, args.export_path+str("/test"),test)

    if args.split == 100:
        export_zip(args.zip_path, args.export_path+str("/train_100"),train_100)
    elif args.split == 1:
        export_zip(args.zip_path, args.export_path+str("/train_1"),train_1)
    elif args.split == 10:
        export_zip(args.zip_path, args.export_path+str("/train_10"),train_10)

if __name__=="__main__":
    parser = argparse.ArgumentParser('ISIC processing script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
