"""
author: Hou Yuxin
date: 2022-06-16
"""

import pandas as pd
import json
import os

from utils.augment import dataset_augmentation
from sklearn.model_selection import train_test_split


class DataProcess(object):
    """
    Base class for data process.
    """

    def __init__(
        self,
        dataset_dir: str,
        dataset_files: list,
        task_type: str,
        src_type: str = "tweet",
        force_update: bool = False,
        augmentation: bool = False,
        split_ratio: list = [0.8, 0.1, 0.1],
        augment_args: dict = None,
        labeller: str = None,
        undersampling_strategy: dict = None,
        oversample_files: list = [],
    ) -> None:
        """
        dataset_dir: str, path of dataset directory
        dataset_files: list, within which each element is the filename of a datafile
        task_type: str, regression or binary
        force_update: bool, when it equals to False, program would use train/validate/test-{task_type}.csv as default datafile
        split_ratio: list, we split the dataset to train, validate, test dataset according to the ratio. Each item should be float and the sum of them should equal to 1
        """
        # task_map = {
        #     "ternary": "binary",
        #     "regression": "regression",
        #     "binary": "binary"
        # }

        self.dataset_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), dataset_dir
        )
        self.dataset_files = dataset_files
        self.task_type = task_type  # task_map[task_type]
        self.src_type = src_type
        self.force_update = force_update
        self.augmentation = augmentation
        self.augment_args = augment_args
        self.split_ratio = split_ratio
        self.labeller = labeller
        self.undersampling_strategy = undersampling_strategy
        self.oversample_files = oversample_files

        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "configs/data_configuration.json",
            ),
            "r",
            encoding="utf8",
        ) as rfile:
            self.data_configuration = json.loads(rfile.read())

    def single_file_handler(self, file_name):
        """
        Read single csv file and convert it to  pandas DataFrame
        """
        file_conf = self.data_configuration.get(
            file_name,
            {
                "text": "text",
                "label": "label",
                "label_strategy": {"binary": {}, "regression": {}, "regression_with_relevance": {}},
            },
        )
        if self.labeller:
            file_conf["label"] = self.labeller
        print(file_conf)

        # print("Warning: no data configuration provided, default configuration used")

        columns = [file_conf["text"], file_conf["label"]]
        csv_name = self.dataset_dir + file_name + ".csv"
        df = pd.read_csv(csv_name, header=0, usecols=columns)
        df = df.dropna(axis=0, how="any")

        df = df.rename(columns={columns[0]: "text", columns[1]: "label"})

        skip_labels = []
        replaced_labels = []
        new_labels = []

        cur_strategy = file_conf["label_strategy"][self.task_type]
        for k, v in cur_strategy.items():
            new_key = k
            new_key = new_key.strip("-")
            if new_key.isdecimal():
                k = int(k)
            if v == "skip":
                skip_labels.append(k)
                continue
            replaced_labels.append(k)
            new_labels.append(v)

        for skip_label in skip_labels:
            df = df[df["label"] != skip_label]
        df = df.replace(replaced_labels, new_labels)

        print(df.label.values)

        # TODO
        # df = df.sample(n=200)

        return df

    def undersampling(self, df, strategy: dict):
        """undersampling

        Args:
            df (Dataframe): the dataframe you want to undersample
            strategy (dict): key - label; value - ratio, a value between 0 and 1, referring to the ratio you want to sample
        """
        for label, ratio in strategy.items():
            sample_out = df[df["label"] == label].sample(
                frac=1 - ratio, random_state=1
            )
            df = df[~df.index.isin(sample_out.index)]
        return df

    def split_df_by_ratio(self, df):
        if sum(self.split_ratio) != 1:
            raise ValueError("Error: the sum of split ratio is larger than 1")
        if len(self.split_ratio) < 3:
            raise ValueError(
                "Error: split_ratio doesn't provide the ratio of train, validate and test dataset"
            )
        print("type", df, type(df))

        train_df = df.sample(frac=self.split_ratio[0])
        remained_df = df[~df.index.isin(train_df.index)]

        validate_df = remained_df.sample(
            frac=self.split_ratio[1]
            / (self.split_ratio[1] + self.split_ratio[2])
        )
        test_df = remained_df[~remained_df.index.isin(validate_df.index)]

        return train_df, validate_df, test_df

    def split_df_by_label(self, df):
        """
        Code from https://stackoverflow.com/questions/50781562/stratified-splitting-of-pandas-dataframe-into-training-validation-and-test-set
        Splits a Pandas dataframe into three subsets (train, val, and test)
        following fractional ratios provided by the user, where each subset is
        stratified by the values in a specific column (that is, each subset has
        the same relative frequency of the values in the column). It performs this
        splitting by running train_test_split() twice.

        Parameters
        ----------
        df_input : Pandas dataframe
            Input dataframe to be split.
        stratify_colname : str
            The name of the column that will be used for stratification. Usually
            this column would be for the label.
        frac_train : float
        frac_val   : float
        frac_test  : float
            The ratios with which the dataframe will be split into train, val, and
            test data. The values should be expressed as float fractions and should
            sum to 1.0.
        random_state : int, None, or RandomStateInstance
            Value to be passed to train_test_split().

        Returns
        -------
        df_train, df_val, df_test :
            Dataframes containing the three splits.
        """

        if sum(self.split_ratio) != 1:
            raise ValueError("Error: the sum of split ratio is larger than 1")
        if len(self.split_ratio) < 3:
            raise ValueError(
                "Error: split_ratio doesn't provide the ratio of train, validate and test dataset"
            )
        print("type", df, type(df))

        X = df  # Contains all columns.
        y = df[["label"]]  # Dataframe of just the column on which to stratify.

        frac_train, frac_val, frac_test = self.split_ratio

        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = train_test_split(
            X, y, stratify=y, test_size=(1.0 - frac_train), random_state=None
        )

        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)

        if relative_frac_test > 0:
            df_val, df_test, y_val, y_test = train_test_split(
                df_temp,
                y_temp,
                stratify=y_temp,
                test_size=relative_frac_test,
                random_state=None,
            )
        else:
            df_val = df_temp
            df_test = pd.DataFrame(columns=df_val.columns)

        assert len(df) == len(df_train) + len(df_val) + len(df_test)

        return df_train, df_val, df_test

    def save_df(self, train_df, validate_df, test_df, total_df):
        train_df.to_csv(
            self.dataset_dir + "train-{}.csv".format(self.task_type)
        )
        validate_df.to_csv(
            self.dataset_dir + "validate-{}.csv".format(self.task_type)
        )
        test_df.to_csv(self.dataset_dir + "test-{}.csv".format(self.task_type))
        total_df.to_csv(
            self.dataset_dir + "total-{}.csv".format(self.task_type)
        )
        print("saving finished")
        return True

    def get_cached_df(self):
        try:
            train_df = pd.read_csv(
                self.dataset_dir + "train-{}.csv".format(self.task_type)
            )
            validate_df = pd.read_csv(
                self.dataset_dir + "validate-{}.csv".format(self.task_type)
            )
            test_df = pd.read_csv(
                self.dataset_dir + "test-{}.csv".format(self.task_type)
            )
            total_df = pd.concat([train_df, validate_df, test_df])
            return train_df, validate_df, test_df, total_df
        except:
            return None, None, None, None

    def get_dataset(self):
        """
        Return a 3-item tuple
        (train_dataset, validate_dataset, test_dataset)
        Each element is a pandas DataFrame
        """
        train_df, validate_df, test_df, dataset_df = None, None, None, None
        if self.force_update is False:
            train_df, validate_df, test_df, dataset_df = self.get_cached_df()

        if train_df is None:
            df_list = []
            train_only_list = []
            for single_file in self.dataset_files:
                single_df = self.single_file_handler(single_file)
                if single_file not in self.oversample_files:
                    df_list.append(single_df)
                else:
                    train_only_list.append(single_df)

            dataset_df = pd.concat(df_list)
            train_df, validate_df, test_df = self.split_df_by_label(dataset_df)
            train_only_list.append(train_df)
            train_df = pd.concat(train_only_list)
        if self.undersampling_strategy:
            train_df = self.undersampling(
                train_df, self.undersampling_strategy
            )
        if self.augmentation:
            if self.augment_args is None:
                # augment train dataset to an even distribution by default
                train_df = dataset_augmentation(train_df, self.src_type)
            else:
                if "train" in self.augment_args.keys():
                    train_df = dataset_augmentation(
                        train_df,
                        self.src_type,
                        target_labels=self.augment_args["train"][
                            "target_labels"
                        ],
                        target_sizes=self.augment_args["train"][
                            "target_sizes"
                        ],
                    )
                if "eval" in self.augment_args.keys():
                    validate_df = dataset_augmentation(
                        validate_df,
                        self.src_type,
                        target_labels=self.augment_args["eval"][
                            "target_labels"
                        ],
                        target_sizes=self.augment_args["eval"]["target_sizes"],
                    )
                if "test" in self.augment_args.keys():
                    test_df = dataset_augmentation(
                        test_df,
                        self.src_type,
                        target_labels=self.augment_args["test"][
                            "target_labels"
                        ],
                        target_sizes=self.augment_args["test"]["target_sizes"],
                    )
            dataset_df = pd.concat([train_df, validate_df, test_df])

        train_df = train_df[["text", "label"]]
        validate_df = validate_df[["text", "label"]]
        test_df = test_df[["text", "label"]]
        dataset_df = dataset_df[["text", "label"]]
        self.save_df(train_df, validate_df, test_df, dataset_df)

        return train_df, validate_df, test_df, dataset_df
