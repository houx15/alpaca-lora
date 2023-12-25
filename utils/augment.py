import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm
from utils.utils import sentence_cleaner
import platform


def dataset_augmentation(
    df, src_type="tweet", target_labels=None, target_sizes=None
):
    """
    Dataset Augmentation with repo nlpaug(https://github.com/makcedward/nlpaug)
    Balance different labels with data augmentation
    :param DataFrame df: pandas Dataframe with `text` and `label`
    :param str src_type, Optional, tweet by default
    :param list target_label, Optional, None by default, list of labels to be augmented, must be used with target_size
    :param list target_sizes, Optional, None by default, list of sizes to be augmented, must be used with target_label
    :return a class-balanced pandas Dataframe  with `text` and `label`
    """

    print(">>>>performing dataset augmentation")
    print(src_type)

    df_count = df["label"].value_counts(ascending=False)
    print(df_count, type(df_count))

    skip_rows = []

    indexes = []
    update_size = {}

    if target_labels and target_sizes:
        indexes = target_labels
        for i, label in enumerate(target_labels):
            target_size = target_sizes[i]
            row_num = df_count[label]
            update_size[label] = {
                "quotient": int(target_size / row_num),
                "remainder": target_size % row_num,
            }
            print("update", update_size)
    else:
        target_size = 0
        for index, row in df_count.items():
            print(index, row)
            indexes.append(index)
            if target_size < row:
                target_size = row
                skip_rows.append(index)
            update_size[index] = {
                "quotient": int(target_size / row),
                "remainder": target_size % row,
            }
            print("update", update_size)

    model_path = "bert-base-uncased"
    if src_type == "weibo":
        model_path = "bert-base-chinese"
    if platform.node().startswith("della"):
        model_path = "/home/junmingh/virus/model-transformers/" + model_path
    print(">>model path", model_path)
    aug = naw.ContextualWordEmbsAug(model_path=model_path, action="substitute")

    aug_dict = {"label": [], "text": []}
    for index in indexes:
        print(">>>processing label {}".format(index))
        if index in skip_rows:
            # skip max rows
            continue

        index_df = df[df["label"] == index]
        sampled_df = index_df.sample(n=update_size[index]["remainder"])
        remained_df = index_df[~index_df.index.isin(sampled_df.index)]
        aug_n = update_size[index]["quotient"]
        for text in tqdm(sampled_df["text"]):
            text = sentence_cleaner(text)
            augmented_texts = aug.augment(text, n=aug_n)
            for atext in augmented_texts:
                aug_dict["label"].append(index)
                aug_dict["text"].append(atext)
        if aug_n - 1 > 0:
            for text in tqdm(remained_df["text"]):
                text = sentence_cleaner(text)
                augmented_texts = aug.augment(text, n=aug_n - 1)
                for atext in augmented_texts:
                    aug_dict["label"].append(index)
                    aug_dict["text"].append(atext)
    aug_df = pd.DataFrame(aug_dict)
    new_df = pd.concat([df, aug_df])
    print(new_df)
    df_count = new_df["label"].value_counts(ascending=False)
    print(df_count, type(df_count))
    return new_df


def bert_augment(text, aug_num, src_type="tweet"):
    model_path = "bert-base-uncased"
    if src_type == "weibo":
        model_path = "bert-base-chinese"
    print(">>model path", model_path)
    aug = naw.ContextualWordEmbsAug(model_path=model_path, action="substitute")
    augmented_texts = aug.augment(text, n=aug_num)
    return augmented_texts


def syn_augment(text, aug_num, src_type="tweet"):
    aug = naw.SynonymAug(
        aug_src="wordnet", lang="eng" if src_type == "tweet" else "cmn"
    )
    augmented_texts = []
    for i in range(aug_num):
        augmented_texts.append(aug.augment(text))
    return augmented_texts
