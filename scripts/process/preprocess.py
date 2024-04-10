"""
Script for data cleaning
"""

import re

import pandas as pd
from environ.constants import DATA_PATH, PROCESSED_DATA_PATH


def rm_exotic(df):
    """
    Delete Korean and Japanese and just remain Chinese
    # 韩文unicode \uac00-\ud7ff
    # 日文片假名unicode \u30a0-\u30ff
    # 日文平假名 \u3040-\u309f

    """

    p = u"[\uac00-\ud7ff]|[\u30a0-\u30ff]|[\u3040-\u309f]"
    regex = re.compile(p)
    df = df[(True ^ df['corpus'].str.contains(regex))]

    # 基本汉字unicode范围
    p2 = u"[\u4E00-\u9FA5]"
    regex2 = re.compile(p2)
    df = df[df['corpus'].str.contains(regex2)]
    return df


def sub_url(s):
    """
    Delete web links
    """
    s = re.sub(r"\s+", "", s)
    # print(s)
    # 删除http网址
    s = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_…@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", s)
    # 删除pic网址
    s = re.sub(
        r'pic(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", s)
    return s


def rm_dup(df):
    """
    Remove duplicate traces
    """
    df = df.drop_duplicates("processed")
    return df


def subLong(s):
    """
    Delete long English or numbers, but remain short ones for they might be jargons
    """
    result_list = re.findall('[a-zA-Z0]+', s)
    for r in result_list:
        if len(r) >= 6:
            s = re.sub(r, "", s)
    # print(s)
    return s


if __name__ == '__main__':
    """
    input: Raw data
    output: Clean data
    """

    raw_data = f"/home/yichen/crypto_jargon/data/weibo.csv"  # read from raw data
    df1 = pd.read_csv(raw_data, encoding="utf8")
    df1["corpus"] = df1["text"]
    df = rm_exotic(df1)
    corpus_df = df["corpus"]

    corpus_df = corpus_df.apply(lambda x: re.sub(r'#\w+#', '', x)) # remove tag between ##
    corpus_df = corpus_df.apply(lambda x: str(x).strip())  # delete spaces
    corpus_df = corpus_df.apply(lambda x: sub_url(x))  # delete urls
    corpus_df = corpus_df.apply(lambda x: re.sub(
        re.compile('</?\w+[^>]*>'), "", x))
    corpus_df = corpus_df.apply(lambda x: re.sub(re.compile('_'), "", x))
    corpus_df = corpus_df.apply(lambda x: re.sub(
        re.compile('[^\w]'), "", x))  # delete symbols

    # remove tags
    corpus_df = corpus_df.apply(lambda x: re.sub(r'#\w+#', '', x))
    
    # remove supertalk
    corpus_df = corpus_df.apply(lambda x: re.sub(r'.*超话', '', x))

    # remove 微博正文
    corpus_df = corpus_df.apply(lambda x: re.sub(r'微博正文', '', x))

    # remove 网页链接
    corpus_df = corpus_df.apply(lambda x: re.sub(r'网页链接', '', x))

    # remove numbers
    corpus_df = corpus_df.apply(lambda x: re.sub(re.compile('[0-9]+'), "0", x))

    corpus_df = corpus_df.apply(lambda x: re.sub(re.compile(
        '([a-zA-Z]+)'), lambda s: s.group(1).upper(), x))  # to upper case
    corpus_df = corpus_df.apply(lambda x: subLong(x))

    new_df = pd.DataFrame()
    new_df["processed"] = corpus_df
    new_df = rm_dup(new_df)

    # save to csv
    new_df.to_csv("/home/yichen/crypto_jargon/processed_data/processed.csv", encoding="utf8") 

    # save to txt
    with open("/home/yichen/crypto_jargon/processed_data/processed.txt", "w", encoding="utf8") as f:
        for i in new_df["processed"]:
            f.write(i + "\n")