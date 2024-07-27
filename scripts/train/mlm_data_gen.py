"""
Script to generate the dataset for MLM training
"""

import collections
import gc
import json
import os
import shelve
import time
from typing import Callable

import jieba
import numpy as np
from bert4torch.tokenizers import Tokenizer, load_vocab

from environ.constants import DATA_PATH, MAXLEN, PROCESSED_DATA_PATH

# Initialize jieba
jieba.initialize()
jieba.load_userdict(f"{PROCESSED_DATA_PATH}/new_words.txt")

# Basic parameters
EPOCHS = 500
NUM_WORDS = 61711


class TrainingDataset:
    """
    Pretrain data generator
    """

    def __init__(self, tokenizer: Tokenizer, sequence_length=256):
        """
        Parameters:
            tokenizer: bert4torch Tokenizer object
            sequence_length: int, the max length of the sequence
        """

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_start_id
        self.token_sep_id = tokenizer._token_end_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocal_size = tokenizer._vocab_size

    def padding(self, sequence: list, padding_value: None):
        """
        Use 0 to pad the sequence to the same length
        """

        if padding_value is None:
            padding_value = self.token_pad_id

        sequence = sequence[: self.sequence_length]
        padding_length = self.sequence_length - len(sequence)
        return sequence + [padding_value] * padding_length

    def sentence_process(self, text: str):
        """
        Process the sentence
        """

        raise NotImplementedError

    def paragraph_process(
        self, texts: list[str], starts: list[int], ends: list[int], paddings: list[int]
    ):
        """
        Process the list of sentences
        texts is the list of single text
        starts is the list of start index of the text
        ends is the list of end index of the text
        padding is the list of padding value
        """

        instances, instance = [], [[start] for start in starts]

        for text in texts:
            # process single sentence
            sub_instance = self.sentence_process(text)
            sub_instance = [i[:self.sequence_length - 2] for i in sub_instance]
            new_length = len(instance[0]) + len(sub_instance[0])

            # if the new length is about to overflow
            if new_length > self.sequence_length - 1:
                # insert end and padding
                complete_instance = []
                for item, end, pad in zip(instance, ends, paddings):
                    item.append(end)
                    item = self.padding(item, pad)
                    complete_instance.append(item)
                # store the result and reset the instance
                instances.append(complete_instance)
                instance = [[start] for start in starts]

            # sample extension
            for item, sub_item in zip(instance, sub_instance):
                item.extend(sub_item)

        # insert end and padding
        complete_instance = []
        for item, end, pad in zip(instance, ends, paddings):
            item.append(end)
            item = self.padding(item, pad)
            complete_instance.append(item)

        # store the final instance
        instances.append(complete_instance)

        return instances

    def serialize(self, instances: list, db: shelve.Shelf, count: int):
        """
        Serialize the instances
        """

        for instance in instances:
            input_ids, masked_lm_labels = instance[0], instance[1]
            assert len(input_ids) <= self.sequence_length
            features = collections.OrderedDict()
            features["input_ids"] = input_ids
            features["masked_lm_labels"] = masked_lm_labels
            db[str(count)] = features
            count += 1

        return count

    def process(self, corpus: list, record_name: str):
        """
        Process the corpus
        """

        count = 0

        db = shelve.open(record_name)
        for texts in corpus:
            instances = self.paragraph_process(texts)
            count = self.serialize(instances, db, count)

        db.close()
        del instances
        gc.collect()

        # record the information
        record_info = {"filename": record_name, "samples_num": count}
        json.dump(record_info, open(record_name + ".json", "w", encoding="utf-8"))

        print('write %s examples into %s' % (count, record_name))


class TrainingDatasetRoBerta(TrainingDataset):
    """
    Pretrain data generator for RoBerta
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        word_segment: Callable,
        mask_rate=0.15,
        sequence_length=256,
    ):
        """
        Parameters:
            tokenizer: bert4torch Tokenizer object
            sequence_length: int, the max length of the sequence
        """

        super(TrainingDatasetRoBerta, self).__init__(tokenizer, sequence_length)
        self.word_segment = word_segment
        self.mask_rate = mask_rate

    def token_process(self, token_id: int):
        """
        Process the token
        80% to mask, 10% to original, 10% to random
        """

        rand = np.random.random()
        if rand < 0.8:
            return self.token_mask_id
        elif rand < 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocal_size)

    def sentence_process(self, text: str):
        """
        Method to process the single sentence

        Workflow: word segmentation, masking

        return the list of token ids
        """

        words = self.word_segment(text)
        rands = np.random.random(len(words))

        token_ids, mask_ids = [], []
        for rand, word in zip(rands, words):
            word_tokens = self.tokenizer.tokenize(text=word)[1:-1]
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)

            if rand < self.mask_rate:
                word_mask_ids = [self.token_process(i) for i in word_token_ids]
                token_ids.extend(word_mask_ids)
                mask_ids.extend(word_token_ids)             
            else:
                token_ids.extend(word_token_ids)
                word_mask_ids = [0] * len(word_tokens)
                mask_ids.extend(word_mask_ids)

        return [token_ids, mask_ids]

    def paragraph_process(self, texts: list[str]):
        """
        Add starts, ends, and padding for the original paragraph_process
        """

        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        padding = [self.token_pad_id, 0]
        return super(TrainingDatasetRoBerta, self).paragraph_process(
            texts, starts, ends, padding
        )


def text_segmentate(
    text: str, max_len: int, seps: str = "\n", strips: str | None = None
):
    """
    Text segmentation
    """

    text = text.strip().strip(strips)
    if seps and len(text) > max_len:
        pieces = text.split(seps[0])
        text, texts = "", []
        for i, p in enumerate(pieces):
            if text and len(text) + len(p) > max_len - 1:
                texts.extend(text_segmentate(text, max_len, seps[1:], strips))
                text = ""
            if i + 1 == len(pieces):
                text += p
            else:
                text = text + p + seps[0]

        if text:
            texts.extend(text_segmentate(text, max_len, seps[1:], strips))
        return texts
    else:
        return [text]


def text_process(text: str):
    """
    Text processing
    """

    texts = text_segmentate(text, 23, "\n。")
    result = ""
    for text in texts:
        if result and len(result) + len(text) > MAXLEN * 1.3:
            yield result
            result = ""
        result += text
    if result:
        yield result

    return texts


def corpus():
    """
    corpus importer
    """

    files = f"{PROCESSED_DATA_PATH}/processed.txt"
    count, texts = 0, []

    with open(files, "r") as f:
        for l in f:
            for text in text_process(l):
                texts.append(text)
                count += 1
                if count == MAX_FILE_NUM:
                    yield texts
                    texts, count = [], 0

    if texts:
        yield texts


def word_segment(text: str):
    """
    Word segmentation
    """

    return jieba.lcut(text)


if __name__ == "__main__":
    MAX_FILE_NUM = 40
    dir_training_data = f"{PROCESSED_DATA_PATH}/pretrain/dataset"

    # tokenizer
    if os.path.exists(f"{DATA_PATH}/RoBERTa-wwm-ext/tokenizer_config.json"):
        token_dict, compound_tokens = json.load(
            open(f"{DATA_PATH}/RoBERTa-wwm-ext/tokenizer_config.json")
        )
    else:
        # load and simplify vocab
        token_dict = load_vocab(
            dict_path=f"{PROCESSED_DATA_PATH}/pretrain/dataset",
            simplified=False,
            startswith=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )
        pure_tokenizer = Tokenizer(token_dict.copy(), do_lower_case=True)
        # add all unique words from corpus to ./v2/vocab.txt
        words = [
            line.strip()
            for line in open(
                f"{PROCESSED_DATA_PATH}/vocab.txt", "r", encoding="utf-8"
            ).readlines()
        ]
        print(words)
        user_dict = []
        for w in words:
            if w not in token_dict:
                token_dict[w] = len(token_dict)
                user_dict.append(w)
            if len(user_dict) == NUM_WORDS:
                break
        compound_tokens = [pure_tokenizer.encode(w)[0][1:-1] for w in user_dict]
        json.dump(
            [token_dict, compound_tokens],
            open(f"{DATA_PATH}/RoBERTa-wwm-ext/tokenizer_config.json", "w"),
        )

    tokenizer = Tokenizer(
        token_dict,
        do_lower_case=True,
    )

    # dataset
    TD = TrainingDatasetRoBerta(tokenizer, word_segment, sequence_length=MAXLEN)

    while True:
        train_file = [
            file
            for file in os.listdir(dir_training_data)
            if ("train_" in file) and ("dat" in file)
        ]

        if len(train_file) < MAX_FILE_NUM:
            record_name = f'{dir_training_data}/train_'+ time.strftime('%Y%m%d%H%M%S', time.localtime())
            TD.process(corpus=corpus(), record_name=record_name)
            time.sleep(1)
            print(f"Generate {record_name}")
        else:
            time.sleep(300)
