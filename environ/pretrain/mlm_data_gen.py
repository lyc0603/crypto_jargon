"""
Functions to generate the dataset for MLM training
"""

import collections
import gc
import json
import shelve
from typing import Callable

import jieba
import numpy as np
from bert4torch.tokenizers import Tokenizer

from environ.constants import MAX_FILE_NUM, MAXLEN, PROCESSED_DATA_PATH

# Initialize jieba
jieba.initialize()
jieba.load_userdict(f"{PROCESSED_DATA_PATH}/new_words.txt")


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

        instances = []

        for text in texts:
            instance = [[start] for start in starts]
            # process single sentence
            sub_instance = self.sentence_process(text)
            sub_instance = [i[:self.sequence_length - 2] for i in sub_instance]

            # sample extension
            for item, sub_item in zip(instance, sub_instance):
                item.extend(sub_item)

            # insert end and padding
            complete_instance = []
            for item, end, pad in zip(instance, ends, paddings):
                item.append(end)
                item = self.padding(item, pad)
                complete_instance.append(item)
            # store the result and reset the instance
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
                if count == 10:
                    yield texts
                    texts, count = [], 0

    if texts:
        yield texts


def word_segment(text: str):
    """
    Word segmentation
    """

    return jieba.lcut(text)
