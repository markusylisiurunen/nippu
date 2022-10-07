import io
import re
import math
import base64

from PIL import Image

import torch
import numpy as np
import transformers

from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv2ForTokenClassification, LayoutXLMTokenizerFast, LayoutLMv2FeatureExtractor, LayoutXLMProcessor


labels = ["ADDRESS", "COMPANY", "DATE", "TOTAL", "TOTAL_TAX_24", "TOTAL_TAX_14", "TOTAL_TAX_10"]

label_to_index = { "O": 0 }

for i, label in enumerate(labels):
    label_to_index[f"B-{label}"] = 2 * i + 1
    label_to_index[f"I-{label}"] = 2 * (i + 1)

index_to_label = { index: label for label, index in label_to_index.items() }

label_to_index_fn = lambda label: label_to_index[label]
index_to_label_fn = lambda idx: index_to_label[idx]

num_labels = len(label_to_index.keys())


class ReceiptDataset(Dataset):
    def __init__(self, sample_loaders, processor, label_to_index):
        self.sample_loaders = sample_loaders
        self.processor = processor
        self.label_to_index = label_to_index

        # init the sample IDs to an array of (sample_loader_idx, List[string]) tuples
        self.sample_ids = []

        for i, sample_loader in enumerate(sample_loaders):
            ids = sample_loader.ids()
            self.sample_ids.append((i, ids))

    def __len__(self):
        total = 0

        for _, ids in self.sample_ids:
            total += len(ids)

        return total

    def __getitem__(self, idx):
        encoding = self.encoding(idx)
        # FIXME: the batch dimension should be removed by passing the `prepend_batch_axis` as `False`
        return { key: value.squeeze() for key, value in encoding.items() }

    def encoding(self, idx):
        # find the correct (sample_loader, sample_id) tuple
        sample_cum_sum = 0

        sample_loader = None
        sample_id = None

        for i, ids in self.sample_ids:
            total_cum_sum = sample_cum_sum + len(ids)

            if idx < total_cum_sum:
                # this sample loader is the one for the index
                sample_loader = self.sample_loaders[i]
                sample_id = ids[idx - sample_cum_sum]

                break
            
            sample_cum_sum = total_cum_sum
            
        if sample_loader is None:
            raise "sample loader not found"

        # retrieve the sample and run it through the processor
        sample = sample_loader.sample(sample_id, with_image=True)

        image = sample["image"]
        width, height = image.size

        words = sample["words"]
        boxes = [self._normalize_bbox(bbox, width, height) for bbox in sample["boxes"]]
        word_labels = [self.label_to_index(label) for label in sample["labels"]]

        return self.processor(image, words, boxes=boxes, word_labels=word_labels,
            # we add padding (and truncate) to the max length to be able to use batches
            max_length=512,

            truncation=True,
            padding=transformers.file_utils.PaddingStrategy.MAX_LENGTH,
            
            # the input is already split into words and not a full sequence of text
            is_split_into_words=True,
            
            # we want pytorch tensors back
            return_tensors="pt",
        )

    def _normalize_bbox(self, bbox, width, height):
        # normalize every bounding box to the range of 0-1000
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]


class CustomSampleLoader:
    def __init__(self, image, blocks):
        self.image = image
        self.blocks = blocks

    def ids(self):
        return ["1"]

    def sample(self, id, with_image=False):
        words = list(map(lambda x: x["Text"], self.blocks))
        size = self.image.size

        boxes = []

        for block in self.blocks:
            bbox = block["Geometry"]["BoundingBox"]
            x, y, width, height = bbox["Left"], bbox["Top"], bbox["Width"], bbox["Height"]

            # convert the coordinates to absolute values to be usable by the processor
            boxes.append((
                x * size[0],
                y * size[1],
                (x + width) * size[0],
                (y + height) * size[1],
            ))

        result = { "words": words, "boxes": boxes, "labels": ["O"] * len(boxes) }

        if with_image:
            result["image"] = self.image

        return result


def tokens_to_entity_candidates(tokens, labels):
    # TODO: it would probably be worth to explore to assign entities at the word-level and not at the token-level.
    #       now, it's possible that the entities contain partial words which is of course not ideal.
    
    entity_names = ["company", "address", "date", "total", "total_tax_24", "total_tax_14", "total_tax_10"]
    entity_candidates = { name: [] for name in entity_names }

    context = None

    for i, token in enumerate(tokens):
        label = labels[i]                           # e.g. B-COMPANY
        entity_name = label.split("-")[-1].lower()  # e.g. company

        if label == "O" or entity_name not in entity_names:
            context = None
            continue

        is_start = label.startswith("B-")

        if is_start:
            # if we are instantly continuing from the previous segment (i.e. B-X, I-X, B-X, I-X, ...)
            is_same_segment = context == entity_name
            
            if is_same_segment:
                # we will consider these intra-segment B- labels as if they were I-
                entity_candidates[entity_name][-1].append(token)
            else:
                entity_candidates[entity_name].append([token])

            context = entity_name
            continue

        if context == entity_name:
            # we will continue the last segment
            entity_candidates[entity_name][-1].append(token)
        else:
            # otherwise, we just started a new segment with a non-start label... ignore (at least for now)
            context = None

    return entity_candidates


def select_longest_candidate(candidates):
    result = ""

    for candidate in candidates:
        if len(candidate) > len(result):
            result = candidate

    return result


def select_date_candidate(candidates):
    date_regex = "(\d{1,4}[\-\./]\d{1,4}[\-\./]\d{1,4})"

    for candidate in candidates:
        result = re.search(date_regex, candidate)
        
        if result:
            return result.group(0)
    
    return select_longest_candidate(candidates)


def select_amount_candidate(candidates):
    date_regex = "(\d+[\.,]\d{1,2})"

    for candidate in candidates:
        result = re.search(date_regex, candidate)
        
        if result:
            return result.group(0)
    
    return select_longest_candidate(candidates)


def select_candidate(entity, candidates):
    if len(candidates) == 0:
        return ""

    if entity == "date":
        return select_date_candidate(candidates)
    
    if entity.startswith("total"):
        return select_amount_candidate(candidates)
    
    return select_longest_candidate(candidates)


def prediction_to_entities(encoding, prediction, tokenizer):
    # convert the prediction to a list of label strings
    pred_labels = prediction.argmax(axis=-1).tolist()
    pred_labels = [index_to_label[idx] for idx in pred_labels]
    
    # we can now extract the entity candidates
    tokens = encoding['input_ids'][0].tolist()
    candidates = tokens_to_entity_candidates(tokens, pred_labels)
    
    # convert the candidate input ID sequences to text
    for key in candidates.keys():
        for i in range(len(candidates[key])):
            candidates[key][i] = tokenizer.decode(candidates[key][i], skip_special_tokens=True)

    # select the best candidate for every entity (and clean up the predictions)
    candidates = { key: select_candidate(key, candidates[key]).strip() for key in candidates.keys() }
    
    return candidates


def model_fn(model_dir):
    tokenizer = LayoutXLMTokenizerFast.from_pretrained("microsoft/layoutxlm-base")
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_dir, num_labels=num_labels)
    return model, tokenizer


def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    processor = LayoutXLMProcessor(LayoutLMv2FeatureExtractor(apply_ocr=False), tokenizer)
    
    # take the inputs (from { inputs: { base64_image: "", blocks: [] } }
    base64_image, blocks = data["inputs"]["base64_image"], data["inputs"]["blocks"]
    
    # process the image
    bytes_image = base64.b64decode(base64_image.encode('ascii'))
    image = Image.open(io.BytesIO(bytes_image))
    
    # init the dataloader
    sample_loader = CustomSampleLoader(image, blocks)
    receipt_dataset = ReceiptDataset([sample_loader], processor, label_to_index_fn)
    receipt_dataloader = DataLoader(receipt_dataset, batch_size=1)

    # run the input through the model
    batch = next(iter(receipt_dataloader))

    model.eval()
    with torch.no_grad():
        batch_out = model(**batch)
        
    # capture the predictions
    encoding = receipt_dataset.encoding(0)
    prediction = batch_out.logits.squeeze()

    entities = prediction_to_entities(encoding, prediction, tokenizer)

    return {"result": entities}
