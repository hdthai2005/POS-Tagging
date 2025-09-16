import nltk
from nltk.corpus import brown, treebank, conll2000
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


# nltk.download('treebank')
# nltk.download('brown')
# nltk.download('conll2000')
# nltk.download('universal_tagset')
# nltk.download('punkt')
# nltk.download('punkt_tab')

class POSDataset(Dataset):
	def __init__(self, sentences, tags, word_to_idx, tag_to_idx):
		self.sentences = sentences
		self.tags = tags
		self.word_to_idx = word_to_idx
		self.tag_to_idx = tag_to_idx

	def __len__(self):
		return len(self.sentences)

	def __getitem__(self, idx):
		sentence = self.sentences[idx]
		tags = self.tags[idx]
		sentence_idx = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in sentence]
		tags_idx = [self.tag_to_idx[tag] for tag in tags]
		return torch.tensor(sentence_idx), torch.tensor(tags_idx)


def prepare_data(train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, batch_size=32):
	treebank_sents = treebank.tagged_sents(tagset='universal')
	brown_sents = brown.tagged_sents(tagset='universal')
	conll2000_sents = conll2000.tagged_sents(tagset='universal')

	all_sents = treebank_sents + brown_sents + conll2000_sents
	sentences = []
	sentence_tags = []
	for sent in all_sents:
		words, tags = zip(*sent)
		sentences.append(words)
		sentence_tags.append(tags)

	tokenizer = nltk.tokenize.word_tokenize
	tokenizer_sentences = []
	tokenizer_tags = []
	for sentence, tags in zip(sentences, sentence_tags):
		tokenizer_sentence = tokenizer(" ".join(sentence))
		tokenizer_tag = tags
		if (len(tokenizer_sentence) != len(tokenizer_tag)):
			min_len = min(len(tokenizer_sentence), len(tokenizer_tag))
			tokenizer_sentence = tokenizer_sentence[:min_len]
			tokenizer_tag = tokenizer_tag[:min_len]
		tokenizer_sentences.append(tokenizer_sentence)
		tokenizer_tags.append(tokenizer_tag)

	word_to_idx = defaultdict(lambda: len(word_to_idx))
	word_to_idx['<PAD>'] = 0
	word_to_idx['<UNK>'] = 1
	tag_to_idx = defaultdict(lambda: len(tag_to_idx))
	tag_to_idx['<PAD>'] = 0
	tag_to_idx['<UNK>'] = 1
	# print(tokenizer_sentences[0])

	for sentence in tokenizer_sentences:
		for word in sentence:
			_ = word_to_idx[word]
	for tags in tokenizer_tags:
		for tag in tags:
			_ = tag_to_idx[tag]

	word_to_idx = dict(word_to_idx)
	tag_to_idx = dict(tag_to_idx)
	idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
	x_train, x_test, y_train, y_test = train_test_split(tokenizer_sentences, tokenizer_tags, test_size=1 - train_ratio,
	                                                    random_state=42)
	x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + val_ratio),
	                                                random_state=42)

	train_dataset = POSDataset(x_train, y_train, word_to_idx, tag_to_idx)
	val_dataset = POSDataset(x_val, y_val, word_to_idx, tag_to_idx)
	test_dataset = POSDataset(x_test, y_test, word_to_idx, tag_to_idx)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=padding)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=padding)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=padding)

	return train_dataloader, val_dataloader, test_dataloader, word_to_idx, tag_to_idx, idx_to_tag


def padding(batch):
	sentences, tags = zip(*batch)
	sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
	tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
	return sentences_padded, tags_padded


if __name__ == '__main__':
	train_loader, val_loader, test_loader, word_to_idx, tag_to_idx, idx_to_tag = prepare_data()
	# batch = next(iter(train_loader))
	# print(batch[1][1])
	# sentences_batch = batch[0].tolist()
	# tags_batch = batch[1].tolist()
	# sentences_words = [[list(word_to_idx.keys())[list(word_to_idx.values()).index(idx)] for idx in sentence] for
	#                    sentence in sentences_batch]
	# tags_labels = [[idx_to_tag[idx] for idx in tag_sequence] for tag_sequence in tags_batch]
	#
	# print(sentences_words)
	# print(tags_labels)
	print(idx_to_tag)