import torch.nn as nn

class LSTMTagger(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, dropout = 0.5):
		super(LSTMTagger, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
		self.dropout = nn.Dropout(dropout)
		self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

	def forward(self, x):
		embeds = self.embedding(x)
		lstm_out, _ = self.lstm(embeds)
		lstm_out = self.dropout(lstm_out)
		tag_space = self.hidden2tag(lstm_out)
		return tag_space