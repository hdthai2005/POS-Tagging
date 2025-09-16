import torch.cuda
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train(model, train_dataloader, val_dataloader, epochs = 100, lr = 0.01, momentum = 0.9):
	optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
	criterion = nn.CrossEntropyLoss(ignore_index=0)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	for epoch in range(epochs):
		model.train()
		train_loss = []
		progress_bar = tqdm(train_dataloader, colour = 'GREEN')
		for iter, (sentences, tags) in enumerate(train_dataloader):
			sentences, tags = sentences.to(device), tags.to(device)
			# Forward
			prediction = model(sentences)
			# print(tag_scores_prediction.shape)
			loss = criterion(prediction.transpose(1, 2), tags)
			# Backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())
			progress_bar.set_description('Epoch : {}/{}. Loss : {:0.4f}'.format(epoch + 1, epochs, np.mean(train_loss)))

		val_losses = []
		correct = 0
		total = 0
		model.eval()
		with torch.no_grad():
			progress_bar = tqdm(val_dataloader, colour = 'YELLOW')
			for iter, (sentences, tags) in enumerate(val_dataloader):
				sentences, tags = sentences.to(device), tags.to(device)
				prediction = model(sentences)
				loss = criterion(prediction.transpose(1, 2), tags)
				val_losses.append(loss.item())

				_, predicted_tags = torch.max(prediction, 2)
				mask = (tags != 0)
				correct += (predicted_tags[mask] == tags[mask]).sum().item()
				total += mask.sum().item()

		val_accuracy = correct / total
		print('Epoch : {}/{}. Loss : {:0.4f}. Val_Accuracy : {}'.format(epoch + 1, epochs, np.mean(val_losses), val_accuracy))
	return model