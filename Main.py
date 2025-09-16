from preprocessing import prepare_data
from model import LSTMTagger
from train import train
import torch
from tqdm import tqdm

def evaluate(model, test_loader, idx_to_tag):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_predictions = []
    all_true_tags = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, colour='BLUE', desc="Evaluating")
        for sentences, tags in progress_bar:
            sentences, tags = sentences.to(device), tags.to(device)

            predictions = model(sentences)
            _, predicted_tags = torch.max(predictions, dim=2)

            mask = tags != 0
            correct += (predicted_tags[mask] == tags[mask]).sum().item()
            total += mask.sum().item()

            all_predictions.extend(predicted_tags[mask].cpu().tolist())
            all_true_tags.extend(tags[mask].cpu().tolist())

    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f}")
    # print("\nSample Predictions vs True Tags:")
    # for i in range(min(5, len(all_predictions))):  # In ra 5 cặp đầu tiên
    #     pred_tag = idx_to_tag[all_predictions[i]]
    #     true_tag = idx_to_tag[all_true_tags[i]]
    #     print(f"Predicted: {pred_tag}, True: {true_tag}")

if __name__ == "__main__":
    train_loader, val_loader, test_loader, word_to_idx, tag_to_idx, idx_to_tag = prepare_data()

    vocab_size = len(word_to_idx)
    embedding_dim = 100
    hidden_dim = 128
    tagset_size = len(tag_to_idx)

    model = LSTMTagger(vocab_size, embedding_dim, hidden_dim, tagset_size)

    trained_model = train(model, train_loader, val_loader)

    evaluate(trained_model, test_loader, idx_to_tag)