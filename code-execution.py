# Load data
data_tensor = torch.tensor([x[1] for x in seq_id_one_hot_code], dtype=torch.float32)#.to(device)
ids = [x[0] for x in seq_id_one_hot_code]
labels_tensor = torch.tensor(labels, dtype=torch.long)#.to(device)
padding_encoding = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
mask = torch.all(data_tensor == padding_encoding, dim=2)
print(data_tensor.shape, labels_tensor.shape)

# Create dataset and dataloaders
#dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor, mask,ids)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, masks, ids):
        self.data = data
        self.labels = labels
        self.masks = masks
        self.ids = ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.masks[idx], self.ids[idx]
dataset = CustomDataset(data_tensor, labels_tensor, mask, ids)
total_size = len(dataset)
train_size = int(0.7 * total_size)
validation_size = int(0.1 * total_size)
test_size = total_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=500, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=True, num_workers=0)

# Calculate class weights for handling imbalanced datasets
class_counts = Counter(labels_tensor.tolist())
total_samples = len(labels_tensor)
class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
#weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Set preparation done!")

# Initialize model, optimizer, and loss function
model = RNASequenceClassifier().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

#criterion = nn.CrossEntropyLoss(weight=weights_tensor)
def custom_loss(outputs, labels, mask):
    # Apply the mask to filter out the padded areas from the loss calculation
    # Let's assume mask is 1 for valid data and 0 for padded data
    mask = mask.any(dim=1)
    masked_outputs = outputs[mask]
    masked_labels = labels[mask]
    return nn.CrossEntropyLoss()(masked_outputs, masked_labels)

print("Running the model")
# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement
best_val_accuracy = 0
epochs_no_improve = 0
# Training loop
model_path = Path('results_accuracy_100.txt')
model_path.write_text('')
losses = list()
accuracies = list()
val_losses = list()
val_accuracies = list()
with open(model_path, 'a') as fh:
    for epoch in range(100):
        epoch_losses = list()
        epoch_accuracies = list()
        model.train()  # Set the model to training mode
        for inputs, labels, mask, ids in train_loader:
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            inputs = inputs.permute(0, 2, 1)  # Ensure correct input dimensions for convolutions
            outputs = model(inputs)
            loss = custom_loss(outputs, labels, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            epoch_accuracies.append(accuracy)

        epoch_loss = np.mean(epoch_losses)
        epoch_accuracy = np.mean(epoch_accuracies)
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            res = ''
            val_epoch_losses = list()
            val_epoch_accuracies = list()
            for inputs, labels, mask, ids in validation_loader:
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
                inputs = inputs.permute(0, 2, 1)
                outputs = model(inputs)
                val_loss = custom_loss(outputs, labels, mask)

                val_epoch_losses.append(val_loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_accuracy = (predicted == labels).sum().item() / labels.size(0)
                val_epoch_accuracies.append(val_accuracy)

                val_epoch_loss = np.mean(val_epoch_losses)
                val_epoch_accuracy = np.mean(val_epoch_accuracies)
                val_losses.append(val_epoch_loss)
                val_accuracies.append(val_epoch_accuracy)

                for id, true_label, predicted_label in zip(ids, labels, predicted):
                    res += f"{id}, {true_label.item()}, {predicted_label.item()}\n"

        result = f'Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Val_Loss: {val_epoch_loss}, Val_Accuracy: {val_epoch_accuracy}\n'
        print(result.strip())
        fh.write(result)
        fh.write(res)
         # Check for improvement
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
torch.save(model, '/home/alekchiri/model_3_kernels_100_after.pth')
