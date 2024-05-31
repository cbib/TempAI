# Loading data
data_tensor = torch.tensor(seq_one_hot_code, dtype=torch.float32)#.to(device)
labels_tensor = torch.tensor(labels, dtype=torch.long)#.to(device)
print(data_tensor.shape, labels_tensor.shape)

# Creating dataset and dataloaders
dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
total_size = len(dataset)
train_size = int(0.7 * total_size)
validation_size = int(0.1 * total_size)
test_size = total_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1100, shuffle=True, num_workers=0)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1100, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1100, shuffle=True, num_workers=0)

# Calculating class weights for handling imbalanced datasets
class_counts = Counter(labels_tensor.tolist())
total_samples = len(labels_tensor)
class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Set preparation done!")

# Initialize model, optimizer, and loss function
model = RNASequenceClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

print("Running the model")
# Early stopping parameters
patience = 5  
best_val_accuracy = 0
epochs_no_improve = 0
# Training loop
model_path = Path('results_accuracy_100.txt')
model_path.write_text('')
losses = []
accuracies = []
val_losses = []
val_accuracies = []
with open(model_path, 'a') as fh:
    for epoch in range(50):
        epoch_losses = []
        epoch_accuracies = []
        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)  # Ensure correct input dimensions for convolutions
            outputs = model(inputs)
            loss = criterion(outputs, labels)

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
        with torch.no_grad():  
            val_epoch_losses = []
            val_epoch_accuracies = []
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.permute(0, 2, 1)  
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                val_epoch_losses.append(val_loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_accuracy = (predicted == labels).sum().item() / labels.size(0)
                val_epoch_accuracies.append(val_accuracy)

            val_epoch_loss = np.mean(val_epoch_losses)
            val_epoch_accuracy = np.mean(val_epoch_accuracies)
            val_losses.append(val_epoch_loss)
            val_accuracies.append(val_epoch_accuracy)

        result = f'Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Val_Loss: {val_epoch_loss}, Val_Accuracy: {val_epoch_accuracy}\n'
        print(result.strip())
        fh.write(result)
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
