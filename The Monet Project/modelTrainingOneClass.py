import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision import models
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import os
import csv
import matplotlib.pyplot as plt
import config

def plot_training_metrics(metrics, save_dir):
    epochs = [x[0] for x in metrics]
    train_losses = [x[1] for x in metrics]
    train_accuracies = [x[2] for x in metrics]
    val_losses = [x[3] for x in metrics]
    val_accuracies = [x[4] for x in metrics]

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def print_confusion_matrix_with_labels(cm):
    """
    Print confusion matrix with clear labels for one-class classification
    
    Confusion Matrix Structure:
    [TN  FP]    [non-Monet predicted correctly    non-Monet predicted as Monet]
    [FN  TP]    [Monet predicted as non-Monet     Monet predicted correctly]
    """
    print("\nConfusion Matrix Breakdown:")
    print(f"True Negatives (non-Monet correctly identified): {cm[0][0]}")
    print(f"False Positives (non-Monet wrongly identified as Monet): {cm[0][1]}")
    print(f"False Negatives (Monet wrongly identified as non-Monet): {cm[1][0]}")
    print(f"True Positives (Monet correctly identified): {cm[1][1]}")

class MonetOneClassClassifier:
    def __init__(self, learning_rate=config.LEARNING_RATE):  # Use learning rate from config
        # Initialize ResNet-101 with pretrained weights
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
        # Modify final layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),  # Light dropout to prevent overfitting
            nn.Linear(num_ftrs, 2)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Standard CrossEntropyLoss without class weights to maintain class imbalance
        self.criterion = nn.CrossEntropyLoss()
        
        # Basic Adam optimizer without weight decay to allow focus on Monet features
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.1,
            patience=3
        )

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        predictions = []
        labels = []
        
        for images, batch_labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, batch_labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_f1 = f1_score(labels, predictions, pos_label=1)
        epoch_acc = np.mean(np.array(predictions) == np.array(labels))
        
        cm = confusion_matrix(labels, predictions)
        print("\nTraining Confusion Matrix:")
        print_confusion_matrix_with_labels(cm)
        
        return epoch_loss, epoch_acc, epoch_f1

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        predictions = []
        labels = []
        
        with torch.no_grad():
            for images, batch_labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, batch_labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        
        val_loss = running_loss / len(val_loader.dataset)
        val_f1 = f1_score(labels, predictions, pos_label=1)
        val_acc = np.mean(np.array(predictions) == np.array(labels))
        
        cm = confusion_matrix(labels, predictions)
        print("\nValidation Confusion Matrix:")
        print_confusion_matrix_with_labels(cm)
        
        # Print predictions and actual labels for debugging
        print("\nValidation Predictions vs Actual Labels:")
        print(f"Predictions: {''.join(map(str, predictions))}")
        print(f"Actual Labels: {''.join(map(str, labels))}")
        
        return val_loss, val_acc, val_f1

    def train(self, train_loader, val_loader, save_dir=os.path.dirname(config.RESULTS_PATH)):
        os.makedirs(save_dir, exist_ok=True)
        results = []
        best_val_f1 = 0
        patience_counter = 0
        early_stopping_patience = 5
        
        for epoch in range(config.NUM_EPOCHS):  # Use NUM_EPOCHS from config
            print(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS}')
            
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc, val_f1 = self.validate(val_loader)
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f'Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1 (Monet class): {train_f1:.4f}')
            print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 (Monet class): {val_f1:.4f}')
            
            results.append([epoch + 1, train_loss, train_acc, val_loss, val_acc])
            
            # Save best model based on validation F1 score for Monet class
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), config.MODEL_PATH)  # Use MODEL_PATH from config
                patience_counter = 0
                print(f"New best model saved! (Monet F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Save and plot results
        with open(config.RESULTS_PATH, 'w', newline='') as f:  # Use RESULTS_PATH from config
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'])
            writer.writerows(results)
        
        plot_training_metrics(results, save_dir)
        return results

if __name__ == "__main__":
    from data_preparation import load_balanced_data
    
    # Load the balanced data
    train_loader, val_loader, _ = load_balanced_data(
        monet_dir=config.MONET_DATA_DIR,
        non_monet_dir=config.NON_MONET_DATA_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    # Initialize and train the classifier
    classifier = MonetOneClassClassifier()
    classifier.train(train_loader, val_loader)