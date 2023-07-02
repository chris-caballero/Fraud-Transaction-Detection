import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

from multiprocessing import freeze_support

'''
#############################################
MODEL
#############################################
'''

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        
        self.dense1 = nn.Linear(input_size, hidden_size)
        # self.dense2 = nn.Linear(hidden_size, hidden_size//2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        # x = self.dense2(x)
        # x = F.relu(x)
        output = F.sigmoid(self.classifier(x))
        
        return output

def train(model, dataloader, epochs=5, verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters())
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            # Forward pass
            inputs, labels = batch['Inputs'].to(device), batch['Class'].to(device)
            # Convert true labels to one-hot encoding
            # labels = nn.functional.one_hot(labels, num_classes=2).type(torch.float32).to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            print(f'Epoch {epoch} Complete\n- Loss: {loss.item()}')
        
    return model

def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            inputs, labels = batch['Inputs'].to(device), batch['Class'].to(device)

            outputs = model(inputs)
            
            predictions = torch.argmax(outputs, dim=1).to(device)

            correct += (predictions == labels).sum().item()
            total += len(labels)
            
    accuracy = correct / total
         
    return accuracy

'''
#############################################
DATA
#############################################
'''
from torch.utils.data import Dataset, DataLoader

class TransactionDataset(Dataset):
    def __init__(self, df):
        data = df.drop(['Class'], 1)
        if 'Class Name' in data.columns:
            data = data.drop(['Class Name'], 1)
        if 'Class Dist' in data.columns:
            data = data.drop(['Class Dist'], 1)
        self.data = data.values
        self.labels = df['Class'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'Inputs': torch.tensor(self.data[idx], dtype=torch.float32),
            'Class': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_dataloader(dataset, batch_size=16):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )

def to_dataloader(dataset, batch_size=16, val_split=0.2):
    from torch.utils.data import random_split

    if val_split >= 1 or val_split <= 0:
        return create_dataloader(dataset, batch_size=batch_size)

    # Define the length of the dataset and the sizes of the training and testing sets
    train_size = int(len(dataset) * (1 - val_split))
    test_size = len(dataset) - train_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders for the training and testing sets
    train = create_dataloader(train_dataset, batch_size=batch_size)
    test = create_dataloader(test_dataset, batch_size=batch_size)

    return train, test, train_dataset, test_dataset
'''
#############################################
SCRIPT
#############################################
'''


def class_distribution(df):
    fraud = round(len(df.loc[df['Class'] == 1]) / len(df) * 100, 2)
    nonfraud = round(len(df.loc[df['Class'] == 0]) / len(df) * 100, 2)
    print('% dataset with class 0: {}%\n% dataset with class 1: {}%\n'.format(nonfraud, fraud))
    

if __name__ == '__main__':
    freeze_support() 
    dataset_file = 'data/creditcard.csv'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    df = pd.read_csv(dataset_file)
    
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    X = df.drop(labels=['Class'], axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    train_df = pd.concat([X_train, y_train], axis=1)
    holdout_df = pd.concat([X_test, y_test], axis=1)

    new_df = train_df.groupby('Class', group_keys=False).apply(lambda x: x.sample(len(train_df[train_df.Class == 1])))

    dataset = TransactionDataset(df=new_df)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    trainset, testset, _, test_dataset = to_dataloader(dataset, batch_size=16)
    
    # for batch in testset:
    #     print(batch)
    #     break
    
    model = NeuralNet(input_size=len(new_df.columns)-1, hidden_size=8, num_classes=2).to(device)
    
    train(model, trainset, epochs=10)
    print('Accuracy:', evaluate(model, testset))
    
    X = test_dataset[:]['Inputs']
    y = test_dataset[:]['Class']
    
    y_pred = model(X)
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)

    acc, precision, recall, roc = accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred)

    scores = [acc, precision, recall, roc]
    scores = [round(score, 4) for score in scores]
    scores_df = pd.DataFrame([scores], columns=['Accuracy', 'Precision', 'Recall', 'AUC-ROC'])
    
    print(scores_df)
    
    _confusion_matrix = confusion_matrix(y, y_pred)

    # Maps the confusion matrix so each row is a distribution for that row. Easier to visualize.
    # confusion_matrix_ = confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1).reshape(2, 1)
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.heatmap(_confusion_matrix, annot=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix - Neural Net')
    
    plt.savefig('confusion-matrix-nn2.png')
