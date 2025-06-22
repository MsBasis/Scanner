#Better late than never
import torch 
import torch.nn as nn 
import torch.optim as optim
from Data_Loader import dara_loaders

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
        
    def forward(self,x):
        return self.net(x)


def training_arc(csv, epochs=20, batch_size=128, lr=0.001, save = "C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\modelScanner.pt"):
    train, test, input_dim = dara_loaders(csv, batch_size=batch_size)
    
    model = MLP(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    #actual traini
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for X_batch, y_batch in train:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            
        avgLoss = run_loss / len(train)
        print(f"[Epoka {epoch:02d}] Średni loss: {avgLoss:.6f}")  

    evaluation(model,test,criterion)
    torch.save(model.state_dict(), save)
    return model

def evaluation(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)
    print(f"\nTest MSE: {avg_test_loss:.6f}")
    return avg_test_loss

#training_arc("C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\PreparedMaterials.csv")
