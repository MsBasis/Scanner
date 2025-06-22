#Better late than never
import torch 
import torch.nn as nn 
import torch_optimizer as optim
import torch.optim as t_optim
from Data_Loader import dara_loaders

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64,2)
        )
        
    def forward(self,x):
        return self.net(x)


def training_arc(csv, epochs=200, batch_size=256, lr=0.001,patience =20, save = "C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\modelScanner2.pt"):
    train, test, input_dim = dara_loaders(csv, batch_size=batch_size)
    
    model = MLP(input_dim)
    #criterion = nn.SmoothL1Loss()
    optimizer = optim.Ranger(model.parameters(), lr=lr)
    scheduler = t_optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.5, patience=5)
    best_loss = float('inf')
    patience_count = 0
    
    #actual traini
    print('We startin this shi')
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for X_batch, y_batch in train:
            optimizer.zero_grad()
            outputs = model(X_batch)
            diff = torch.abs(outputs - y_batch)
            loss = torch.mean(diff[:, 0]) + 2.0 * torch.mean(diff[:, 1])
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            
        avgLoss = run_loss / len(train)
        print(f"[Epoka {epoch:02d}] Sredni loss: {avgLoss:.6f}")  

        scheduler.step(avgLoss)
        if avgLoss < best_loss - 1e-5:
            best_loss = avgLoss
            patience_count = 0
            torch.save(model.state_dict(), save)
            print('Heck yea it works')
        else:
            patience_count += 1
            if patience_count >= patience:
                print("U crossed me")
                break
    model.load_state_dict(torch.load(save))
    evaluation(model,test)
    return model

def evaluation(model, test_loader, Ntolerance = 0.05, Ktolerance = 0.05):
    model.eval()
    total_MSE = 0.0
    total_MAE = 0.0
    G_n = 0
    G_k = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            diff = torch.abs(outputs - y_batch)
            mse = torch.mean((outputs - y_batch)**2)
            mae = torch.mean(diff[:, 0]) + 5.0 * torch.mean(diff[:, 1])
            total_MSE += mse.item()
            total_MAE += mae.item()
            #we need accuracy cause thats the only parameter i actually 100% understand (jk jk)
            nTrue, kTrue = y_batch[:,0], y_batch[:,1]
            nPred, kPred = outputs[:,0], outputs[:,1]
            
            G_n += torch.sum(torch.abs(nTrue - nPred) < Ntolerance).item()
            G_k += torch.sum(torch.abs(kTrue - kPred) < Ktolerance).item()
            total += len(y_batch)
            
    avg_MSE = total_MSE / len(test_loader)
    avg_MAE = total_MAE / len(test_loader)
    A_n = G_n / total
    A_k = G_k / total

    print(f"\nTest MSE: {avg_MSE:.6f}")
    print(f"Test MAE: {avg_MAE:.6f}")
    print(f"Accuracy n ({Ntolerance}): {A_n:.2%}")
    print(f"Accuracy k ({Ktolerance}): {A_k:.2%}")

    return avg_MSE, avg_MAE, A_n, A_k

training_arc("C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\PreparedMaterials.csv")
