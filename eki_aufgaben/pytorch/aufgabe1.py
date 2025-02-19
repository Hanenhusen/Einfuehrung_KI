import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import os 

# Einleitung
# ##########
#
# Wir wollen heute ein erstes "echtes" Bildverarbeitungsproblem auf dem FashionMNIST Datensatz
# Zalando lösen. 
#
#   https://www.kaggle.com/datasets/zalando-research/fashionmnist
#
# Der Datensatz besteht aus 60.000 Grauwertbildern der Größe 28x28 Pixel von insgesamt 
# 10 verschiedenen Kleidungskategorien (Hemden, Schuhe, etc..)
# Wir laden zunächst die Datensets und erzeugen die dazugehörigen DataLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dataset = torchvision.datasets.FashionMNIST("fashionMNIST", 
                                            download=True,
                                            train=True,
                                            transform=torchvision.transforms.ToTensor())

loader = DataLoader(dataset, batch_size=16, shuffle=True)

dataset_test = torchvision.datasets.FashionMNIST("fashionMNIST", 
                                            download=True,
                                            train=False,
                                            transform=torchvision.transforms.ToTensor())

loader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)

# Mit der torchvision.utils.make_grid Methode können wir einige der Bilder in einem Gittermuster anordnen und
# dann mit MatPlotLib visualisieren
batch, labels = loader.__iter__().__next__()
grid = torchvision.utils.make_grid(batch, 4).permute(1,2,0)
plt.imshow(grid)
plt.show()

# Aufgabe 1
# In dieser Übung wollen wir versuchen die Bilder aus dem Fashion-MNIST Datensatz
# mit Hilfe der tradionellen voll-vernetzten Netzwerkarchitektur aus der Vorlesung
# zu klassifizieren.
class FullNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        # TODO:
        # Erzeugen Sie hier mit Hilfe des nn.Linear Moduls
        # drei Schichten mit sinnvoller Eingabe- und Ausgabegröße
        # Hinweis: Die Bild-Daten haben eine Auflösung von 28x28 Pixeln mit 1 Kanalen
        # Nach dem flatten entspricht dies 28*28 = 784 Eingabedimensionen
        # Verwenden Sie noch 392 Neuronen in der ersten und 128 Neuronen in der zweiten
        # Schicht. Auf der letzten Schicht brauchen Sie exakt 10 Neuronen (für jede Klasse 1)
        self.linear1 = nn.Linear(784, 392)
        self.linear2 = nn.Linear(392, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, x):
        # TODO: 
        # Implementieren Sie den Forward-Pass ihres Netzwerkes
        # indem Sie die Daten zunächst flatten und dann suksezive 
        # durch die linearen Schichten und den Sigmoid geben. 
        # ACHTUNG: Auf der letzten Schicht brauchen Sie keine Nicht-Linearität
        x = self.flatten(x)
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.linear3(x)

        return x

# Aufgabe 2
# Wir wollen nun eine klassische Faltungsarchitektur ausprobieren.
# HINWEIS:
# Denken Sie daran das sie weiter unten im Code ihr ConvNet instantieren
# müssen anstatt dem FullNet von oben
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        # TODO: 
        # Erzeugen Sie mittels nn.MaxPool2D und nn.Conv2d 
        # geeigente Layer. Sie benötigen 2 Faltungen mit je einer kernel_size von (5,5)
        # Verwenden Sie padding="same". Da sie nach jeder Faltung eine MaxPooling anwenden
        # werden vergrößern Sie gleichzeitig die Anzahl Kanäle (also z.B. 4, dann 8)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.linear1 = nn.Conv2d(1, 4, kernel_size=(5,5),padding="same")
        self.linear2 = nn.Conv2d(4, 8, kernel_size=(5,5), padding="same")

       
        # TODO:
        # Überlegen Sie wieviele "Neuronen" ihr Netzwerk nach den drei Faltungslayern
        # noch hat (bei 8 Kanälen) und erzeugen Sie ein entsprechend großes Fully-Connected
        # Layer mit nn.Linear (wie oben))
        self.linear = nn.Linear(392, 10)
        
    def forward(self, x):
        # TODO:
        # Implementieren Sie den Forward-Pass ihres Faltungsnetzwerkes
        # indem Sie immer abwechseln zwischen Faltung und Pooling + ReLU. 
        x = self.relu(self.pool(self.linear1))
        x = self.relu(self.pool(self.linear2))

        # Flatten Sie nun das Ergebniss und wenden Sie den letzten Fully-Connected Teil
        # an um ihr Ergebniss zu erzeugen
        x = self.linear(self.flatten(x))
        return x

net = FullNet() ## ACHTUNG: Hier müssen sie für Aufgabe 2 das ConvNet instantieren anstatt das FullNet
optim = torch.optim.SGD(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()



### AB HIER BRAUCHEN SIE NICHTS ZU TUN AUSSER ZU VERSTEHEN WAS PASSIERT
for epoch in range(10):
    total_loss = 0
    total_cnt = 0
    total_correct = 0

    bar = tqdm(loader)
    for batch, labels in bar:
        optim.zero_grad()

        out = net(batch)
        loss = criterion(out, labels)
        loss.backward()
        
        total_correct += torch.sum(torch.argmax(out, dim=1) == labels)
        total_loss += loss.item()
        total_cnt += batch.shape[0]

        bar.set_description(f"train: epoch={epoch}, loss={1000.0*total_loss / total_cnt:.3f}, acc={total_correct / total_cnt * 100:.2f}%")

        optim.step()

    total_loss = 0
    total_cnt = 0
    total_correct = 0

    bar = tqdm(loader_test)
    for batch, labels in bar:
        with torch.no_grad():
            out = net(batch)
            loss = criterion(out, labels)
        
        total_correct += torch.sum(torch.argmax(out, dim=1) == labels)
        total_loss += loss.item()
        total_cnt += batch.shape[0]

        bar.set_description(f"test: epoch={epoch}, loss={1000.0*total_loss / total_cnt:.3f}, acc={total_correct / total_cnt * 100:.2f}%")





