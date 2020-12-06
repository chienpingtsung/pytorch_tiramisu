import torch
from tqdm import tqdm

import utils.training as train_utils
from torch.utils.data import DataLoader

from datasets.CRACK500 import CRACK500

from models.tiramisu import FCDenseNet103

BATCH_SIZE = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_dset = CRACK500("enhance/")
train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)

val_dset = CRACK500("testdst/")
val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=False)

model = FCDenseNet103(n_classes=2).to(device)
model.apply(train_utils.weights_init)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

for epoch in range(1, 1000):
    ### Train ###
    model.train()
    tq = tqdm(train_loader)
    for data in tq:
        inputs = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long) // 255
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        tq.set_description("Trainning epcoh {:3} loss is {}: ".format(epoch, loss.item()))

    ### Test ###
    model.eval()
    test_loss = 0.0
    test_times = 0
    tq = tqdm(val_loader)
    for data in tq:
        with torch.no_grad():
            inputs = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device, dtype=torch.long) // 255

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_times += 1

            tq.set_description("Testing epoch {:3}: ".format(epoch))

    test_loss /= test_times
    print("Epoch {:3} test_loss: {}".format(epoch, test_loss))

    torch.save(model.state_dict(), "checkpoint_at_{}.pth".format(epoch))

    scheduler.step(test_loss)
