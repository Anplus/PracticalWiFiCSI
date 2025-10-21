import torch
from torch import nn
from train_dataset import TrainDataset
from torch.optim.lr_scheduler import StepLR
from AlexNet import AlexNet
from Resnet import ResNet18
from Transformer import RF_Transformer
from test import test

def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv3d, nn.Conv2d, nn.Conv1d]:
        nn.init.normal_(model.weight, std=0.03)

def train(model_type, device):
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("model_type: ", model_type)

    if model_type == "AlexNet":
        model = AlexNet()
    elif model_type == "ResNet":
        model = ResNet18()
    elif model_type == "Transformer":
        model = RF_Transformer()

    model.apply(init_weights)

    model.to(device)
    batch_size = 32  # Reduced from 64
    epoch = 200

    myloss = nn.CrossEntropyLoss().to(device)

    init_lr = 3e-3
    optimizer = torch.optim.SGD(model.parameters(), lr = init_lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr = init_lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

   
    train_dataset = TrainDataset()

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    ) 


    for i in range(epoch):

        model.train()  
        total_loss = 0
        correct = 0

        print("-----{} epoch for training-----".format(i + 1))
        
        for data in train_dataloader:
            csi, label = data
            csi = csi.to(device)
            label = label.to(device)
            x = model(csi)
            loss = myloss(x, label)
            loss = loss.requires_grad_()
            total_loss += loss * x.size(0)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (x.argmax(axis=1) == label).sum().item()


        print("Loss:{}, acc:{}".format(total_loss/train_dataset.__len__(), correct/train_dataset.__len__()))

        scheduler.step()

        if (i+1)%5==0:  # for every 5 epochs to save model and test
            save_path = "/srv/csj/tutorial_submit/model_file/" + model_type + "_model_{}.pth".format(i+1)
            torch.save(model, save_path)
            acc, avg_loss = test(device, save_path)
            print("-----{} epoch for testing-----".format(i + 1))
            print("Loss:{}, acc:{}".format(acc, avg_loss))




if __name__ == "__main__":
    # show GPU status
    if torch.cuda.is_available():
        print("GPU is available")
        print("GPU name: ", torch.cuda.get_device_name(0))
        print("Number of GPUs: ", torch.cuda.device_count())
        print("Current GPU index: ", torch.cuda.current_device())
        print("Compute capability:", torch.cuda.get_device_capability(0))
        print("CUDA version (PyTorch build):", torch.version.cuda)
    print("Start training...")
    # train("AlexNet", device = torch.device("cuda:0"))
    train("AlexNet", device = torch.device("cpu"))
    # train("ResNet", device = torch.device("cuda:0"))
    # train("Transformer", device = torch.device("cuda:0"))