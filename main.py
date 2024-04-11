import os
from src.dataloader import MyDataset
from src.model import *
from src.args import get_public_config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images in dataloader:
        images = images.to(device)
        embeddings = model(images)
        # Truncate or pad embeddings to ensure they are divisible by 3
        batch_size = embeddings.size(0)
        if batch_size % 3 != 0:
            padding_size = 3 - (batch_size % 3)
            embeddings = torch.cat((embeddings, embeddings[-padding_size:]), dim=0)
        
        loss = criterion(embeddings[0::3], embeddings[1::3], embeddings[2::3])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# 测试函数 
def test(model, dataloader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            embedding = model(images)
            embeddings.append(embedding)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def main(args):
    mytransform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # 创建 Dataset 和 DataLoader
    train_dataset = MyDataset(root_dir=args.dataset+'/train', transform=mytransform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    test_dataset = MyDataset(root_dir=args.dataset+'/test', transform=mytransform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 加载预训练的 ResNet-50 作为 backbone    
    resnet50 = models.resnet50(pretrained=True)
    in_features = resnet50.fc.in_features
    resnet50.fc = nn.Identity()
    model = SimilarityNet(resnet50, in_features)
    if not os.path.exists('save'):
        os.makedirs('save')
    torch.save(model.state_dict(), 'save/model.pth')
    # print(model)

    # 定义损失函数和优化器
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)

    # 训练循环
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device
    model = model.to(device)

    for epoch in range(args.max_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{args.max_epochs}], Loss: {train_loss:.4f}')
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'save/model_epoch_{epoch}.pth')

    torch.save(model.state_dict(), 'save/best_model.pth')

    # 提取测试集的特征向量
    test_embeddings = test(model, test_loader, device)

    # 计算余弦相似度
    similarity_matrix = torch.cosine_similarity(test_embeddings.unsqueeze(1), test_embeddings.unsqueeze(0), dim=2)

    # 根据阈值计算准确率等指标
    threshold = 0.7
    eye_matrix = torch.eye(len(similarity_matrix)).long().to(device)
    preds = (similarity_matrix > threshold).long() - eye_matrix
    labels = torch.tensor([i // 10 for i in range(len(preds))])
    labels = labels.view(-1, 1).to(device)
    accuracy = (preds == labels).float().mean()

    # print(f'Accuracy: {accuracy.item():.4f}')
    


if __name__ == '__main__':
    parser = get_public_config()
    args = parser.parse_args()
    print(args)
    main(args)
