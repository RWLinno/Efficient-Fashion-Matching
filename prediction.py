import os
from src.dataloader import MyDataset
from src.model import *
from src.args import get_public_config
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

def predict_similarity(model, image1_path, image2_path, device, threshold=0.8):
    # 图片预处理
    mytransform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # 加载图片
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    # 对图片进行预处理
    image1 = mytransform(image1).unsqueeze(0).to(device)
    image2 = mytransform(image2).unsqueeze(0).to(device)

    # 提取图片特征
    with torch.no_grad():
        feature1 = model(image1)
        feature2 = model(image2)

    # 计算余弦相似度
    similarity = torch.cosine_similarity(feature1, feature2)

    # 根据阈值判断是否相似
    is_similar = similarity.item() > threshold

    return is_similar, similarity.item()

if __name__ == '__main__':
    # 加载预训练的模型
    resnet50 = models.resnet50(pretrained=True)
    in_features = resnet50.fc.in_features
    resnet50.fc = nn.Identity()
    model = SimilarityNet(resnet50, in_features)
    model.load_state_dict(torch.load('./save/best_model.pth'))
    model.eval()

    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 预测两张图片的相似度
    image1_path = 'data/sample/pic1.jpg'
    image2_path = 'data/sample/pic2.jpg'
    image3_path = 'data/sample/pic3.jpg'

    is_similar, similarity_score = predict_similarity(model, image1_path, image2_path, device)
    print(f"Pic 1 and 2:Is similar: {is_similar}")
    print(f"Similarity score: {similarity_score:.4f}")

    is_similar, similarity_score = predict_similarity(model, image1_path, image3_path, device)
    print(f"Pic 1 and 3:Is similar: {is_similar}")
    print(f"Similarity score: {similarity_score:.4f}")

    is_similar, similarity_score = predict_similarity(model, image3_path, image2_path, device)
    print(f"Pic 2 and 3:Is similar: {is_similar}")
    print(f"Similarity score: {similarity_score:.4f}")