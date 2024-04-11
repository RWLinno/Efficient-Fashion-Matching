# Efficient Fashion Matching



### 题目 / 翻译

**任务**
你的任务是编写一个神经网络来解决这个问题。请注意

- 我们不关心图片是否包含同一个人。
- 我们对消费者的图像不做任何假设，例如，图像没有对齐或美白。
- 我们对使用什么方法/框架/语言没有任何要求。您可以根据自己的想法自由设计整个流程。
- 我们并不期望有一个训练有素模型的完美管道。但是，我们希望在足够的时间内，您的模型能够在准确性和运行时间方面达到良好的性能。如果您没有时间实现管道中的所有功能，您可以在代码中定义功能，并在注释中添加一些说明。

**我们更关心**

- 您如何定义目标和管道？哪个部分最重要？
- 您使用了哪些方法和技术，为什么？
- 如何平衡准确性和运行时间？如何加快代码速度？
- 如何分配有限的时间来完成这项任务？
- 实施管道时会遇到哪些挑战？如何解决？
为了重新选择一个正常的工作环境，我们不限制您在线访问其他资料，例如 StackOverfow 或您以前工作过的代码，只要这些资料有助于您完成任务。我们的要求是，您使用或重复使用的任何内容都必须明确引用--代码中的注释足以满足这一要求。

**数据集**
您应使用 DeepFashion 数据集进行训练，该数据集在 deepfash-ion_train test_256.zip 压缩包中提供。请注意，测试集是固定的，但如果您愿意，可以清理/修改训练集。

**提交**
您需要提交源代码文件以及 ReadMe 文档。阅读说明文件应描述

- 您的管道的简要说明。
- 如何运行训练和测试代码。
- 您使用了哪些参考资源。
- 您将有 12 个小时的时间来实现您的管道。欢迎您在 8 小时内提交您的作品。我们更感兴趣的是您是如何解决问题的



### 思路 / 草稿

判断两服装图片是否相似的方法：相似度计算（余弦相似度）

模型选用： 

- 提取图片特征（BackBone）：MLP / CNN / ResNet / InceptionV3 / ViT
- 相似度匹配：先将图片预测结果拉到特征向量，然后算相似度>=阈值
- 不需要卷效果的话，我就直接从简了使用ResNet+MLP来做了

Code Base： 使用了我之前的一个项目，放到github上了https://github.com/RWLinno/ViT-Model-based-Medical-Image-Assisted-Diagnostic-System

目录结构：

```
FashionMatching/
│───README.md #该说明文档 
└───data/  #数据集
│   │───few-shot/  #轻量训练数据
│   │   │   train/
│   │   │   test/
│   └───train_test_256/ #原训练数据
│   │   │   ...
│   └───sample/ #预测素材
└───src/
│   │───args.py  #参数设计
│   │───dataloader.py  #数据预处理
│   └───model.py  #训练模型
│   main.py  #训练程序
│   prediction.py  #结果demo
│   ...
```



### 环境配置

```
conda create -n 虚拟环境名 python>=3.8
conda activate 虚拟环境名
pip install -r requirements.txt
```

这里租用了autoDL的4090跑，然后图片大小统一改成了64*64了，并且由于时间关系用数据集的一部分进行了训练(放在few-shot文件夹)。



### 数据集下载

全部数据集请使用谷歌网盘下载

```
pip install gdown
gdown https://drive.google.com/uc?id=1t2yp2co8mJqYlniFBZQB2Lc0O1qhLECj
```



### 训练 + 预测

参数配置见`src/args.py`

训练命令：

```
python main.py --max_epochs=20 --batch_size=32  --lrate=1e-3 --wdecay=1e-4
```

预测命令(这里用三张图片来分别给出相似和不相似的两组结果)：

```
python prediction.py
```

结果：

![image-20240314165637659](https://s2.loli.net/2024/03/14/iu5SfzDCd6pQ4UX.png)

![image-20240314165626985](https://s2.loli.net/2024/03/14/92QlVfqTvaHod6j.png)



### 声明

时间和精力有限，训练资源珍贵。

任务仅供练习，实际制作用时三小时。（含文档）
