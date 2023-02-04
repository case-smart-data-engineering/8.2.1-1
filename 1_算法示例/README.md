# BERT-BiLSTM-CRF模型进行NER命名实体识别

### 文件清单
- data：数据集
- model：模型参数、保存训练后的模型、模型输入所需的文件
- clue_process.py：数据处理文件
- conlleval.py：模型评估代码
- models.py：BERT+BiLstm+CRF模型
- solution.py：运行入口
- utils.py：工具文件


### 模型下载之后
1. 先运行do_trian之后会生成模型bin文件，之后再进行验证和测试
2. 对于大模型网络的运行，建议下载到本地进行
3. 把训练的模型参数文件放在model下面
4. 训练的模型文件下载链接：https://github.com/KaiserLord/bigFiles/tree/master/model/


### 数据处理

运行`clue_process.py`文件对数据集进行处理
把数据从json文件处理成BIO格式，保存在txt同名文件
```
彭	B-name
小	I-name
军	I-name
认	O
为	O
，	O
国	O
内	O
银	O
行	O
现	O
在	O
走	O
的	O
是	O
台	B-address
湾	I-address

温	B-name
格	I-name
的	O
球	O
队	O
终	O
于	O
```
预测结果保存在modeltoken_labels_.txt中，第一列为输入的BIO格式，第二列是预测的实体类型
        

### 使用方法
```
直接运行solution.py即可
```
