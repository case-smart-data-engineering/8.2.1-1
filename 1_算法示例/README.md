# BERT-BiLSTM-CRF模型进行NER命名实体识别

### 文件清单
- data：数据集
- model：模型参数、保存训练后的模型、模型输入所需的文件
- clue_process.py：数据处理文件
- conlleval.py：模型评估代码
- models.py：BERT+BiLstm+CRF模型
- solution.py：运行入口
- utils.py：工具文件


### 运行步骤
1. 在指定url（步骤3和步骤4）下载大文件（模型文件）到本地后再执行上传操作，在bert-base-chinese目录下上传pytorch_model.bin文件，在model目录下上传`已训练`的pytorch_model.bin文件，即可直接运行solution.py进行测试。
2. 已训练的模型文件下载链接：https://github.com/KaiserLord/bigFiles/tree/master/model/ 。
3. 未训练的模型文件下载链接：https://huggingface.co/bert-base-chinese/tree/main 。

### 训练模型
1. 删除model目录下的所有文件，并在solution.py中把`do_train`和`do_eval`设置为True。
2. 在`1_算法示例`下运行：python solution.py 即可进行模型训练。

### 数据处理
运行`clue_process.py`文件对数据集进行处理(已完成)，把数据从json文件处理成BIO格式，保存在txt同名文件。
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
预测结果保存在modeltoken_labels_.txt中。
