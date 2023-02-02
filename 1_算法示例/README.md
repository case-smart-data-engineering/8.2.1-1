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
3. 把下载的bert-base-chinese放在pretrained_bert_model文件夹下面
4. 把训练的模型参数文件放在model下面
5. requirements.txt中包含的是模型运行环境的详情
6. bert-base-chinese下载链接：https://huggingface.co/bert-base-chinese/tree/main ，下载config.json、vocab.txt和bin文件即可
7. 训练的模型文件下载链接：https://github.com/KaiserLord/bigFiles/tree/master/model/，下载后放在model目录下


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
python ner.py \
    --model_name_or_path ${BERT_BASE_DIR} \
    --do_train True \
    --do_eval True \
    --do_test True \
    --max_seq_length 256 \
    --train_file ${DATA_DIR}/train.txt \
    --eval_file ${DATA_DIR}/dev.txt \
    --test_file ${DATA_DIR}/test.txt \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_train_epochs 10 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn True \
    --rnn_dim 256 \
    --clean True \
    --output_dir $OUTPUT_DIR
```

### 在中文CLUENER2020的eval集上的结果
https://github.com/CLUEbenchmark/CLUENER2020