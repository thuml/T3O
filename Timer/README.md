## Timer HostDemo使用手册

### 1.Anaconda安装

> **系统要求：Linux  x86_64**

在待安装目录执行以下指令：

```bash
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

之后按照指示安装即可

### 2.虚拟环境配置

首先激活安装的anaconda，创建新的Python11虚拟环境（名字可以修改）并激活

```bash
# 创建虚拟环境，<name>修改为您需要的名字即可
conda create -n <name> python==3.11
# 激活刚创建的虚拟环境
conda activate <name>
```

完成虚拟环境的激活后安装`packages`目录下的离线依赖包。*注意，此处如果系统架构不同于场外环境的话可能会出现复杂bug，届时请与笔者联系*

```bash
# 按照requirements.txt安装packages目录下的离线依赖包
pip install --no-index --find-links=packages -r requirements.txt
```

### 3.代码执行测试

在代码根目录（即跟run.py同级的目录）执行:
```bash
bash scripts/HostDemo/ETTh1.sh
```
代码将生成目录`\test_results`，其中包含若干个`.npz`文件，记录了数据集的预测结果（predict）与真实值（Ground Truth）。

请将包含`.npz`文件的目录复制到根目录下文件`app.py`的`folder_path`变量中,如下所示：
```python
# 文件夹路径
folder_path = 'test_results/large_finetune_2G_{672}_{96}_{96}__Timer_ETTh1_ftM_sl672_ll576_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp24-04-10_20-04-02/ETTh1.csv/96/'
```

文件`app.py`为Gradio展示文件。最终，运行`app.py`进行展示：
```bash
# Method 1 (vanilla start-up)
python app.py

# Method 2 (Dynamic start-up)
gradio app.py
```


*如果运行过程中出现显存不足、卡死以及其他问题导致没有正常输出，请与笔者联系*