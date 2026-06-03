# Knowledge-Editing-Experiment
本仓库是基于EasyEdit进行的一次ROME和MEMIT大模型知识编辑实验。

## 1. 开发环境搭建

步骤1：下载Python 3.10。建议使用conda创建虚拟环境。

```
conda create -n ke python=3.10
conda activate ke
```

步骤2：代码部署。

```
git clone https://github.com/suishixingkong/Knowledge-Editing-Experiment.git
cd Knowledge-Editing-Experiment
git submodule update --init --recursive    #部署依赖子模块EasyEdit
cd EasyEdit
git am ../0001-support-for-Qwen3.patch     #把上让EasyEdit支持Qwen3的补丁
```

步骤3：安装依赖库

```
pip install -r requirements.txt
```

步骤4：下载大模型

```
hf download Qwen/Qwen3-1.7B
```

## 2. 实验内容

**基准测试**

执行如下命令，进行基准测试。打印大模型在知识编辑前的输出。

```
python baseline.py
```

**单条知识编辑测试**

执行如下命令，进行单条编辑测试，采用ROME算法，并且打印大模型知识编辑后的输出。测试数据来自`test_set.json`。

```
python edit_rome.py
```

**批量知识编辑测试**

执行如下命令，进行批量知识编辑测试，采用MEMIT算法，并且统计编辑花费的总时间和显存占用情况。测试数据来自`zsre_data.json`。

```
python edit_memit.py
```

**评估脚本**

执行如下命令，对单条知识编辑的效果进行评估统计。分别统计编辑成功率、泛化性和局部性。

```
python evaluate.py
```

如果要进行跨语种泛化测试，则修改evaluate.py脚本中的data_path变量为`./test_set2.json`，然后重新执行脚本。
