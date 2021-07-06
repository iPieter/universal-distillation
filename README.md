<div align="center">    
 
# Universal Language Model Distillation

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/ipieter/universal-distillation/workflows/CI%20testing/badge.svg?branch=master&event=push)


</div>
 
## Description   
What it does   

## How to run   
First, clone the project and install the dependencies.

```bash
# clone project   
git clone https://github.com/iPieter/universal-distillation

# install project   
cd universal-distillation
pip install -e .   
pip install -r requirements.txt
 ```   
 
## Distillation using the Command Line Interface

In this tutorial, we will show you how to set up a distillation task from the command line. You will need two things:

1. A teacher model that you want to distill. All models from [the Hugginface model repository](https://huggingface.co/models?pipeline_tag=fill-mask) with a fill-mask / MLM head will work. In this tutorial, we will use the standard `bert-base-uncased` model.
2. A dataset that you want to use for distillation. In this tutorial, we a 'small', but high-quality, dataset: Europarl.

### Step 1: Get your dataset
We will use the English section of the [Europarl corpus](https://opus.nlpl.eu/Europarl.php). 
This is a very high-quality parallel corpus from the European Parlement created by professional interpreters and translators.
It's also quite small for a language corpus nowadays, only 114 MB, but for our distillation tutorial that's ok.

```bash
wget https://opus.nlpl.eu/download.php\?f\=Europarl/v8/mono/en.txt.gz -O en.txt.gz
gunzip en.txt.gz
```

The data is now unzipped and stored in the file `en.txt`.

---

### Step 2: Start training
Now we have the data, we can start training. Downloading the teacher model will happen automatically, so no need to do this manually. If you feel this takes too long and you just want to try out the training, for example to get a sense of timings, you can add `--limit_train_batches N`. This limits each epoch to `N` batches during training.

```bash
python universal_distillation/distillation.py \
    --batch_size 8 \
    --gpus 1 \
    --max_epochs 3 \
    --save_dir my_distilled_model/ \
    --teacher bert-base-uncased \
    --data en.txt
```

There are a few things that happen in the background once you run that command. First, this library creates a student and a teacher model. The teacher is `bert-base-uncased` and the student will use the same architecture as the teacher by default, only the number of heads is smaller: 6 instead of 12. Since we are training on a specific domain (Europarl), this should be enough. Of course, you can mix and match different and bigger teachers with smaller students, but the performance will vary a lot.  

Second, the Huggingface library downloads the teacher model and the tokenizer. Third, the dataset is loaded from disk and initialized with the tokenizer, notice that the tokenization itself happens later by default. Finally, the distillation loop starts. 

---

### Step 3: Use your model
Finally, you can use the model with the Huggingface library! All the files from the student (Pytorch model and tokenizer) are saved in the folder we defined earlier: `my_distilled_model/`. You can import the model from this folder directly and test the masked language modeling task with only 3 lines: 

```python
from transformers import pipeline
p = pipeline("fill-mask", model="my_distilled_model/")

p("This is a [MASK].")
```

Although this was a straitforward example, this is often enough to create your own domain-adapted model. In this case, it's 