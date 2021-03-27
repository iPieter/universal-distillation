# Distillation using the Command Line Interface

In this tutorial, we will show you how to set up a distillation task from the command line. You will need two things:

1. A teacher model that you want to distill. All models from [the Hugginface model repository](https://huggingface.co/models?pipeline_tag=fill-mask) with a fill-mask / MLM head will work. In this tutorial, we will use the standard `bert-base-uncased` model.
2. A dataset that you want to use for distillation. In this tutorial, we a 'small', but high-quality, dataset: Europarl.

---
## Step 0: Install Universal-distillation and it's dependencies


### GPU support



---
## Step 1: Get your dataset
We will use the English section of the [Europarl corpus](https://opus.nlpl.eu/Europarl.php). 
This is a very high-quality parallel corpus from the European Parlement created by professional interpreters and translators.
It's also quite small for a language corpus nowadays, only 114 MB, but for our distillation tutorial that's ok.

```bash
curl https://opus.nlpl.eu/download.php?f=Europarl/v8/mono/en.txt.gz 
```

---

## Step 2: Start training


```bash
python universal_distillation/distillation.py 
```

---

## Step 3: Use your model


## credits
