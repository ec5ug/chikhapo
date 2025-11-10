
# Introduction

This package contains the code to run ChiKhaPo. Our benchmark as described in [ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models](https://www.arxiv.org/abs/2510.16928).
ChiKhaPo contains 4 word-level tasks, with two directions each (comprehension and generation), intended to benchmark generative models for lexical competence. The processed lexicon data that our tasks rely on can be found on [HuggingFace](https://huggingface.co/datasets/ec5ug/chikhapo) and will be automatically downloaded as needed by this package.

# Tasks

The 4 tasks (referenced by their task keys) are as follows:

* ```word_translation```: Prompts LLM directly for word translation (2746 languages)
* `word_translation_with_context`: Prompts LLM to translate a word given monolingual context.
* (Coming soon to this package) `translation_conditioned_lm`: Softly measures LLM capability to understand or generate a word in a natural MT setting.
* (Coming soon to this package) `bow_mt`: Word-level MT evaluation.

Each task has two directions (`comprehension` and `generation`) that tests the models abilities to comprehend or generate each word respectively. See more details on the description and evaluation procedure for each task and direction in the paper.  

# Extracting data
To extract data, you will need to use the `TaskFeeder` class. This object uses the Factory software design pattern, meaning you need to specify the task you want to extract data from. This can be done as
```
from chikhapo import TaskFeeder
wt_feeder = TaskFeeder("word_translation")
wtwc_feeder = TaskFeeder("word_translation_with_context")
```
We outline several functionalities of `TaskFeeder` below

**Get the set of languages available**:

If `DIRECTION=None`, language pairs in the `X->eng` and `eng->X` directions are ported. To retrieve all language pairs that translate into English, set `DIRECTION="X_to_eng"`. Similarly, set `DIRECTION="eng_to_X"` to retrieve all language pairs that translate from English.
```
word_translation_language_pairs = wt_feeder.get_lang_pairs(DIRECTION=None)
```

**Obtain the task data for that language pair**:

To obtain the task data for a language pair, specify the language pair in the method call `get_data_for_lang_pair`. Note that this method returns a dictionary. The dictionary keys are source-language words, and each key’s value is a list of translations in the target language.
```
word_translation_data = wt_feeder.get_data_for_lang_pair(lang_pair="spa_eng", lite=True)
```
There is another parameter `lite` that determines whether an abridged version of the data is returned. Should `lite=True`, the dataset is (deterministically) shuffled and 300 samples are returned. If you want like the **unabridged** version of the data, set `lite=False`.

**Retrieve default prompts for each task**:

To obtain a list of default prompts, specify the language pair in the method `get_prompts_for_lang_pair`. A list of prompts will be returned. The parameter `lite` is also included; its functionality in `get_prompts_for_lang_pair` is the same as detailed in `get_data_for_lang_pair`. You are invited to write your own prompts should the default fail.
```
word_translation_with_context_prompts = wtwc_feeder.get_prompts_for_lang_pair(lang_pair="spa_Latn-eng", lite=True)
```

# Evaluation

Broadly, each task computes a word score per language. Using the `Evaluator` class, we compute a language score as an aggregate over the word scores of that language, and the task score as an aggregate over language scores.

Before you use the methods outlined in the `Evaluator` class, you will have to run inference on your LLM and format the responses of your model into a JSON file (see **Output File Formats** for more information). Please note that much like the `TaskFeeder` class, the `Evaluator` class adopts a factory design pattern. To access the correct evaluator for a task, you only need to specify the task you want in the instantiation of an `Evaluator` class. This can be done as follows

```
from chikhapo import Evaluator
wt_evaluator = Evaluator("word_translation")
```

To compute a language score, specify the path to the (model) output file containing information on the source and target language as well as the (parsed) model outputs you would like to run the evaluator on
```
wt_evaluator.evaluate(file_path="path/to/file/file.json")
lang_score = wt_evaluator.get_lang_score()
```

Features coming soon
* retrieve language family information for languages `get_lang_family`
* Aggregate your language scores. 

# Output file formats

We expect model predictions to be placed into a JSON file with a particular format depending on task. Users need only to provide the path to the JSON file to the evaluator as per (6). 

## Word Translation

This task requires the model to output the translation of single words. For example, we may have:
```
Prompt:
Translate the following word from Magahi to English. Respond with a single word.

Word:निर्णय
Translation:
---
Raw Model Output:
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following word from Magahi to English. Respond with a single word.

Word:निर्णय
Translation:<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>decision
---
Parsed Model Prediction:
decision
```

We expect outputs to be placed in a JSON file with the following format:

```
{
    "src_lang": {source_language},
    "tgt_lang": {target_language},
    "data": [
        {
            "word": {word_1_to_translate},
            "prediction": {model_translation_for_word_1}
        },
        {
            "word": {word_2_to_translate},
            "prediction": {model_translation_for_word_2}
        },
        {
            "word": {word_3_to_translate},
            "prediction": {model_translation_for_word_3}
        }
    ]
}
```
For an example, please see [tests/raw_test_data/wt_equivalence_spa_eng.json](tests/raw_test_data/wt_equivalence_spa_eng.json)
## Word Translation with Word Context
The output format is identical to that used in Word Tranlslation.



# Cite
If you use this data or code, please cite
```
@article{chang2025chikhapo,
  title={ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models},
  author={Chang, Emily and Bafna, Niyati},
  journal={arXiv preprint arXiv:2510.16928},
  year={2025}
}
```
