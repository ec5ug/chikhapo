
# Introduction

This package contains the code to run ChiKhaPo, our benchmark as described in [ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models](https://www.arxiv.org/abs/2510.16928).
ChiKhaPo contains 4 word-level tasks, with two directions each (comprehension and generation), intended to benchmark generative models for lexical competence. The processed lexicon data that our tasks rely on can be found on [HuggingFace](https://huggingface.co/datasets/ec5ug/chikhapo) and will be automatically downloaded as needed by this package.

# Tasks

The 4 tasks (referenced by their task keys) are as follows:

* ```word_translation```: Prompts LLM directly for word translation (2746 languages)
* `word_translation_with_context`: Prompts LLM to translate a word given monolingual context.
* (Coming soon to this package) `translation_conditioned_lm`: Softly measures LLM capability to understand or generate a word in a natural MT setting.
* (Coming soon to this package) `bow_mt`: Word-level MT evaluation.

Each task has two directions (`comprehension` and `generation`) that tests the models abilities to comprehend or generate each word respectively. See more details on the description and evaluation procedure for each task and direction in the paper.  

# Evaluation

Broadly, each task computes a word score per language. We then compute a language score as an aggregate over the word scores of that language, and the task score as an aggregate over language scores.
Running evaluation for each task and direction consists of the following steps:

1. Get the set of languages available:
```
word_translation_language_pairs = ...
```

We also provide language family information for languages. 
```
from chikhapo import ...
lang_family = get_lang_family(lang)
```

2. For each language pair, obtain the task data for that language pair:

```
...
```

3. Format the data into the relevant prompt. We provide a default prompt per task, but you can use your own.

```
...
```

4. Run inference on your LLM over the resulting list of prompts.

5. Format the responses of your model into a JSON file with specifications per task (more detail below).

6. Obtain the language score.
```
...
```


7. Aggregate your language scores. 

```
import statistics
# All languages aggregate
task_score = statistics.mean(...
```

It may be useful also to aggregate by language family; these can be accessed as in (1).


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
