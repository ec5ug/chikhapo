
# Introduction

This [package](https://pypi.org/project/chikhapo/) contains the Python package to run ChiKhaPo. Our benchmark is described in [ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models](https://www.arxiv.org/abs/2510.16928).
ChiKhaPo contains 4 word-level tasks, with two directions each (comprehension and generation), intended to benchmark generative models for lexical competence. The processed lexicon data that our tasks rely on can be found on [HuggingFace](https://huggingface.co/datasets/ec5ug/chikhapo) and will be automatically downloaded as needed by this package.

# Setup

**Huggingface Token**
To access our datasets, you will need a HuggingFace token. This can be done by entering the following line in command line

```
export HF_TOKEN="HF_XXXXXXXXXXXXX"
```

This access token will be read in as an environment variable.

For more details, go to this [link](https://medium.com/@manyi.yim/store-your-hugging-face-user-access-token-in-an-environment-variable-fee94fcb58fc)

**Dataset Access**
We draw on [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) and [GLOTLID](https://huggingface.co/datasets/cis-lmu/glotlid-corpus), Huggingface datasets that require users to apply for access. Please visit both links to apply for access.

# Tasks

The 4 tasks (referenced by their task keys) are as follows:

* ```word_translation```: Prompts LLM directly for word translation (2746 languages)
* `word_translation_with_context`: Prompts LLM to translate a word given monolingual context (525 languages)
* `translation_conditioned_language_modeling`: Softly measures LLM capability to understand or generate a word in a natural MT setting (211 languages)
* `bag_of_words_machine_translation`: Word-level MT evaluation (211 languages)

Each task has two subtasks corresponding to two directions: `X_to_eng` testing comprehension and `eng_to_X` testing generation. These tasks test the models' abilities to comprehend or generate a list of words respectively, in various settings. See more details on the description and evaluation procedure for each task and direction in the paper.  

# Getting data per subtask

Instantiate an object of the `TaskFeeder` class for your task:
```
from chikhapo import TaskFeeder
wt_feeder = TaskFeeder("word_translation")
wtwc_feeder = TaskFeeder("word_translation_with_context")
```

This object allows you to obtain a list of language pairs available per task and direction, and to obtain subtask data for each language pair. 

**Get the set of languages available**:

Our lexicons are English-centric, and may either be `xxx_eng` (used for `comprehension` evaluation) or `eng_xxx` (used for `generation` evaluation). 

Retrieve the set of languages available for a particular task as follows, specifying the direction (`X_to_eng`, `eng_to_X` or `None`).

Setting `DIRECTION=None` retrieves language pairs in both directions.
```
word_translation_language_pairs = wt_feeder.get_lang_pairs(DIRECTION="X_to_eng")
```

**Obtain the task data for each language pair**:

Obtain the task data for a particular language pair as follows. The `lite` version of our task datasets contain at most 300 words per language pair and direction, and can be used for faster evaluation. 

```
word_translation_data = wt_feeder.get_data_for_lang_pair(lang_pair="spa_eng", lite=True)
```
This method returns a dictionary. The dictionary keys are source-language words, and each key’s value is a list of translations in the target language.

**Retrieve default formatted prompts for each task**:

We provide a default prompt per task, and a formatter that returns a list of ready-to-use task prompts (one per input) for the task and language pair.

```
word_translation_prompts = wt_feeder.get_prompts_for_lang_pair(lang_pair="spa_eng", lite=True)
```

You may also use your own custom prompt.

# Evaluation

Broadly, each task evaluation computes word scores for each word for that task and language pair. We compute a language score as an aggregate over the word scores of that language, and the task score as an aggregate over language scores.

You will need to run inference with your LLM on the prompts from the previous step to get its responses. 

Instantiate the task evaluator as follows:

```
from chikhapo import Evaluator
wt_evaluator = Evaluator("word_translation")
```

To compute a language pair score (such as `spa_eng`), specify the path to the output file containing the output file for that language pair. The evaluation requires a particular JSON format for the responses of your model (see **Output File Formats** for more information). The direction is set automatically from the required fields in the JSON. 
```
wt_evaluator.evaluate(file_path="path/to/file/file.json")
lang_score = wt_evaluator.get_lang_score()
```

The benchmark reports aggregate language scores (or language family scores) for each task and direction.  

# Output file formats

We expect model predictions to be placed into a JSON file with a particular format depending on task. 

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
```
The parsed model output for this source word would be `decision`.

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

## Translation-Conditioned Language Modeling
We expect outputs to be placed in a JSON file with the following format:

```
    {
        "src_lang": {source_language},
        "tgt_lang": {target_language},
        "data": [
            {
                "src_sentence": {source_sentence_1},
                "tgt_sentence_gt": {target_sentence_1_ground_truth}, 
                "next_word_to_predict": {word_to_predict_1},
                "probability": {probability_1}
            }, {
                "src_sentence": {source_sentence_2},
                "tgt_sentence_gt": {target_sentence_2_ground_truth}, 
                "next_word_to_predict": {word_to_predict_2},
                "probability": {probability_2}
            }, {
                "src_sentence": {source_sentence_3},
                "tgt_sentence_gt": {target_sentence_3_ground_truth}, 
                "next_word_to_predict": {word_to_predict_3},
                "probability": {probability_3}
            },
        ]
    }
```

For an example, please see [tests/raw_test_data/translation_conditioned_language_modeling/amharic_english.json](tests/raw_test_data/translation_conditioned_language_modeling/amharic_english.json)

## Bag-of-Words Machine Translation
We expect outputs to be placed in a JSON file with the following format:

```
{
    "src_lang": {source_language},
    "tgt_lang": {target_language},
    "data": [
        {
            "src_sentence": {source_sentence_1},
            "tgt_sentence_gt": {target_sentence_1_ground_truth},
            "tgt_sentence_pred": {target_sentence_1_prediction}
        }, {
            "src_sentence": {source_sentence_2},
            "tgt_sentence_gt": {target_sentence_2_ground_truth},
            "tgt_sentence_pred": {target_sentence_2_prediction}
        }, {
            "src_sentence": {source_sentence_3},
            "tgt_sentence_gt": {target_sentence_3_ground_truth},
            "tgt_sentence_pred": {target_sentence_3_prediction}
        }
    ]
}
```

For an example, please see [tests/raw_test_data/bag_of_words_machine_translation/amharic_english.json](tests/raw_test_data/bag_of_words_machine_translation/amharic_english.json)

# Analyzing model results

The `Evaluator` computes the language score for _one_ language pair. To compute language scores over _numerous_ language pairs and conduct language family analysis, you must instantiate `ResultAnalyzer` with the task you would like to perform evaluator on.

```
analyzer = ResultAnalyzer("word_translation")
```

Make sure all language pair JSON files you would like to run evaluation on are housed under a specific folder which is referred to in the documentation as the `results_directory`. Each file in this directory contains model outputs for one language pair. We assume that all language pairs are English-centric and translate in the same direction: all files either to English or from English. Calling `ResultAnalyzer.get_results_by_language(path/to/results_directory)` runs `Evaluator.evaluate` on every single file over every JSON file in the directory.

```
analyzer.get_results_by_language("path/to/results_directory")
```

Scores can be accessed using a field of `ResultAnalyzer`: `results_by_language`. Aggregate statistics can be collected using `ResultAnalyzer.get_language_score_average()` and `ResultAnalyzer.get_language_score_standard_deviation()`. To analyze results by language family, call `ResultAnalyzer.get_results_by_language_family()`. Language family scores can be accessed using the field `results_by_language_family`.

Example Usage
```
avg = analyzer.get_language_score_average()
std_dev = analyzer.get_language_score_standard_deviation()
analyzer.get_results_by_language_family()
```

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
