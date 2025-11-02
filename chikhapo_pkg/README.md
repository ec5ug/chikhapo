# Evaluation

When running evaluation, we assume that the raw model output has been parsed to return the model's actual prediction. See the example below:

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

We expect model predictions to be placed into a JSON file. Users need only to provide the path to the JSON file. However we expect the JSON to follow a specifc format. See below for more details.

## Word Translation
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
(see Word Translation)