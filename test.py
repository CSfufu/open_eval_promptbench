import copy
import json
import os
import sys
import tiktoken

sys.path.append('/Users/csgo/Desktop/research/promptbench')

import promptbench as pb
from promptbench.models import LLMModel
from promptbench.dataload import DatasetLoader

# The prompts used for paraphrasing and evaluation agents.
from promptbench.mpa import MPA_DEFAULT_PRMOPTS

# The agents for paraphrasing and evaluation, one paraphraser and one evaluator consists of a pipeline.
from promptbench.mpa import ParaphraserAgent, EvaluatorAgent, Pipeline

# The input process of different paraphrase rules
from promptbench.mpa import ParaphraserBasicInputProcess, ParaphraserQuestionOutputProcess, \
    ParaphraserChoicesOutputProcess

# The choice permutation can be implemented without LLMs, so it does not need any agents
from promptbench.mpa import ChoicePermuter

# The input process of different evaluation rules
from promptbench.mpa import EvaluatorMMLUQuestionInputProcess, EvaluatorBasicOutputProcess, \
    EvaluatorMMLUParaphrasedChoicesInputProcess, EvaluatorMMLUNewChoiceInputProcess

api_key = ""

def count_tokens(text, model_name="gpt-4-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

# The paraphraser and evaluator models,
paraphraser = LLMModel("gpt-4-turbo", max_new_tokens=1000, temperature=0.7, api_key=api_key)
evaluator = LLMModel("gpt-4-turbo", max_new_tokens=1000, temperature=0, api_key=api_key)

# the storage path for the resulting paraphrased evaluation data
results_dir_name = f"./mpa_results/"

if not os.path.exists(results_dir_name):
    os.makedirs(results_dir_name)
print("mkdir successful")
# The data format depends on the prompts and the preprocess function of different paraphrase rules. (e.g., ParaphraserBasicInputProcess loaded above)

# You can use your own prompts and your own preprocess functions to generate the data for the paraphraser and evaluator

# here we provide the datasets used in our paper, you could download it here: https://drive.google.com/drive/folders/14wRz4WyTM5pmmT55QQtEpHE4vGAgXAZ6?usp=sharing
with open("data/new_data.json", 'r') as file:
    data = json.load(file)
print("data loaded")

"""
Rule 0: Paraphrase the question
"""
paraphrase_question_prompt = MPA_DEFAULT_PRMOPTS["mmlu"]["paraphraser_paraphrase_question"]
evaluate_paraphrase_question_prompt = MPA_DEFAULT_PRMOPTS["mmlu"]["evaluator_paraphrase_question"]

paraphrase_question_agent = ParaphraserAgent(paraphraser, paraphrase_question_prompt, ParaphraserBasicInputProcess(),
                                             ParaphraserQuestionOutputProcess())
evaluate_question_agent = EvaluatorAgent(evaluator, evaluate_paraphrase_question_prompt,
                                         EvaluatorMMLUQuestionInputProcess(), EvaluatorBasicOutputProcess())

# based on the paraphrase question agent and the evaluate question agent, we can create a pipeline
# the iteration of the pipeline is 1, which means the pipeline will run once, please refer to the implementation of the pipeline for more details about the iteration
paraphrase_question_pipeline = Pipeline(paraphrase_question_agent, evaluate_question_agent, iters=1)

print("pipeline created")

"""
Rule 1: Add context to the question
"""
add_question_context_prompt = MPA_DEFAULT_PRMOPTS["mmlu"]["paraphraser_add_question_context"]
evaluate_add_question_context_prompt = MPA_DEFAULT_PRMOPTS["mmlu"]["evaluator_add_question_context"]
add_question_context_agent = ParaphraserAgent(paraphraser, add_question_context_prompt, ParaphraserBasicInputProcess(),
                                              ParaphraserQuestionOutputProcess())
evaluate_add_question_context_agent = EvaluatorAgent(evaluator, evaluate_add_question_context_prompt,
                                                     EvaluatorMMLUQuestionInputProcess(), EvaluatorBasicOutputProcess())
add_question_context_pipeline = Pipeline(add_question_context_agent, evaluate_add_question_context_agent, iters=1)

print("add context pipeline created")


paraphrase_question_prompt_tokens = count_tokens(paraphrase_question_prompt)
add_question_context_prompt_tokens = count_tokens(add_question_context_prompt)

print(f"Tokens in paraphrase_question_prompt: {paraphrase_question_prompt_tokens}")
print(f"Tokens in add_question_context_prompt: {add_question_context_prompt_tokens}")

print(f"all tokens in paraphrase_question_prompt {paraphrase_question_prompt_tokens + add_question_context_prompt_tokens}")

"""
Rule 2: Paraphrase the choices
"""
paraphrase_choices_prompt = MPA_DEFAULT_PRMOPTS["mmlu"]["paraphraser_paraphrase_choices"]
evaluate_paraphrase_choices_prompt = MPA_DEFAULT_PRMOPTS["mmlu"]["evaluator_paraphrase_choices"]
paraphrase_choices_agent = ParaphraserAgent(paraphraser, paraphrase_choices_prompt, ParaphraserBasicInputProcess(),
                                            ParaphraserChoicesOutputProcess())
evaluate_choices_agent = EvaluatorAgent(evaluator, evaluate_paraphrase_choices_prompt,
                                        EvaluatorMMLUParaphrasedChoicesInputProcess(), EvaluatorBasicOutputProcess())
paraphrase_choices_pipeline = Pipeline(paraphrase_choices_agent, evaluate_choices_agent, iters=1)

print("paraphrase choices pipeline created")

"""
Rule 3: Add a new choice
"""
add_new_choice_prompt = MPA_DEFAULT_PRMOPTS["mmlu"]["paraphraser_add_new_choice"]
evaluate_new_choice_prompt = MPA_DEFAULT_PRMOPTS["mmlu"]["evaluator_add_new_choice"]
add_new_choice_agent = ParaphraserAgent(paraphraser, add_new_choice_prompt, ParaphraserBasicInputProcess(),
                                        ParaphraserChoicesOutputProcess())
evaluate_new_choice_agent = EvaluatorAgent(evaluator, evaluate_new_choice_prompt, EvaluatorMMLUNewChoiceInputProcess(),
                                           EvaluatorBasicOutputProcess())
add_new_choice_pipeline = Pipeline(add_new_choice_agent, evaluate_new_choice_agent, iters=1)

print("add new choice pipeline created")
"""
Rule 4: Permute the choices
"""
# This rule does not need any agents, so we can directly use the ChoicePermuter


# This list will store the paraphrased data with all the rules (0, 1, 2, 3, 4) applied
paraphrased_data_0_1_2_3_4 = []
# for simplicity, we only paraphrase the first 10 questions
data = data[:5]

for idx, d in enumerate(data):
    print(f"Question {idx} paraphrasing")

    d_0_list = paraphrase_question_pipeline(d)
    paraphrase_0 = d_0_list[-1]
    num_tokens = count_tokens(json.dumps(paraphrase_0))
    print(f"Tokens in paraphrase_0: {num_tokens}")

    print("paraphrase_0", paraphrase_0)
    d_2_list = paraphrase_choices_pipeline(d)
    paraphrase_2 = d_2_list[-1]
    num_tokens += count_tokens(json.dumps(paraphrase_2))
    print(f"Tokens in paraphrase_2: {num_tokens}")

    d_0_1_list = add_question_context_pipeline(paraphrase_0)
    paraphrase_0_1 = d_0_1_list[-1]
    num_tokens += count_tokens(json.dumps(paraphrase_0_1))
    print(f"Tokens in paraphrase_0_1: {num_tokens}")

    d_2_3_list = add_new_choice_pipeline(paraphrase_2)
    paraphrase_2_3 = d_2_3_list[-1]
    num_tokens += count_tokens(json.dumps(paraphrase_2_3))
    print(f"Tokens in paraphrase_2_3: {num_tokens}")

    new_choices, new_answer = ChoicePermuter.permute(paraphrase_2_3["choices"], paraphrase_2_3["answer"])
    paraphrase_2_3_4 = copy.deepcopy(paraphrase_2_3)
    paraphrase_2_3_4["choices"] = new_choices
    paraphrase_2_3_4["answer"] = new_answer

    paraphrase_0_1_2_3_4 = copy.deepcopy(paraphrase_0_1)
    paraphrase_0_1_2_3_4["choices"] = new_choices
    paraphrase_0_1_2_3_4["answer"] = new_answer
    paraphrased_data_0_1_2_3_4.append(paraphrase_0_1_2_3_4)

    with open(f"{results_dir_name}/paraphrased_data_0+1+2+3+4.json", 'w') as file:
        json.dump(paraphrased_data_0_1_2_3_4, file, indent=4)

    print(f"Question {idx} paraphrased")
    print(f"all tokens in paraphrased{idx} paraphrased are {num_tokens}")

print("paraphrased data saved")


