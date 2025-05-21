# # from transformers import AutoTokenizer, AutoModelForCausalLM

# # tokenizer = AutoTokenizer.from_pretrained("openflamingo/OpenFlamingo-9B-vitl-mpt7b")
# # model = AutoModelForCausalLM.from_pretrained("openflamingo/OpenFlamingo-9B-vitl-mpt7b")  # openflamingo/OpenFlamingo-9B-vitl-mpt7b

# # import pandas as pd
# # from vlmeval.smp import *

# # # df = load("./outputs/VLM-R1/VLM-R1_ScienceQA_TRAIN_openai_result.xlsx")
# # possible_result_files = "outputs/VLM-R1/T20250505_G802c153f/VLM-R1_ScienceQA_TRAIN_QCME_openai_result.xlsx"
# # if osp.exists(possible_result_files):
# #     df = load(possible_result_files)
# #     df = load(possible_result_files)
# # print(df.head())

# import copy

# def can_infer_option(answer, choices):
#     # Choices is a dictionary
#     if 'Failed to obtain answer via API' in answer:
#         return False

#     reject_to_answer = [
#         "Sorry, I can't help with images of people yet.",
#         "I can't process this file.",
#         "I'm sorry, but without the image provided",
#         'Cannot determine the answer'
#     ]
#     for err in reject_to_answer:
#         if err in answer:
#             return 'Z'

#     def count_choice(splits, choices, prefix='', suffix=''):
#         cnt = 0
#         for c in choices:
#             if prefix + c + suffix in splits:
#                 cnt += 1
#         return cnt

#     answer_mod = copy.copy(answer)
#     chars = '.()[],:;!*#{}'
#     for c in chars:
#         answer_mod = answer_mod.replace(c, ' ')

#     splits = [x.strip() for x in answer_mod.split()]
#     count = count_choice(splits, choices)

#     if count == 1:
#         for ch in choices:
#             if 'A' in splits and len(splits) > 3:
#                 return False
#             if ch in splits:
#                 return ch
#     elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
#         return 'Z'
#     return False

# if __name__ == "__main__":
#     # Example usage
#     answer = "B. This is the answer."
#     choices = {'A', 'B', 'C', 'D'}
#     result = can_infer_option(answer, choices)
#     print(f"Result: {result}")

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f(x):
    a = max(x[0], x[1])
    b = x[2] / x[3]
    c = x[4] + x[5]
    u = 5 * a * b - c
    v = np.log(u) + 0.5
    return sigmoid(v)

def grad_f(x):
    # Compute intermediate variables
    a = max(x[0], x[1])
    a_index = 0 if x[0] > x[1] else 1
    b = x[2] / x[3]
    c = x[4] + x[5]
    u = 5 * (a * b - c)
    v = np.log(u) + 0.5
    s = sigmoid(v)
    ds_dv = s * (1 - s)
    dv_du = 1 / u
    du = np.zeros(6)

    # Partial derivatives of u
    if a_index == 0:
        da_dx1 = 1
        da_dx2 = 0
    else:
        da_dx1 = 0
        da_dx2 = 1

    du[0] = 5 * da_dx1 * b
    du[1] = 5 * da_dx2 * b
    du[2] = 5 * a * (1 / x[3])
    du[3] = -5 * a * x[2] / (x[3] ** 2)
    du[4] = -1
    du[5] = -1

    grad = ds_dv * dv_du * du
    return grad

# Evaluate at the given point
x_hat = np.array([-1.0, 3.0, 4.0, 5.0, -5.0, 7.0])
gradient = grad_f(x_hat)
print("Gradient at x_hat:", gradient)
