import numpy as np
import torch
import random
from transformers import CLIPTextModel, CLIPTokenizer
from utils import search_min_sentence_iteration, genetic, PGDattack, get_char_table, train, object_key


from diffusers import StableDiffusionPipeline
from torch import autocast
from utils import image_grid

from torchmetrics.multimodal.clip_score import CLIPScore
# import clip
from PIL import Image
import torchvision.transforms as transforms

import pdb


device = 'cuda'

len_prompt = 5

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = text_encoder.to(device)
char_table = get_char_table()

# attack_sentence = "a snake and a young man"

# #  Greedy
# greedy_sentence = search_min_sentence_iteration(attack_sentence, char_table, len_prompt, 
#                                                 1, tokenizer=tokenizer, text_encoder=text_encoder)
# print("greedy_prompt:",greedy_sentence)

# #  Genetic
# for i in range(5):
#     genetic_prompt = genetic(attack_sentence, char_table, len_prompt, tokenizer=tokenizer, 
#                              text_encoder=text_encoder)
#     genetic_sentence = attack_sentence + ' ' + genetic_prompt[0][0]
#     print("genetic_prompt:",genetic_sentence)
    
# #  PGD
# for i in range(5):
#     max_tensor, loss_list, pgd_prompt, max_loss = train(init_per_sample=1, sentence=attack_sentence, 
#                                                         len_prompt=len_prompt, char_list=char_table, 
#                                                         model=text_encoder.text_model, iter_num = 100, 
#                                                         eta_u=1, tokenizer=tokenizer, text_encoder=text_encoder)  
#     pgd_sentence = attack_sentence + ' ' + pgd_prompt
#     print("pgd_prompt:",pgd_sentence)

sentence_list = [
    "A black panther lying in a jungle and a tree",
    "A fishing boat on a lake at sunrise and a tree",
    "A tea cup on a saucer with a teapot and a tree",
    "A man playing guitar on a street corner and a tree",
    "A group of flamingos standing in a pond and a tree",
    "A fireflies in a field at dusk and a tree",
    "A train chugging through a countryside and a tree",
    "A butterfly on a colorful flower and a tree",
    "A soccer game being played on a stadium and a tree",
    "A man kayaking down a river through rapids and a tree"
]
mask = object_key(sentence_list, 'and a tree', thres=9, tokenizer=tokenizer, text_encoder=text_encoder)

# pdb.set_trace()


# attack_sentence = "two apple and a tree"
# mask = mask.view(-1)

# #  Greedy
# greedy_sentence = search_min_sentence_iteration(attack_sentence, char_table, len_prompt, 
#                                                 1, tokenizer=tokenizer, text_encoder=text_encoder,  mask=mask)
# print("greedy_prompt:",greedy_sentence)

# #  Genetic
# for i in range(10):
#     genetic_prompt = genetic(attack_sentence, char_table, len_prompt, tokenizer=tokenizer, 
#                              text_encoder=text_encoder,  mask=mask)
#     genetic_sentence = attack_sentence + ' ' + genetic_prompt[0][0]
#     print("genetic_prompt:",genetic_sentence)
    
# #  PGD
# for i in range(10):
#     max_tensor, loss_list, pgd_prompt, max_loss = train(init_per_sample=1, sentence=attack_sentence, 
#                                                         len_prompt=len_prompt, char_list=char_table, 
#                                                         model=text_encoder.text_model, iter_num = 100, 
#                                                         eta_u=1, tokenizer=tokenizer, text_encoder=text_encoder,  mask=mask)  
#     pgd_sentence = attack_sentence + ' ' + pgd_prompt
#     print("pgd_prompt:",pgd_sentence)

pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4', revision='fp16',
    torch_dtype=torch.float16, use_auth_token=True)
pipe = pipe.to(device)

generator = torch.Generator("cuda").manual_seed(27)

original_sentence = 'two apple and a tree'
perturbation_prompt = 'SRV25'
# sentence = original_sentence + ' ' + perturbation_prompt
sentence = original_sentence

num_images = 5
prompt = [sentence] * num_images
with autocast('cuda'):
    images = pipe(prompt, generator=generator, num_inference_steps=50).images

grid = image_grid(images, rows=1, cols=5)
grid.save("output_grid5.jpg")

# image = Image.open("output_grid4.jpg")
transform = transforms.Compose([
    transforms.PILToTensor()
])



metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
for image in images:
    score = metric(transform(image), "two apple and a tree")
    print(score)
    score.detach()
