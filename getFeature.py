import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import argparse
import torch

import sys
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
from tqdm import tqdm
from collections import defaultdict

import json
import pandas as pd



def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def generate_feature(question_type, image_root, tensor_root, image_type, model, tokenizer, args, image_processor):
    # print("generate feature")
    conv_mode = "mplug_owl2"

    if question_type == 1: #common sense / coherence
        inp = "Does the image convey valid and clear semantic information?"
        conv = conv_templates[conv_mode].copy()
        inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif question_type == 2: # semantic content
        inp = "Do the visual elements of the image present a natural coherent whole?"
        conv = conv_templates[conv_mode].copy()
        inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        raise NotImplementedError

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    image_tensors = []
    image_names = []

    image_name_list = os.listdir(image_root)
    for i, image_name in enumerate(tqdm(image_name_list, desc=f"{image_type}/{question_type}")):
        imag_path = os.path.join(image_root, image_name)
        image = load_image(imag_path)
        
        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)
        
        image_tensors.append(image_tensor)
        image_names.append(image_name)
        
        if i % 8 == 7 or i == len(image_name_list) - 1:
            with torch.inference_mode():
                output_logits = model(input_ids.repeat(len(image_tensors), 1), images=torch.cat(image_tensors, 0))["logits"]
                output_tensors = model(input_ids.repeat(len(image_tensors), 1), images=torch.cat(image_tensors, 0), output_hidden_states=True)['hidden_states'][-1]
                output_tensors = torch.mean(output_tensors, dim = 1, keepdim=True) #torch.Size([8, 1, 4096])
            for j, image_name in enumerate(image_names):
                tensor = output_tensors[j]
                image_name_head, _ = os.path.splitext(image_name)
                
                if question_type == 1:
                    tensor_save_path = os.path.join(tensor_root, image_name_head + "_coherence.pt")
                else:
                    tensor_save_path = os.path.join(tensor_root, image_name_head + "_semantic_content.pt")
                
                torch.save(tensor, tensor_save_path)
                image_tensors = []
                image_names = []

def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    model_type_list = ["ADM", "BigGAN", "Glide", "Midjourney", "sd_1.5", "VQDM", "wukong"]
    image_type_list = ["val/1_fake", "val/0_real"]
    
    
    for model_type in model_type_list:
        basic_image_root = f"/data3/chenweiyan/dataset/aigc-detection-dataset/genimage/{model_type}"
        basic_tensor_root = "/data3/chenweiyan/dataset/aigc-detection-dataset/genimage/{model_type}/tensor-root"
        
        for image_type in image_type_list:
            print(f"Processing {model_type}/{image_type}...")
            image_root = os.path.join(basic_image_root, image_type)
            tensor_root = os.path.join(basic_tensor_root, image_type)
            os.makedirs(tensor_root, exist_ok=True)
            generate_feature(question_type=1, image_root=image_root, tensor_root=tensor_root, image_type=image_type, model=model, tokenizer=tokenizer, args=args, image_processor=image_processor)
            generate_feature(question_type=2, image_root=image_root, tensor_root=tensor_root, image_type=image_type, model=model, tokenizer=tokenizer, args=args, image_processor=image_processor)
        
    # for json_ in jsons:
    #     with open(json_) as f:
    #         iqadata = json.load(f)  

    #         image_tensors = []
    #         batch_data = []
    #         if question_type == 1:
    #             json_ = json_.replace("combined/", "combined-")
    #             with open(f"results/{args.model_path}/{json_.split('/')[-1]}", "w") as wf:
    #                 wf.write("[\n")
    #         for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
    #             try:
    #                 filename = llddata["name"]
    #             except:
    #                 filename = llddata["img_path"]
    #             llddata["logits"] = defaultdict(float) # 添加一个 "logits" 键 原本存在name, prompt, mos
    #             imag_path = os.path.join(img_root, filename)
    #             image = load_image(imag_path) # load image here
    #             def expand2square(pil_img, background_color):
    #                     width, height = pil_img.size
    #                     if width == height:
    #                         return pil_img
    #                     elif width > height:
    #                         result = Image.new(pil_img.mode, (width, width), background_color)
    #                         result.paste(pil_img, (0, (width - height) // 2))
    #                         return result
    #                     else:
    #                         result = Image.new(pil_img.mode, (height, height), background_color)
    #                         result.paste(pil_img, ((height - width) // 2, 0))
    #                         return result
    #             image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    #             image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)

    #             image_tensors.append(image_tensor)
    #             batch_data.append(llddata)

    #             if i % 8 == 7 or i == len(iqadata) - 1: # 相当于控制batch_size为8                    
    #                 with torch.inference_mode():
    #                     output_logits = model(input_ids.repeat(len(image_tensors), 1),
    #                         images=torch.cat(image_tensors, 0))["logits"] #(batch_size, sequence_length, config.vocab_size) torch.Size([8, 148, 32000]) sequence随着输入问题改变而改变
    #                     # print("i:", i," output_logits:", output_logits.shape)
    #                     output_tensors = model(input_ids.repeat(len(image_tensors), 1),
    #                         images=torch.cat(image_tensors, 0), output_hidden_states=True)['hidden_states'][-1]
    #                     output_tensors = torch.mean(output_tensors, dim = 1, keepdim=True) #torch.Size([8, 1, 4096])
    #                     # print("i:", i," outputs:", outputs)
    #                     # print("len1:", len(outputs), "len2:", len(outputs[0]), "len3:", len(outputs[0][0]), "len4:", len(outputs[0][0][0])) #list of 33 tensors torch.Size([8, 148, 4096])
    #                 for j, data in enumerate(batch_data):
    #                     try:
    #                         img_name = data["name"]
    #                     except:
    #                         img_name = data["img_path"]
    #                     if question_type == 1:
    #                         tensor = output_tensors[j] #torch.Size([1, 4096])
    #                         if img_name.endswith(".jpg"):
    #                             save_path = os.path.join(args.tensor_root, img_name.replace(".jpg", "") + "_coherence.pt")
    #                         elif img_name.endswith(".png"):
    #                             save_path = os.path.join(args.tensor_root, img_name.replace(".png", "") + "_coherence.pt")
    #                         else:
    #                             raise ValueError("img_name should end with .jpg or .png")
    #                          #common sense
    #                         torch.save(tensor, save_path) 
    #                         # print("tensor: ", tensor.shape, "save_path: ", save_path)
    #                     elif question_type == 2:
    #                         tensor = output_tensors[j]
    #                         if img_name.endswith(".jpg"):
    #                             save_path = os.path.join(args.tensor_root, img_name.replace(".jpg", "") + "_semantic_content.pt")
    #                         elif img_name.endswith(".png"):
    #                             save_path = os.path.join(args.tensor_root, img_name.replace(".png", "") + "_semantic_content.pt")
    #                         else:
    #                             raise ValueError("img_name should end with .jpg or .png")
    #                         torch.save(tensor, save_path)


                            
    #                 image_tensors = []
    #                 batch_data = []



if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/AGIQA_3k/MA_AGIQA.yaml', help="config file")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()

    # Load configurations from the provided config file
    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load('configs/base.yaml')
    config = OmegaConf.merge(base, cfg)

    # Merge command line arguments into the config
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key not in config or value is not None:
            config[key] = value

    main(config)
