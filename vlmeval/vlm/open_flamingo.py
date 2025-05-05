import sys
import torch
from PIL import Image
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import *
from huggingface_hub import snapshot_download


class OpenFlamingo(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 name,
                 mpt_pth=None,
                 ckpt_pth=None,
                 **kwargs):

        if mpt_pth is None:
            raise ValueError(
                'Please set `mpt_pth` to the directory of MPT-7B, which is cloned from here: '
                'https://huggingface.co/mosaicml/mpt-7b. '
                'change to https://huggingface.co/anas-awadalla/mpt-7b'
            )
            raise ValueError
        if ckpt_pth is None:
            raise ValueError(
                'Please set `ckpt_pth` to the openflamingo ckpt, which is the `checkpoint.pt` file downloaded '
                'from: https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b/tree/main. '
            )
        else:
            if osp.exists(ckpt_pth):
                if ckpt_pth.endswith('checkpoint.pt'):
                    pass
                elif osp.isdir(ckpt_pth):
                    ckpt_pth = osp.join(ckpt_pth, 'checkpoint.pt')
                    if not osp.exists(ckpt_pth):
                        raise ValueError(f'File {ckpt_pth} does not exist. ')
            elif splitlen(ckpt_pth, '/') == 2:
                cache_path = get_cache_path(ckpt_pth)
                if cache_path is None:
                    snapshot_download(ckpt_pth)
                cache_path = get_cache_path(ckpt_pth)
                if cache_path is None:
                    raise ValueError(f'Directory {cache_path} does not exist. ')
                else:
                    ckpt_pth = osp.join(cache_path, 'checkpoint.pt')

        self.name = name
        assert name in ['v2']
        self.mpt_pth = mpt_pth
        try:
            from open_flamingo import create_model_and_transforms
        except Exception as e:
            logging.critical('Please first install open_flamingo to use OpenFlamingo')
            raise e

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path='ViT-L-14',
            clip_vision_encoder_pretrained='openai',
            lang_encoder_path=mpt_pth,
            tokenizer_path=mpt_pth,
            cross_attn_every_n_layers=1)
        ckpt = torch.load(ckpt_pth)
        model.load_state_dict(ckpt, strict=False)
        torch.cuda.empty_cache()
        self.model = model.eval().cuda()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.image_proc = image_processor

        kwargs_default = dict(max_new_tokens=5, num_beams=3, length_penalty=0.0)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        vision_x = []
        prompt = ''
        # add ICL, messages can be multi-turn with multiple images
        for msg in message:
            if msg['type'] == 'image':
                img = Image.open(msg['value'])
                vision_x.append(self.image_proc(img).unsqueeze(0))
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += f'Question: {msg["value"]}'
            elif msg['type'] == 'answer':
                prompt += f' Short answer: {msg["value"]}<|endofchunk|>\n'
        
        prompt += ' Short answer:'

        print(f"\033[31m{prompt}\033[0m")
        
        vision_x = torch.cat(vision_x, dim=0) if len(vision_x) > 1 else vision_x[0]
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        lang_x = self.tokenizer([prompt], return_tensors='pt')
        generated_text = self.model.generate(
            vision_x=vision_x.cuda(),
            lang_x=lang_x['input_ids'].cuda(),
            attention_mask=lang_x['attention_mask'].cuda(),
            **self.kwargs)

        text = self.tokenizer.decode(generated_text[0][len(lang_x['input_ids'][0]):], skip_special_tokens=True)

        return text
    
    def call_inner(self, message, dataset=None, output_hidden_states=False, output_attentions=False):
        vision_x = []
        prompt = ''
        # add ICL, messages can be multi-turn with multiple images
        for msg in message:
            if msg['type'] == 'image':
                img = Image.open(msg['value'])
                vision_x.append(self.image_proc(img).unsqueeze(0))
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value']
            elif msg['type'] == 'answer':
                prompt += f'Answer: {msg["value"]}'
        
        prompt += 'Answer: '

        print(f"\033[31m{prompt}\033[0m")
        
        vision_x = torch.cat(vision_x, dim=0) if len(vision_x) > 1 else vision_x[0]
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        lang_x = self.tokenizer([prompt], return_tensors='pt')

        outputs = self.model(
            vision_x=vision_x.cuda(),
            lang_x=lang_x['input_ids'].cuda(),
            attention_mask=lang_x['attention_mask'].cuda(),
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        
        return outputs


if __name__ == "__main__":
    import requests

    # Example usage:
    vision_encoder_path = "ViT-L-14"
    mpt_pth="anas-awadalla/mpt-1b-redpajama-200b"
    ckpt_pth="/nethome/chuang475/flash/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt"
    # lm_path = "anas-awadalla/mpt-1b-redpajama-200b"
    # checkpoint_path = "/nethome/chuang475/flash/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt"
    # lm_tokenizer_path = "anas-awadalla/mpt-1b-redpajama-200b"
    cross_attn_every_n_layers = 1
    vision_encoder_pretrained = "openai"
    precision = "amp_bf16"

    device = "cuda:0"

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)

    samples = {
        "image_raw": [[image], [image], [image], [image]],
        "text_input_raw": ["What is this?", "Is there a car?", "What color is the car?", "What is the brand of the car?"],
    }

    with torch.inference_mode():
        model = OpenFlamingo(
            name='v2',
            mpt_pth=mpt_pth,
            ckpt_pth=ckpt_pth,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
        )
        sample = [
            {'type': 'image', 'value': '/nethome/chuang475/LMUData/images/TextVQA_VAL/38250.jpg'},
            {'type': 'text', 'value': 'when was this paper published?'}
            ]
        output = model.generate_inner(sample, max_generation_length=5)
        print(output)