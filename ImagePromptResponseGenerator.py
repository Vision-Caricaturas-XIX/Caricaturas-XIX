import os
import torch
import requests
import pandas as pd

from PIL import Image
from io import BytesIO
from bert_score import score
from transformers import TextStreamer
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class ImagePromptResponseGenerator:
    def __init__(self, model_path, device='cuda', temperature=0.2, max_new_tokens=512, load_8bit=False, load_4bit=False):
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)

        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def load_image(self, image_file):
        self.conv = conv_templates["llava_v0"].copy()
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            self.image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            self.image = Image.open(image_file).convert('RGB')
        self.image_tensor = process_images([self.image], self.image_processor, self.model.config)
        if type(self.image_tensor) is list:
            self.image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in self.image_tensor]
        else:
            self.image_tensor = self.image_tensor.to(self.model.device, dtype=torch.float16)
        print(f"Image {image_file} loaded succesfully!")

    def generate_response(self, user_prompt):
        if self.image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                self.inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_prompt
            else:
                self.inp = DEFAULT_IMAGE_TOKEN + '\n' + user_prompt
            self.conv.append_message(self.conv.roles[0], self.inp)
            self.image = None
        else:
            # later messages
            self.conv.append_message(self.conv.roles[0], user_prompt)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        #streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=self.image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.conv.messages[-1][-1] = outputs
        return outputs

    def generate_captions(self, folder_path, OCR_instance=None):
        images_paths = [folder_path+"/"+path for path in os.listdir(folder_path) if ".jpg" in path]
        data = []
        for image_path in images_paths:
            self.load_image(image_path)
            #extracted_words = ", ".join(OCR_instance.get_df_results()[OCR_instance.get_df_results()["image"]==image_path.split(".")[0]]["texts"].values[0])
            #data.append({'image': image_path.split(".")[0], 'texts': self.generate_response(f'Describe en español esta imagen. Ten en cuenta que las siguientes palabras se encuentran en la imagen: {extracted_words}. Extiendete')})
            data.append({'image': image_path.split(".")[0], 'texts': self.generate_response("Describe esta imagen en español, incluye los elementos, las personas, los animales, el texto y la interacción entre estos (si es que existen). Extiendete.")})
        self.df_results = pd.DataFrame(data)      

    def get_df_results(self):
        return self.df_results  
    
    def get_bert_score(self):
        desc_df = pd.read_excel('FINAL_DESCRIPTIONS.xlsx')
        desc_df['descripcion ajustada'] = desc_df.apply(lambda x: x['descripcion'] if pd.isna(x['descripcion ajustada']) else x['descripcion ajustada'], axis=1)
        desc_df['image'] = "cropped_processed_images/" + desc_df['nombre'].astype(str)
        joined_df = self.df_results.merge(desc_df, on='image', how='left')
        P, R, F1 = score(joined_df["texts"].tolist(), joined_df["descripcion ajustada"].tolist(), lang="es", verbose=True, rescale_with_baseline=True)
        return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
        }
