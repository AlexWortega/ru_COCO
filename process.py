import pandas as pd
import csv

from transformers import FSMTForConditionalGeneration, FSMTTokenizer



df = pd.read_json('train_caption.json')
captions = df['caption']
image_ids = df['image_id']
ids = df['id']


mname = "facebook/wmt19-en-ru"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname).to('cuda:0')


def mt(text):
  
  input_ids = tokenizer.encode(text, return_tensors="pt").to('cuda')
  outputs = model.generate(input_ids)
  decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return (decoded)


fieldnames = ['image_id',	'id',	'caption']
with open('captions_ru_1.csv', 'w', newline='') as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        from tqdm import tqdm

        for i in tqdm(range(len(captions)//2, len(captions))):
            
            writer.writerow({'image_id':image_ids[i],'id':ids[i] ,"caption": mt(captions[i]) } )


