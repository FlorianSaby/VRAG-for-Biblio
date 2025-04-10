# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:37:45 2024
@author: florian saby
@mail: flo.saby@hotmail.fr
"""
# Import libraries
from qwen_vl_utils import process_vision_info
from pathlib import Path
from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path
import pickle
import os
import shutil



def create_new_database(index_name="content"):
    folder_path="./"+index_name+"/"
    os.makedirs("./rag_database/", exist_ok=True)
    # Initialize RAGMultiModalModel
    index_name=os.path.basename(os.path.normpath(folder_path))
    model = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0",verbose=0)
    model.index(input_path=Path(folder_path),
        index_name=index_name,
        store_collection_with_index=False,
        overwrite=True,
    )
    with open("./rag_database/"+index_name+'.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def add_file_to_db(db_name="content",file_path="./content/MoA.pdf"):
    with open("./rag_database/"+db_name+'.pickle', 'rb') as handle:
        model = pickle.load(handle)
    
    model.add_to_index(file_path,
                       store_collection_with_index=False)
    
    with open("./rag_database/"+db_name+'.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


max_length=2000
temperature=0.9
# Define a function for send request to the model
def send_request(query,model_llm, processor,image_lst=[],video_lst=[], previous_chat=None,device_llm='cpu'):
  if previous_chat:
      messages=previous_chat
  else: 
      messages=[{"role": "system", "content": "You are a helpful assistant."}]

  img_placeholders = [{"type": "image", "image": url} for url in image_lst]
  video_placeholders = [{"type": "video", "video": url,"fps": 1} for url in video_lst]  
  messages.append({"role": "user","content": [*img_placeholders,*video_placeholders,{"type": "text","text": query}]})
  # Apply the chat template
  text = processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )

  image_inputs, video_inputs = process_vision_info(messages)

  # Tokenize the chat 
  inputs = processor(
      text=[text],
      images=image_inputs,
      videos=video_inputs,
      padding=True,
      return_tensors="pt",
  )
  # Move the tokenized inputs to the same device the model is on (GPU/CPU)
  inputs = inputs.to(device_llm)
  # Generate text from the model
  generated_ids = model_llm.generate(**inputs, max_new_tokens=max_length,                                   
                                   temperature=temperature,  # Increased from 0.7
                                   top_k=50,         # Added top_k sampling
                                   top_p=0.95,       # Added nucleus sampling
                                   do_sample=True ,   # Enable sampling
                                   repetition_penalty=1.05,)
    
  generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
  ]

  # Decode the output back to a string
  output_text = processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
  )[0]
  messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": output_text
            },
        ],
  })

  return output_text, messages

def load_database(db_name):
    with open("./rag_database/"+db_name+'.pickle', 'rb') as handle:
        model = pickle.load(handle)
    return model

def rag(model,query,llm,processor,previous_chat=None,nb_file_to_retrieve=1): 
    def search_similarity(model,query ,k=5,max_nb_file=3):
        def export_relevant_page(file,page_id,name_export="tmp"):
            images = convert_from_path("file")
            images[page_id].save("./tmp/"+name_export+".jpg")
        os.makedirs("./tmp/", exist_ok=True)
        shutil.rmtree("./tmp/")
        os.makedirs("./tmp/", exist_ok=True)
        results = model.search(query, k)
        file_name_lst=model.get_doc_ids_to_file_names()
        image_lst=[]
        pdf_lst="\n"
        for i in range(0,max_nb_file):
            file_id=results[i].doc_id
            page_id=results[i].page_num
            images = convert_from_path(file_name_lst[file_id]) 
            pdf_lst+="- "+file_name_lst[file_id]+" Page "+str(page_id)+"\n <br>"
            images[page_id-1].save("./tmp/"+str(i)+".jpg")
            image_lst.append("./tmp/"+str(i)+".jpg")
        return image_lst,pdf_lst

    image_lst,pdf_lst=search_similarity(model,query ,k=5,max_nb_file=nb_file_to_retrieve)
    prompt_engi="Use only the information provided in the image to answer the following query.\n query:"+query
    output_text,messages=send_request(prompt_engi,model_llm=llm, processor=processor,image_lst=image_lst,previous_chat=previous_chat)
    #print("\033[36m Your query:\033[0m {}\n\033[32m Your answer:\033[0m {}\n \033[33mReferences:\033[0m  {}".format(query, output_text,pdf_lst))
    return output_text,messages,pdf_lst

if __name__ == '__main__':
    None
    #create_new_database(index_name="Myxo_db")


