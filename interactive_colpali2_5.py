# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:01:50 2025

@author: fsaby
https://github.com/spmallick/learnopencv/blob/master/Multimodal-RAG-with-ColPali-Gemini/Final-ColPali-with-patch-activations-.ipynb
"""
from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
import os
from pdf2image import convert_from_path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
import pickle
from pathlib import Path

def load_rag(model_name = "vidore/colqwen2.5-v0.2"):
    print("###Load RAG###")
    device_rag = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_rag = ColQwen2_5.from_pretrained(model_name,
                                              torch_dtype=torch.float16, # set the dtype to bfloat16
                                              device_map=device_rag,
                                              attn_implementation="sdpa").eval()  
    processor_rag = ColQwen2_5_Processor.from_pretrained(model_name)
    return model_rag, processor_rag,device_rag


def load_llm(model_name="Qwen/Qwen2.5-VL-7B-Instruct",min_pixels = 3 * 128 * 128,max_pixels = 3 * 512*512):
    print("###Load LLM###")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Set False for 8-bit
        bnb_4bit_compute_dtype=torch.float16
    )
    device_llm = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=device_llm,
        attn_implementation="sdpa"
    ).eval() 
    processor_llm = AutoProcessor.from_pretrained(model_name,min_pixels=min_pixels, max_pixels=max_pixels)
    
    return model_llm,processor_llm,device_llm


def create_database(path_files, model_rag, processor_rag, device_rag):
    
    path=Path(path_files)

    output_path="./rag_database/"+path.name+".pkl"
    os.makedirs("./rag_database/", exist_ok=True)
    
    files = [os.path.join(path_files, file) for file in os.listdir(path_files) if file.lower().endswith('.pdf')]
    
    images = []
    image_metadata = []  # To store (filename, page_number)
    document_embeddings = []

    for file in tqdm(files,desc="Indexing"):
        pdf_images = convert_from_path(file)
        images.extend(pdf_images)
        # Add filename and page number metadata
        image_metadata.extend([(os.path.basename(file), i + 1) for i in range(len(pdf_images))])

    # Create DataLoader
    dataloader = DataLoader(
        images,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor_rag.process_images(x)
    )

    for batch in tqdm(dataloader,desc="Embedding"):
        with torch.no_grad():
            batch = {key: value.to(device_rag) for key, value in batch.items()}
            embeddings = model_rag(**batch)
        document_embeddings.extend(list(torch.unbind(embeddings.to("cpu"))))

    # Memory usage reporting
    total_memory = sum(e.element_size() * e.nelement() for e in document_embeddings)
    print(f'Total Embedding Memory (CPU): {total_memory / 1024 ** 2:.2f} MB')

    total_image_memory = sum(img.width * img.height * 3 for img in images)
    print(f'Total Image Memory: {total_image_memory / 1024 ** 2:.2f} MB')
    output_data = {
        "embeddings": document_embeddings,
        "images": images,
        "metadata": image_metadata,
    }
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"Saved indexed data to: {output_path}")

def load_database(path_files):
    print("###Load database###")
    path = Path(path_files)
    pickle_path = f"./rag_database/{path.name}.pkl"

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"No database found at: {pickle_path}")

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    return data["embeddings"], data["images"], data["metadata"]
    
  
def retrieve_top_document(model_rag,processor_rag,device_rag,query,document_embeddings,document_images,image_metadata,k=3,show=True):
    query_embeddings = []
    # Create a placeholder image
    #placeholder_image = Image.new("RGB", (448, 448), (255, 255, 255))

    with torch.no_grad():
        # Process the query to obtain embeddings
        query_batch = processor_rag.process_queries([query]) #,placeholder_image)
        query_batch = {key: value.to(device_rag) for key, value in query_batch.items()}
        query_embeddings_tensor = model_rag(**query_batch)
        query_embeddings = list(torch.unbind(query_embeddings_tensor.to("cpu")))

    # Evaluate the embeddings to find the most relevant document

    similarity_scores = np.nan_to_num(processor_rag.score_multi_vector(query_embeddings, document_embeddings), nan=-np.inf)
    # Get the indices of the top k documents (sorted in descending order of similarity)
    top_k_indices = np.argsort(similarity_scores[0])[-k:]  # [0] because similarity_scores is likely a 1D array for a single query
    # Collect the top k documents' metadata
    top_k_images = [document_images[idx] for idx in top_k_indices]
    top_k_filenames = [image_metadata[idx][0] for idx in top_k_indices]
    top_k_page_nbs = [image_metadata[idx][1] for idx in top_k_indices]
    if show:
        plt.rcParams['savefig.pad_inches'] = 0
        for i, idx in enumerate(top_k_indices):
            ax = plt.axes([0, 0, 1, 1], frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(document_images[idx])
            plt.title(f"Rank {i+1}: Page {top_k_page_nbs[i]} from file: {top_k_filenames[i]}")
            plt.show()
            print(f"Rank {i+1}: Best match is page {top_k_page_nbs[i]} from file: {top_k_filenames[i]}")
    # Return the best matching document text and image
    return top_k_images, top_k_indices.tolist(), top_k_filenames, top_k_page_nbs

def send_request(query,model_llm, processor,image_lst=[],video_lst=[], previous_chat=None,device_llm='cpu'):
  max_length=2000
  temperature=0.9
  if previous_chat==None:
      previous_chat=[{"role": "system", "content": "You are a helpful assistant."}]
  history=list(previous_chat)
  img_placeholders = [{"type": "image", "image": url} for url in image_lst]
  video_placeholders = [{"type": "video", "video": url,"fps": 1} for url in video_lst]  
  history.append({"role": "user","content": [*img_placeholders,*video_placeholders,{"type": "text","text": query}]})
  # Apply the chat template
  text = processor.apply_chat_template(
      history, tokenize=False, add_generation_prompt=True
  )

  image_inputs, video_inputs = process_vision_info(history)

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
  #keep only text in history
  history=list(previous_chat)
  history.append({"role": "user",
                  "content": [{"type": "text","text": query}]})
  history.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": output_text
            },
        ],
  })

  return output_text, history




if __name__ == '__main__':
    r"""
    query="What is motility A?"
    model_rag,processor_rag,device_rag=load_rag()
    #create_database(r"C:\Users\fsaby\Desktop\vlm\Myxo_db",model_rag,processor_rag,device_rag)
    document_embeddings,images,image_metadata=load_database(r"C:\Users\fsaby\Desktop\vlm\Myxo_db")
    image,index,filename,page_nb=retrieve_top_document(model_rag,processor_rag,device_rag,query,document_embeddings,images,image_metadata)
    model_llm,processor_llm,device_llm=load_llm()
    query="Use only the information provided in the image to answer the following query.\n query:"+query
    output_text, history=send_request(query,model_llm, processor_llm,image_lst=image,video_lst=[], previous_chat=None,device_llm=device_llm)
    print(output_text)
    """
    # Chargement des modèles et base de données
    model_rag, processor_rag, device_rag = load_rag()
    document_embeddings, images, image_metadata = load_database(r"C:\Users\fsaby\Desktop\vlm\Myxo_db")
    model_llm, processor_llm, device_llm = load_llm()
    history=None
    print("Send your question (or 'exit' to leave):")
    while True:
        query = input("\nVotre question: ")
        if query.lower() in ['exit', 'quit']:
            print("Fin du programme.")
            break

        try:
            # Étape 1 : Récupération de l'image la plus pertinente via le RAG
            image, index, filename, page_nb = retrieve_top_document(
                model_rag, processor_rag, device_rag,
                query, document_embeddings, images, image_metadata
            )

            # Étape 2 : Construction de la requête pour le LLM
            prompt = f"Use only the information provided in the images to answer the following query.\nQuery: {query}"

            # Étape 3 : Envoi au modèle LLM avec l'image
            output_text, history = send_request(
                prompt, model_llm, processor_llm,
                image_lst=image, video_lst=[],
                previous_chat=history, device_llm=device_llm
            )

            # Affichage de la réponse
            print("\nRéponse:")
            print(output_text)


        except Exception as e:
            print(f"Une erreur est survenue: {e}")
    
