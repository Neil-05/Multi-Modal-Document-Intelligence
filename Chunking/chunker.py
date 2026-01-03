import json 
from pathlib import Path
import uuid

max_Chars=2000

def chunk_text(text,max_chars= max_Chars):
    paragraphs=[p.strip() for p in text.split("\n") if p.strip()]
    chunks=[]

    current=""

    for para in paragraphs:
        if len(current) + len(para) <=max_chars:
            current+=" "+para
        else:
            chunks.append(current.strip())
            current=para
    if current:
        chunks.append(current.strip())
    return chunks

def chunk_content(in_path,out_path):
    with open(in_path,"r", encoding="utf-8") as f:
        data=json.load(f)
    chunked_data=[]
    
    for item in data:
        modality= item["modality"]
        page=item["page"]
        source= item["source"]
        content=item["content"]


        if modality =="text":
            text_chunks=chunk_text(content)
            for ch in text_chunks:
                chunked_data.append({
                    "chunk_id": str(uuid.uuid4()),
                    "content" : ch,
                    "page" : page,
                    "modality": modality,
                    "source" : source
                })
        else:
            chunked_data.append({
                "chunk_id":str(uuid.uuid4()),
                "content": content.strip(),
                "page" : page,
                "modality" : modality,
                "source" : source
            })
    with open(out_path, "w",encoding="utf-8") as f:
        json.dump(chunked_data,f,indent=2,ensure_ascii=False)
    
    print(f"✅ Created {len(chunked_data)} chunks")

if __name__ == "__main__":
    chunk_content(
        "data/processed/all_data.json",
        "data/processed/chunks.json"
    )