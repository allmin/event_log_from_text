import pandas as pd
import os, sys
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils.nlp_tools as nlp_tools
nlp = nlp_tools.TextLib("en_core_web_lg")
def extract_sentences(text):
    sentences_raw = nlp.sentence_splitter(text,span=False)
    sentences = [sent['text'] for sent in sentences_raw]
    return sentences 


import os, sys
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

report_counter = 0
import importlib
import utils.event_extractor  # your module
import glob, os
importlib.reload(utils.event_extractor)
from utils.event_extractor import EventExtractor  # re-import your class if needed
from config import event_types, event_description_dict_embedder, event_description_dict_llm
suffix = "7_14_days"
version = 2
report_counter = 0
def extract_events(sentences, extractor):
    global report_counter, version, event_types, event_description_dict_embedder
    report_counter+=1
    event_description_dict = event_description_dict_embedder
    if extractor.event_name_model_type == "biolord" and version == 2:
        event_types = [f"{k} : {v}" for (k,v) in event_description_dict.items()]
    events = extractor.extract_events(sentences=sentences, event_names=event_types, threshold=0.2)
    assert(len(sentences)==len(events)), f"{len(events)}, {events}, {len(sentences)}, {sentences}"
    return events

#dictionary denotes keyword matching, biolord denotes embedding similarity
for model in ["dictionary","biolord"]:
    extractor = EventExtractor(event_name_model_type=model, attribute_model_type="None")
    export_folder = f"../exports/selected_reports_with_event_log_only_{model}_v{version}"
    os.makedirs(export_folder,exist_ok=True)
    batch_size = 5000
    notes_selected = pd.read_pickle(f"../exports/filtered_patient_reports_{suffix}.pkl")
    notes_selected["Events"] = ''
    
    for i in range(0, len(notes_selected), batch_size):
        print(f"Processing batch {i//batch_size + 1} of {len(notes_selected)//batch_size + 1}  at {datetime.now().strftime('%H:%M:%s')}")
        batch = notes_selected.iloc[i:i+batch_size]
        batch.loc[:,"Events"] = batch['Sentences_Cleaned'].apply(lambda x: extract_events(x, extractor=extractor))
        batch.to_pickle(f"{export_folder}/batch_{i//batch_size:08d}.pkl")
    batch_files = sorted(glob.glob(f"{export_folder}/batch_*.pkl"))
    if len(batch_files) == 0:
        batch.to_pickle(f"{export_folder}/combined.pkl")
    else:
        combined_df = pd.concat([pd.read_pickle(f) for f in batch_files], ignore_index=True)
        combined_df.to_pickle(f"{export_folder}/combined.pkl")
    print(f"Combined file saved to {export_folder}/combined.pkl, at {datetime.now().strftime('%H:%M:%S')}")
