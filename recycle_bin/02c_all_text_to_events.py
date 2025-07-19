import pandas as pd
import os, sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils.nlp_tools as nlp_tools
nlp = nlp_tools.TextLib("en_core_web_lg")
def extract_sentences(text):
    sentences_raw = nlp.sentence_splitter(text,span=False)
    sentences = [sent['text'] for sent in sentences_raw]
    return sentences 

def clean_sentences(list_of_sentences):
    result_list_of_sentences = []
    for full_sentence in list_of_sentences:
        splitted_sentences = full_sentence.split(";")
        for sentence in splitted_sentences:
            sentence = sentence.strip().lower().rstrip('.').replace("  "," ")
            result_list_of_sentences.append(sentence)
    return result_list_of_sentences

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

notes_selected = pd.read_pickle("../data/NOTEEVENTS_NURSINGNOTES_REMOVED_ERROR_STRINGS_FILLED_ABBREVIATIONS.pkl").iloc[:10]
notes_selected['Sentences'] = [[]]*len(notes_selected)
print(f"Time to extract sentences from 100 reports: ")
import time
start_time = time.time()
notes_selected.loc[:100,'Sentences'] = notes_selected['TEXT'].iloc[:100].apply(extract_sentences)
end_time = time.time()
time_per_100 = end_time - start_time
print(f"Time taken: {time_per_100} seconds")
print(f" projected time for {len(notes_selected)} reports: {time_per_100 * (len(notes_selected) / 100) / 60} minutes")
print("extracting sentences for all reports...")
notes_selected.loc[:,'Sentences'] = notes_selected['TEXT'].apply(extract_sentences)
notes_selected['Sentences_Cleaned'] = notes_selected['Sentences'].apply(clean_sentences)
notes_selected.to_pickle("../data/ALL_NURSING_NOTES_SENTENCES.pkl")





extractor = EventExtractor(event_name_model_type="biolord", attribute_model_type="None")

def extract_events(sentences):
    global extractor, report_counter
    report_counter+=1
    event_types = ["Pain", "Sleep", "Alert And Oriented", "Excretion", "Eating", "Family"]
    events = extractor.extract_events(sentences=sentences, event_names=event_types, threshold=0.2)
    if report_counter<=10:
        print(events)
    return events

export_folder = "../exports/all_nursing_reports_with_event_log_only_biolord"
os.makedirs(export_folder,exist_ok=True)
batch_size = 1000


notes_selected["Events"] = ''
for i in range(0, len(notes_selected), batch_size):
    print(i)
    batch = notes_selected.iloc[i:i+batch_size]
    batch["Events"] = batch['Sentences_Cleaned'].apply(extract_events)
    batch.to_pickle(f"{export_folder}/batch_{i//batch_size:08d}.pkl")
batch_files = sorted(glob.glob(f"../{export_folder}/batch_*.pkl"))
if len(batch_files) == 0:
    batch.to_pickle(f"{export_folder}/combined.pkl")
else:
    combined_df = pd.concat([pd.read_pickle(f) for f in batch_files], ignore_index=True)
    combined_df.to_pickle(f"..{export_folder}/combined.pkl")
