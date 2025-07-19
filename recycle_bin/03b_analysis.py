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

def clean_sentences(list_of_sentences):
    result_list_of_sentences = []
    for full_sentence in list_of_sentences:
        splitted_sentences = full_sentence.split(";")
        for sentence in splitted_sentences:
            sentence = sentence.strip().lower().rstrip('.').replace("  "," ")
            result_list_of_sentences.append(sentence)
    return result_list_of_sentences

extractor = EventExtractor(event_name_model_type="biolord", attribute_model_type="None")

def extract_events(sentences):
    global extractor, report_counter
    report_counter+=1
    event_types = ["Pain", "Sleep", "Alert And Oriented"]
    events = extractor.extract_events(sentences=sentences, event_names=event_types, threshold=0.2)
    if report_counter<=10:
        print(events)
    return events

os.makedirs("../exports/filtered_patient_reports_with_event_log_only_embedders/",exist_ok=True)
batch_size = 10
filtered_reports_df = pd.read_pickle("../exports/filtered_patient_reports.pkl").iloc[:10]
filtered_reports_df['Sentences_Cleaned'] = filtered_reports_df['Sentences'].apply(clean_sentences)
filtered_reports_df["Events"] = ''
for i in range(0, len(filtered_reports_df), batch_size):
    print(i)
    batch = filtered_reports_df.iloc[i:i+batch_size]
    batch["Events"] = batch['Sentences_Cleaned'].apply(extract_events)
    batch.to_pickle(f"../exports/filtered_patient_reports_with_event_log_only_embedders/batch_{i//batch_size:08d}.pkl")
# batch_files = sorted(glob.glob("../exports/filtered_patient_reports_with_event_log_only_embedders/batch_*.pkl"))
# combined_df = pd.concat([pd.read_pickle(f) for f in batch_files], ignore_index=True)
# combined_df.to_pickle("../exports/filtered_patient_reports_with_event_log_only_embedders/combined.pkl")
