import pandas as pd
import os, sys
from datetime import datetime
from glob import glob

from sklearn.metrics import f1_score, precision_score, recall_score

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
import os
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
importlib.reload(utils.event_extractor)
from utils.event_extractor import EventExtractor  # re-import your class if needed

suffix = "7_14_days"
version = 2
report_counter = 0
def prepare_df(df, type="biolord"):
    df = df.copy()
    print("Loaded file len: ",len(df))
    df['Sent_ID'] = df['Events'].apply(lambda x: [f"{i:04d}" for i in range(len(x))])
    df = df.explode(["Sent_ID","Events"])
    print("After exploding file len: ",len(df))
    df['UID'] = df['ROW_ID'].astype(str) + "_" + df['Sent_ID'].astype(str)
    df = df.dropna(subset="Events")
    df['Event_Name'] = df['Events'].apply(lambda x: x['event'])
    df['Sentence'] = df['Events'].apply(lambda x: x['sentence'])
    df['Time'] = df['Events'].apply(lambda x: x['event_detection_time'])
    
    if type == "dictionary":
        df['Keyword'] = df['Events'].apply(lambda x: x['keyword'])
        df['Lemma'] = df['Events'].apply(lambda x: x['lemma'])
    if type == "biolord":
        df['Similarity'] = df['Events'].apply(lambda x: x['similarity'])
        df['Similarity'] = df['Similarity'].apply(lambda x: {k:v for (k,v) in x.items() if k!="Alert And Oriented"})
        df["Sleep_similarity"] = df['Similarity'].apply(lambda x:x["Sleep"])
        df["Pain_similarity"] = df['Similarity'].apply(lambda x:x["Pain"])
        df["Excretion_similarity"] = df['Similarity'].apply(lambda x:x["Excretion"])
        df["Eating_similarity"] = df['Similarity'].apply(lambda x:x["Eating"])
        df["Family_similarity"] = df['Similarity'].apply(lambda x:x["Family"])
        # df["Alert And Oriented_similarity"] = df['Similarity'].apply(lambda x:x["Alert And Oriented"])
    return df



def extract_events(model,sentences):
    # print(sentences)
    global extractor, report_counter, version
    report_counter+=1
    print(report_counter)
    event_types = ["Pain", "Sleep", "Excretion", "Eating", "Family"]
    event_description_dict = {"Eating":["Patient had breakfast"],
                              "Excretion":['Patient is experiencing variable urinary and bowel output, including episodes of incontinence and some preserved voiding ability.','Foley catheter or diaper is often used to manage incontinence; urine characteristics range from clear to purulent or bloody depending on the case.','Bowel elimination supported pharmacologically with agents like lactulose and senna; some improvement noted during shift.'], 
                              "Family":["A visit, call or communication with a member of the family"], #Interaction with a family member
                              "Pain":["The reporting of pain or an observation of pain signals by the doctor/nurse"],
                              "Sleep":['Patient is intermittently alert, oriented, and able to follow commands, with motor activity preserved on at least one side.',    'Vital signs are generally stable with some labile blood pressure during sleep and periods of agitation, but responsive to interventions.',    'Patient often sleeps in short naps with improvement after medications like lorazepam, percocet, or roxicet; oxygenation mostly adequate with supportive therapy.']
                              }
    anti_event_description_dict = {"notEating":["bite tube"],
                              "notExcretion":['Stool and urine samples were sent for evaluation, indicating concern for infection or other pathology.','Fecal containment devices like rectal bags used in anticipation of stool incontinence or during limited mobility.','Some interventions like limiting fluids or foley placement were done or considered for symptom control.'],
                              "notFamily":["Colleague visited the patient"], 
                              "notPain":["It want to eat"],
                              "notSleep":['Patient has frequent difficulty sleeping, often requesting or requiring sleep aids.','Concerns exist around use of sedatives due to history of psychosis or hallucinations related to benzodiazepines.','Sleep apnea and desaturation during sleep are contributing to clinical concerns, especially regarding cardiac stress.']
                              }
    event_description_dict.update(anti_event_description_dict)
    
    if model == "biolord":
        
        extractor = EventExtractor(event_name_model_type=model, attribute_model_type="None")
        events = extractor.extract_events(sentences=sentences, 
                                          event_names=[f"{k} : {v}" for (k,vl) in event_description_dict.items() for v in vl],  
                                          threshold=0.2,embedder_method=2)
    else:
        extractor = EventExtractor(event_name_model_type=model, attribute_model_type="None",dictionary_file=dictionary_file)
        events = extractor.extract_events(sentences=sentences, 
                                          event_names=event_types, 
                                          threshold=0.2)
    assert(len(sentences)==len(events)), f"{len(events)}, {events}, {len(sentences)}, {sentences}"
    return events

version=2
for ET in ["Sleep", "Excretion"]:
    os.makedirs(f"../exports/iter2/{ET}",exist_ok=True)

    for model in ["dictionary"]:
        print(f"{ET}_{model}")
        if model == "dictionary":
            dictionary_file = "../resources/keyword_dict_annotated_iter2.xlsx"
        else:
            dictionary_file = "../resources/keyword_dict_annotated.xlsx"
        
        export_folder = f"../exports/selected_reports_with_event_log_only_{model}_v{version}"
        os.makedirs(export_folder,exist_ok=True)
      
        notes_selected = pd.read_excel(glob(f"../exports/groundtruth/T-SET/Annotated/{ET}*.xlsx")[0])

        notes_selected = notes_selected.groupby("UID")[[f"{ET}_similarity", f"gt_{ET}", "is_keyword_present", "Sentence_dictionary","Lemma"]].agg(lambda x: max(x) if len(set(x))>1 else set(x).pop()).reset_index()
        notes_selected["Events"] = extract_events(model,notes_selected.Sentence_dictionary)
        notes_selected["Keyword_dictionary_New"] = notes_selected["Events"].apply(lambda x: x['keyword'])
        notes_selected["Event_Name_dictionary_New"] = notes_selected["Events"].apply(lambda x: x['event'])
        notes_selected["is_keyword_present_new"] = notes_selected["Events"].apply(lambda x: ET in x['event'])
        # notes_selected = notes_selected.groupby("UID")[[f"{ET}_similarity", f"gt_{ET}", "is_keyword_present", "Sentence_dictionary"]].agg(lambda x: max(x) if len(set(x))>1 else set(x).pop()).reset_index()
        # notes_selected.dropna(subset=f"gt_{ET}",inplace=True)
        y_true = notes_selected[f"gt_{ET}"].astype(int)
        y_pred = notes_selected["is_keyword_present"].astype(int)
        y_pred2 = notes_selected["is_keyword_present_new"].astype(int)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f12 = f1_score(y_true, y_pred2)
        precision2 = precision_score(y_true, y_pred2)
        recall2 = recall_score(y_true, y_pred2)
        print(y_true.value_counts())
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix (y_true vs y_pred):")
        print(cm)
        print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        cm = confusion_matrix(y_true, y_pred2)
        print("Confusion Matrix (y_true vs y_pred2):")
        print(cm)
        print(f"F1 Score: {f12:.4f}, Precision: {precision2:.4f}, Recall: {recall2:.4f}")
        # Build 2x2 contingency table for McNemar's test
        tb = pd.crosstab(y_pred, y_pred2, rownames=['y_pred'], colnames=['y_pred2'])
        # Ensure the table is 2x2
        tb = tb.reindex(index=[0,1], columns=[0,1], fill_value=0)
        result = mcnemar(tb, exact=True)
        print("McNemar's test statistic=%.4f, p-value=%.4f" % (result.statistic, result.pvalue))
        notes_selected.to_excel(f"../exports/iter2/{ET}/{model}.xlsx")
        
        
        

