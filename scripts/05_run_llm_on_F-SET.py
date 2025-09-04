import pandas as pd
import os, sys
from datetime import datetime
from itertools import product
from glob import glob

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils.event_extractor import EventExtractor
from config import event_types, event_description_dict_llm

def extract_events_funct(sentences, extractor=None, evidence={'keywords':[],'event_names':[],'similarities':[]}, keyword_output=None, phrase_output=None, keyword_input=None, embedder_input=None):
    global event_description_dict_llm, event_types
    events = extractor.extract_events(sentences=sentences, event_names=event_types, event_descriptions=event_description_dict_llm, threshold=0.2, prompt_evidence=evidence, keyword_input=keyword_input, embedder_input=embedder_input, keyword_output=keyword_output, phrase_output=phrase_output)
    return events

def get_col_suffix(keyword_input, embedder_input):
    col_suffix = "no"
    if keyword_input and embedder_input:
        col_suffix = "all"
    elif keyword_input and not embedder_input:
        col_suffix = "dict"
    elif not keyword_input and embedder_input:
        col_suffix = "embedder"
    return col_suffix
    
for keyword_output, phrase_output in [i for i in product([False,True],[False,True])]:
    for ET in event_types:
        os.makedirs(f"../exports/llm/{ET}", exist_ok=True)
        try:
            file = glob(f"../exports/groundtruth/F-SET/Generated/{ET}*.pkl")[0]
        except:
            print(f"No file found for {ET}")
            continue
        file_name = os.path.basename(file).strip(".pkl")
        df = pd.read_pickle(file)
        df.Similarity = df.Similarity.astype(str)
        df_grouped = df.groupby(['Sentence_dictionary'])[["UID","Event_Name_dictionary","Keyword","Similarity"]].agg(lambda x: tuple(set(x)) if len(set(x))>1 else set(x).pop()).reset_index()
        df_grouped.Similarity = df_grouped.Similarity.apply(eval)     
        disagreement_df_temp = df_grouped.copy()
        print(ET,len(disagreement_df_temp), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file_name)
        print(f"KW:{keyword_output}, Phrase:{phrase_output}, Time Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
        evidence={'keywords':disagreement_df_temp.Keyword.tolist(), 'event_names':disagreement_df_temp.Event_Name_dictionary.tolist(), 'similarities':disagreement_df_temp.Similarity.tolist()}
        for keyword_input, embedder_input in [i for i in product([False,True],[False,True])]:
            col_suffix = get_col_suffix(keyword_input, embedder_input)
            disagreement_df_temp.loc[:,f"LLM_Events_{col_suffix}_evidence"] = extract_events_funct(disagreement_df_temp.Sentence_dictionary,
                                                                                                   extractor=EventExtractor(
                                                                                                       event_name_model_type="llama3",
                                                                                                       attribute_model_type="None"
                                                                                                       ),
                                                                                                   keyword_input=keyword_input,
                                                                                                   embedder_input=embedder_input,
                                                                                                   keyword_output=keyword_output,
                                                                                                   phrase_output=phrase_output,
                                                                                                   evidence=evidence)
            disagreement_df_temp.loc[:,f"Event_Name_LLM_Events_{col_suffix}_evidence"] = disagreement_df_temp[f"LLM_Events_{col_suffix}_evidence"].apply(lambda x: x['event'])
        disagreement_df_temp.to_pickle(f"../exports/llm/{ET}/{file_name}_kw_{keyword_output}_phrase_{phrase_output}.pkl")
        disagreement_df_temp.to_excel(f"../exports/llm/{ET}/{file_name}_kw_{keyword_output}_phrase_{phrase_output}.xlsx", index=False)
