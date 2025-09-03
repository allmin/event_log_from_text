import pandas as pd
import os, sys
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils.event_extractor import EventExtractor
from config import event_types, event_description_dict_llm

def extract_events_funct(sentences, extractor=None, evidence={'keywords':[],'event_names':[],'similarities':[]}, get_keyword=None, get_phrase=None):
    global event_description_dict_llm, event_types
    # event_description_list = [f"{k} : {v}" for (k,v) in event_description_dict.items()]
    events = extractor.extract_events(sentences=sentences, event_names=event_types, event_descriptions=event_description_dict_llm, threshold=0.2, prompt_evidence=evidence, get_keyword=get_keyword, get_phrase=get_phrase)
    return events



from itertools import product
from glob import glob
import pandas as pd

event_types
for get_keyword, get_phrase in [i for i in product([False,True],[False,True])]:
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
        print(f"KW:{get_keyword}, Phrase:{get_phrase}, Time Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
        disagreement_df_temp.loc[:,"LLM_Events_no_evidence"] = extract_events_funct(disagreement_df_temp.Sentence_dictionary, 
                                                                            get_keyword=get_keyword, get_phrase=get_phrase,
                                                                            extractor=EventExtractor(event_name_model_type="llama3", 
                                                                                                    attribute_model_type="None"))
        disagreement_df_temp.loc[:,"LLM_Events_dict_evidence"] = extract_events_funct(disagreement_df_temp.Sentence_dictionary, 
                                                                                get_keyword=get_keyword, get_phrase=get_phrase,
                                                                                extractor=EventExtractor(event_name_model_type="llama3", attribute_model_type="None"),
                                                                                evidence={'keywords':disagreement_df_temp.Keyword.tolist(), 
                                                                                        'event_names':disagreement_df_temp.Event_Name_dictionary.tolist(), 
                                                                                        'similarities':[]})
        disagreement_df_temp.loc[:,"LLM_Events_embedder_evidence"] = extract_events_funct(disagreement_df_temp.Sentence_dictionary, 
                                                                                    extractor=EventExtractor(event_name_model_type="llama3", attribute_model_type="None"),
                                                                                    evidence={'keywords':[], 
                                                                                            'event_names':[], 
                                                                                            'similarities':disagreement_df_temp.Similarity.tolist()})
        disagreement_df_temp.loc[:,"LLM_Events_all_evidence"] = extract_events_funct(disagreement_df_temp.Sentence_dictionary, 
                                                                            extractor=EventExtractor(event_name_model_type="llama3", attribute_model_type="None"),
                                                                            evidence={'keywords':disagreement_df_temp.Keyword.tolist(), 
                                                                                        'event_names':disagreement_df_temp.Event_Name_dictionary.tolist(), 
                                                                                        'similarities':disagreement_df_temp.Similarity.tolist()})
        disagreement_df_temp.loc[:,'Event_Name_LLM_Events_no_evidence'] = disagreement_df_temp['LLM_Events_no_evidence'].apply(lambda x: x['event'])
        disagreement_df_temp.loc[:,'Event_Name_LLM_Events_dict_evidence'] = disagreement_df_temp['LLM_Events_dict_evidence'].apply(lambda x: x['event'])
        disagreement_df_temp.loc[:,'Event_Name_LLM_Events_embedder_evidence'] = disagreement_df_temp['LLM_Events_embedder_evidence'].apply(lambda x: x['event'])
        disagreement_df_temp.loc[:,'Event_Name_LLM_Events_all_evidence'] = disagreement_df_temp['LLM_Events_all_evidence'].apply(lambda x: x['event'])
        disagreement_df_temp.to_pickle(f"../exports/llm/{ET}/{file_name}_kw_{get_keyword}_phrase_{get_phrase}.pkl")
        disagreement_df_temp.to_excel(f"../exports/llm/{ET}/{file_name}_kw_{get_keyword}_phrase_{get_phrase}.xlsx", index=False)
        