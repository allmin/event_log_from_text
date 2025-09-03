import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import subprocess
import ollama
import regex as re
import pandas as pd
import sys, os
from itertools import product
import time

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

class EventExtractor:
    def __init__(self, is_event_model_type=None, event_name_model_type=None, attribute_model_type=None, dictionary_file=None):
        self.event_name_model_type = event_name_model_type
        self.event_name_known = False
        self.event_detection_time = []
        self.is_event_model_type = is_event_model_type
        self.attribute_model_type = attribute_model_type
        self.error_logs = []
        #dictionary denotes keyword matching, biolord denotes embedding similarity
        if event_name_model_type == "biolord":
            self.model = SentenceTransformer('FremyCompany/BioLORD-2023')
        elif event_name_model_type == "dictionary":
            if not dictionary_file:
                dictionary_file = "../resources/keyword_dict_annotated.xlsx"

            self.dictionary_input_df = pd.read_excel(dictionary_file)
            self.lemma_data = self.get_lemma_data() #['lemma', 'all_forms']
            self.dictionary_positive_lemmas = {lemma_keyword:event_type for keyword,event_type,positive in zip(self.dictionary_input_df['label'], self.dictionary_input_df['class'], self.dictionary_input_df['positive']) if positive==1 for lemma_keyword in self.lemmatize_keyword(keyword)}
            #update self.dictionary_positive_lemmas with all other forms of the key of the same lemma
            self.dictionary_positive_lemmas_list = [{'form':form,'event_type':event_type, 'lemma':lemma} for (lemma,event_type) in self.dictionary_positive_lemmas.items() for form in self.get_all_forms(lemma)]
            #write self.dictionary to a file "resources/keyword_dict_annotated_with_medication_expanded.xlsx"
            self.dictionary_expanded_df = pd.DataFrame(self.dictionary_positive_lemmas_list)
            #soring the dataframe by class
            self.dictionary_expanded_df.sort_values(by=['event_type','form'],ascending = True, inplace=True)
            self.dictionary_expanded_df.to_excel(dictionary_file.rstrip(".xlsx")+"_expanded.xlsx", index=False)
            self.dictionary = {form:{'event_type':event_type, 'lemma':lemma} for form,event_type,lemma in zip(self.dictionary_expanded_df['form'], self.dictionary_expanded_df['event_type'], self.dictionary_expanded_df['lemma'])}
               
        self.is_event_cache = {}
        self.event_name_cache = {}
        self.keywords_cache = {}
        self.lemmas_cache = {}
        self.attribute_cache = {}
        self.event_attribute_cache = {}
        self.raw_output_cache = {}
        self.phrases_cache = {}
    
    
    def lemmatize_keyword(self,keyword):
        lemma_dict = {}
        for sub_keyword in keyword.split("_"):
            returned_lemmas = self.lemma_data[self.lemma_data['all_forms'].apply(lambda x: sub_keyword in x)]['lemma'].tolist()
            if len(returned_lemmas) == 0:
                returned_lemmas = [sub_keyword]
            lemma_dict[sub_keyword] = returned_lemmas
        combinations = ['_'.join(p) for p in product(*lemma_dict.values())]
        return combinations
    
    def flatten_list_of_lists(self,list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]
    
    def get_all_forms(self,lemma):
        all_form_dict = {}
        for sub_lemma in lemma.split("_"):
            returned_all_forms = self.lemma_data[self.lemma_data['lemma'] == sub_lemma]['all_forms'].tolist()
            returned_all_forms = self.flatten_list_of_lists(returned_all_forms)
            if len(returned_all_forms) == 0:
                 returned_all_forms = [sub_lemma]
            all_form_dict[sub_lemma] = returned_all_forms
        combinations = ['_'.join(p) for p in product(*all_form_dict.values())]
        return combinations
    
    def extract_events(self, sentences, event_names, event_descriptions=None, threshold=0.2, 
                       prompt_evidence={'keywords':[],'event_names':[],'similarities':[]},
                       get_keyword=False, get_phrase=False, embedder_method=1):
        self.event_list = []
        self.embedder_method=embedder_method
        self.sentences = sentences
        self.event_descriptions = event_descriptions
        self.similarities_dict = [{}]*len(self.sentences)
        self.keywords = [""]*len(self.sentences)
        self.phrases = [""]*len(self.sentences)
        self.raw_outputs = [""]*len(self.sentences)
        self.lemmas = [""]*len(self.sentences)
        self.event_name_prompt_list = [""]*len(self.sentences)
        self.predefined_event_names = event_names
        self.predefined_event_names_w_unknown = event_names + ["Unknown"]
        self.threshold = threshold
        self.get_keyword = get_keyword
        self.get_phrase = get_phrase
        self.prompt_evidence = prompt_evidence
        self.extract_is_event()
        self.extract_event_names()
        self.extract_attributes()
        if len(self.event_detection_time) == 0:
            self.event_detection_time = [""]*len(self.sentences)
        assert len(self.sentences) == len(self.predicted_events) == len(self.similarities_dict) == len(self.keywords) == len(self.phrases) == len(self.lemmas) == len(self.attribute_dict_list) == len(self.event_name_prompt_list) == len(self.raw_outputs) == len(self.event_detection_time), \
            f"{len(self.sentences)}, {len(self.predicted_events)}, {len(self.similarities_dict)}, {len(self.keywords)}, {len(self.phrases)}, {len(self.lemmas)}, {len(self.attribute_dict_list)}, {len(self.event_name_prompt_list)} ,{len(self.raw_outputs)} ,{len(self.event_detection_time)}"
        for sentence, event, similarity_dict, keyword, phrase, lemma, attribute_dict, prompt, raw_output, time in zip(self.sentences, self.predicted_events, self.similarities_dict, self.keywords, self.phrases, self.lemmas, self.attribute_dict_list, self.event_name_prompt_list , self.raw_outputs, self.event_detection_time):
            if self.event_name_model_type == "biolord":
                self.event_list.append({"sentence":sentence, "event":event, "similarity":similarity_dict, "attributes": attribute_dict, "event_detection_time":time})
            elif self.event_name_model_type == "dictionary":
                self.event_list.append({"sentence":sentence, "event":event, "keyword":keyword, "lemma":lemma, "attributes": attribute_dict, "event_detection_time":time})
            elif self.event_name_model_type == "llama3":
                self.event_list.append({"sentence":sentence, "event":event, "keyword":keyword, "phrase":phrase, "raw_output":raw_output,"attributes": attribute_dict, "event_name_prompt":prompt, "event_detection_time":time})
            else:
                self.event_list.append({"sentence":sentence, "event":event, "attributes": attribute_dict, "event_detection_time":time})
        return self.event_list
    
    
    def extract_attributes_given_event(self, sentence, event_name, skip_others=True):
        attributes_dict = {}
        if skip_others and event_name == "Unknown":
            return attributes_dict
        if self.attribute_model_type != "llama3":
            return attributes_dict
        if sentence in self.event_name_cache:
            print("skipping LLM since sentence found in cache")
            return self.event_name_cache[sentence]
        prompt = f"""You are an expert medical language model that extracts structured data from clinical notes.

            Given a sentence that describes a patient-related {event_name} event, your task is to:
            1. Detect the relevant attribute types associated with that event (e.g., location, quality, duration, time, medication, dosage, etc.).
            2. Extract the corresponding attribute values from the sentence.

            Only extract attributes that are explicitly mentioned. Do not infer missing information.

            ### Input Sentence:
            "{sentence}"

            ### Output Format:
            {{  "<attribute_type_1>": "<attribute_value_1>",
                "<attribute_type_2>": "<attribute_value_2>",
                ...}}

            If no attributes are found, return an empty "attributes" dictionary.

            Only output valid JSON. 

        """
        json_response, raw_output = self.get_json_response(prompt)

        try:
            attributes_dict = json.loads(json_response)
        except Exception as e:
            attributes_dict = {"ERROR":e, "input":json_response}
            self.error_logs.append(["is-event", sentence, attributes_dict ])
        self.event_name_cache[sentence] = attributes_dict
        return attributes_dict
    
    def extract_event_attributes_given_is_event(self, sentence, is_event):
        event_names = {}
        attributes_dict = {"event_name":"None", "attributes":{}}
        if is_event == "false":
            return attributes_dict
        if self.attribute_model_type != "llama3":
            return attributes_dict
        if sentence in self.event_attribute_cache:
            return self.event_attribute_cache[sentence]
        prompt = f"""You are an expert medical language model that extracts structured data from clinical notes.
            Given a sentence that describes a patient-related event among {self.predefined_event_names}, your task is to:
            1. Detect the main event in the sentence among {self.predefined_event_names}. If nothing applies, set it as "Unknown".
            2. Detect the relevant attribute types associated with that main event (e.g., location, quality, duration, time, medication, dosage, etc.).
            3. Extract the corresponding attribute values from the sentence.
            Only extract attributes that are explicitly mentioned. Do not infer missing information.
            ### Input Sentence:
            "{sentence}"

            ### Output Format:
            {{  "event_name": "<name_of_main_event>",
                "attributes":{{          
                "<attribute_type_1>": "<attribute_value_1>",
                "<attribute_type_2>": "<attribute_value_2>",
                ...}}}}

            If no attributes are found, return an empty "attributes" dictionary.
            ONLY output valid JSON. 

        """
        try:
            json_response,raw_output = self.get_json_response(prompt)
            attributes_dict = json.loads(json_response)
            if attributes_dict == {}:
                attributes_dict = {"event_name":"None", "attributes":{}}
        except Exception as e:
            self.error_logs.append(["event-attribute",sentence, json_response])
            attributes_dict={"event_name":"Error", "attributes":json_response}
        self.event_attribute_cache[sentence] = attributes_dict
        return attributes_dict

    
    
    def extract_is_event(self):
        if self.is_event_model_type == "llama3":
            self.extract_is_event_llama()

    def extract_event_names(self):
        if self.event_name_model_type == "biolord":
            self.extract_event_names_biolord()
        if self.event_name_model_type == "llama3":
            self.extract_event_names_llama()
        if self.event_name_model_type == "dictionary":
            self.dictionary = {form:values for (form,values) in self.dictionary.items() if values["event_type"] in self.predefined_event_names}
            self.extract_event_names_dictionary()
            
    
    def extract_event_names_dictionary(self):
        self.predicted_events = []
        self.event_detection_time = []
        self.keywords = []
        self.lemmas = []
        for ind,sentence in enumerate(self.sentences):
            start_time = time.perf_counter()
            sentence_events = []
            sentence_keywords = []
            sentence_lemmas = []
            if sentence in self.event_name_cache:
                self.predicted_events.append(self.event_name_cache[sentence])
                self.keywords.append(self.keywords_cache[sentence])
                self.lemmas.append(self.lemmas_cache[sentence])
            else:
                for index,(keyword, event_name_lemma_dict) in enumerate(self.dictionary.items()):
                    keyword_w_space = keyword.replace('_', ' ')
                    if re.search(rf'\b{re.escape(keyword_w_space)}\b', sentence, re.IGNORECASE):
                        num_of_occurrence = len(re.findall(rf'\b{re.escape(keyword_w_space)}\b', sentence, re.IGNORECASE))
                        event_name = event_name_lemma_dict['event_type']
                        lemma = event_name_lemma_dict['lemma']
                        for i in range(num_of_occurrence):
                            sentence_events.append(event_name)
                            sentence_keywords.append(keyword_w_space)
                            sentence_lemmas.append(lemma)
                if len(sentence_events)==0 and index==len(self.dictionary)-1:
                    sentence_events = ["Unknown"]
                    sentence_keywords = [""]
                    sentence_lemmas = [""]
                self.predicted_events.append(sentence_events)
                self.keywords.append(sentence_keywords)
                self.lemmas.append(sentence_lemmas)
                self.event_name_cache[sentence] = sentence_events
                self.keywords_cache[sentence] = sentence_keywords
                self.lemmas_cache[sentence] = sentence_lemmas
            end_time = time.perf_counter()  # End timing
            elapsed_time = end_time - start_time
            self.event_detection_time.append(elapsed_time)
        self.event_name_known = True
            
    def extract_attributes(self):
        self.attribute_dict_list = []
        if self.event_name_known:
            for event_name, sentence in zip(self.predicted_events, self.sentences):
                self.attribute_dict_list.append(self.extract_attributes_given_event(sentence,event_name))

        else:
            self.predicted_events = []
            for is_event, sentence in zip(self.is_events, self.sentences):
                self.result = self.extract_event_attributes_given_is_event(sentence,is_event)
                if "attributes" not in self.result:
                    self.result['attributes'] = {}
                if "event_name" not in self.result:
                    self.result['event_name'] = "None"
                self.attribute_dict_list.append(self.result['attributes'])
                self.predicted_events.append(self.result['event_name'])

                
            

    def extract_event_names_biolord(self,use_faiss=False):
        self.predicted_events = []
        self.similarities = []
        start_time = time.perf_counter()
        
        self.sentence_embeddings = self.model.encode(self.sentences, normalize_embeddings=True)
        self.label_embeddings = self.model.encode(self.predefined_event_names, normalize_embeddings=True)

        # FAISS expects float32
        query_vecs = np.array(self.sentence_embeddings).astype('float32')
        label_vecs = np.array(self.label_embeddings).astype('float32')
        
        if use_faiss:
            # Build FAISS index for cosine similarity (via inner product after normalization)
            dim = label_vecs.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine if vectors are normalized
            self.index.add(label_vecs)

            # Perform search: top-1 similar label per sentence
            self.similarities, self.indices = self.index.search(query_vecs, 1)
            self.predicted_events = self.get_event_names(self.threshold)
            self.event_name_known = True
        else:
            if self.embedder_method == 1:
                self.similarities = np.dot(query_vecs, label_vecs.T)
                self.indices = np.argmax(self.similarities, axis=1).reshape(-1, 1)
                self.similarities_dict = [{k.split(' : ')[0]: round(v, 2) for k, v in zip(self.predefined_event_names, row)}
                            for row in self.similarities]
                self.predicted_events = [
                    self.predefined_event_names[i[0]] if (i[0] != -1 and s[i[0]] > self.threshold) else "Unknown"
                    for i,s in zip(self.indices, self.similarities)
                ]
                self.event_name_known = True
            elif self.embedder_method != 1:
                self.label_embeddings = self.model.encode(self.predefined_event_names, normalize_embeddings=True)
                anti_label_vecs = np.array(self.anti_label_embeddings).astype('float32')
                self.anti_similarities = np.dot(query_vecs, anti_label_vecs.T)
                self.similarities = np.dot(query_vecs, label_vecs.T)
                
                
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_per_sentence = elapsed_time/len(self.sentences)
        self.predicted_events = [i.split(' : ')[0] for i in self.predicted_events]
        self.event_detection_time = [time_per_sentence]*len(self.sentences)



    
    def get_event_names(self, threshold):
        indices = [
            [int(idx[0]),sim[0]] if sim[0] > threshold else [-1,sim]
            for idx, sim in zip(self.indices, self.similarities)
        ]
        events = [[self.predefined_event_names[i],sim] if i != -1 else ["Unknown",sim] for i,sim in indices]
        return events

    def get_json_response(self, prompt):
        response = ollama.generate(model='llama3.1:70b', prompt=prompt, options={"temperature": 0}, format='json')
        raw_output = response['response'].strip()
        # json_response = re.search(r'\{.*?\}', raw_output)
        json_response = re.search(r'\{(?:[^{}]|(?R))*\}', raw_output, re.DOTALL)
        # json_response = re.search(r'```(?:json)?\s*({.*?})\s*```', raw_output, re.DOTALL)
        if json_response: 
            try:      
                return json_response.group(0), raw_output
            except:
                return "{}",raw_output
        else:
            json_response="{}"
            return json_response,raw_output
    
    
    def extract_is_event_llama(self):
        self.is_events = []
        for sentence in self.sentences:
            if sentence in self.is_event_cache:
                self.is_events.append(self.is_event_cache[sentence])
            else:
                prompt = f"""Does this sentence describe an activity related to any of {self.predefined_event_names}?. 
                Output ONLY a JSON: {{"is_activity":<boolean>}}
                Sentence: {sentence}"""
                json_response, raw_output = self.get_json_response(prompt)
                try:
                    json_dict = json.loads(json_response)
                    is_event = json_dict.get("is_activity", False)
                    if (type(is_event) == str):
                        if is_event.lower() in ['true']:
                            is_event = True
                        else:
                            is_event = False
                except json.JSONDecodeError:
                    is_event = False
                self.is_event_cache[sentence]=is_event
                self.is_events.append(is_event)
    
    def get_evidence(self,ind=None):
        if ind < len(self.prompt_evidence["keywords"]):
            keyword_evidence = self.prompt_evidence["keywords"][ind]
            event_evidence= self.prompt_evidence["event_names"][ind]
        additional_facts_clause = "\n You may consider the additional facts if they are reasonable. Ignore otherwise"
        if len(self.prompt_evidence["keywords"]) == 0 and len(self.prompt_evidence["similarities"]) == 0:
            evidence = ""
        elif len(self.prompt_evidence["keywords"])!=0 and len(self.prompt_evidence["similarities"]) == 0:
            evidence = f"""Additional facts:
            A keyword matching algorithm without context, detected keyword(s): {keyword_evidence}
            and assigned event type: {event_evidence}.""" + additional_facts_clause
        elif len(self.prompt_evidence["keywords"])==0 and len(self.prompt_evidence["similarities"])!= 0:
            similarities = self.prompt_evidence["similarities"][ind]
            similarities = {k:f"{v:0.2f}" for (k,v) in similarities.items()}
            evidence = f"""Additional facts:
            A sentence embedder, assigned following similarity score to  
            each of the event type labels: {similarities}.""" + additional_facts_clause
        elif len(self.prompt_evidence["keywords"])!=0 and len(self.prompt_evidence["similarities"])!= 0:
            similarities = self.prompt_evidence["similarities"][ind]
            similarities = {k:f"{v:0.2f}" for (k,v) in similarities.items()}
            evidence = f"""Additional facts:
            A keyword matching algorithm without context, detected keyword(s): {self.prompt_evidence["keywords"][ind]}
            and assigned event type: {event_evidence}
            A sentence embedder, assigned following similarity score to  
            each of the event type labels: {similarities}.""" + additional_facts_clause       
        return evidence                            
            
    def extract_event_names_llama(self):
        self.event_detection_time = []
        self.predicted_events = []
        self.event_name_prompt_list = []
        self.raw_outputs = []
        self.keywords = []
        self.phrases = []
        for ind, sentence in enumerate(self.sentences):
            start_time = time.perf_counter()
            evidence = self.get_evidence(ind)   
            
            if not self.get_keyword and not self.get_phrase:
                get_keyword_clause = ""
                output_format = """{{"event type":<chosen event type>}}"""
            elif self.get_keyword and not self.get_phrase:
                get_keyword_clause = " along with keywords that support each event type"
                output_format = """{{"event type":<chosen event type>,"keywords":{{<chosen event type>:<keyword>}}}}"""
            elif not self.get_keyword and self.get_phrase:
                get_keyword_clause = " along with quotes from the text that support each event type"
                output_format = """{{"event type":<chosen event type>,"phrases":{{<chosen event type>:<quote>}}}}"""
            else:
                get_keyword_clause = " along with quotes and keywords from the text that support each event type"
                output_format = """{{"event type":<chosen event type>, "keywords":{{<chosen event type>:<keyword>}}, "phrases":{{<chosen event type>:<quote>}}}}"""

            if self.event_descriptions:
                event_w_description = self.event_descriptions
                event_w_description["Unknown"] = """choose "Unknown" if none of the other event type are applicable."""
                unknown_clause = ""
            else:
                event_w_description = self.predefined_event_names_w_unknown
                unknown_clause = """choose "Unknown" if none of the other event type are applicable."""
            prompt = f"""Given the sentence: {sentence}. 
            and the following event types: {event_w_description} 
            choose the most relevant event type(s) for the sentence{get_keyword_clause}.
            {unknown_clause}
            {evidence}
            Output ONLY a JSON: {output_format}"""
            if sentence in self.event_name_cache:
                self.predicted_events.append(self.event_name_cache[sentence])
                self.keywords.append(self.keywords_cache[sentence])
                self.phrases.append(self.phrases_cache[sentence])
                self.raw_outputs.append(self.raw_output_cache[sentence])
            else:                
                json_response, raw_output = self.get_json_response(prompt)
                if json_response:
                    try:
                        event = json.loads(json_response)
                        event_name = event.get("event type", "Unknown")
                        keywords = event.get("keywords", "")
                        phrases = event.get("phrases","")
                    except json.JSONDecodeError:
                        event_name = "Unknown"
                self.predicted_events.append(event_name)
                self.keywords.append(keywords)
                self.phrases.append(phrases)
                self.raw_outputs.append(raw_output)
                self.event_name_cache[sentence]=event_name
                self.keywords_cache[sentence]=keywords
                self.raw_output_cache[sentence]=raw_output
                self.phrases_cache[sentence]=raw_output
            self.event_name_prompt_list.append(prompt)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            self.event_detection_time.append(elapsed_time)
        self.event_name_known = True
    
   
    def get_lemma_data(self):
        with open("../resources/lemma.en.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
        lemma_data = []
        for line in lines:
            lemma_part, forms_part = line.strip().split(" -> ")
            lemma = lemma_part.split("/")[0]  # Remove frequency
            all_forms = forms_part.split(",") + [lemma]
            lemma_data.append({"lemma": lemma, "all_forms": all_forms})
        lemma_data = pd.DataFrame(lemma_data)
        return lemma_data
    
if __name__ == "__main__":
    msentences = ["He slept well","He slept well","He began to break bread","Trigonometry"]
    msentences = ["at midnight after being asleep in bed for several hours patient awoke coughing unable to raise unless sitting up in chair",
                  "labile bp, apnea ventilation when asleep even when off sedation, vent mode changed to mmv by rt",
                  "patient restless and agitated @ times, pulling cpap off, more cooperative when medicated, short periods of apnea noted when off cpap, will attempt to keep patient on cpap when asleep",
                  "asleep ft on aggressive pulm toilet done cpt with effect bs controlled with insulin drops and prn boluses [**md number(3) **] output great lasix tid cont to monitor cr and renal fx"]
#     msentences = ['bp lower when asleep',
#  'sleeping in naps',
#  'slept well',
#  'sleeping in long naps',
#  'short naps',
#  'slept most of night',
#  'slept in naps',
#  'sleeping',
#  'slept most of noc',
#  'slept most of night']
    mevent_names = ["Eating","Excretion","Family","Pain", "Sleep"]
    event_names_w_descriptions = ["Eating : To take food into the body by mouth",
                                  "Exretion : The process by which waste matter is discharged from the body",
                                  "Pain : The sensation of discomfort, distress, or agony, resulting from the stimulation of specialized nerve endings.",
                                  "Sleep : A natural and periodic state of rest during which consciousness of the world is suspended."]

    DICT = EventExtractor(event_name_model_type='dictionary', attribute_model_type='None')
    DICT.extract_events(sentences=msentences, event_names=mevent_names)
    print("Dictionary_events:",DICT.event_list)
    BIOLORD = EventExtractor(event_name_model_type='biolord', attribute_model_type='None')
    BIOLORD.extract_events(sentences=msentences, event_names=mevent_names)
    print("BIOLORD_events:",[i['Sleep'] for i in BIOLORD.similarities_dict])

    BIOLORD = EventExtractor(event_name_model_type='biolord', attribute_model_type='None')
    BIOLORD.extract_events(sentences=msentences, event_names=event_names_w_descriptions)
    print("BIOLORD_events:",[i['Sleep'] for i in BIOLORD.similarities_dict])


    
    # LLAMA1 = EventExtractor(event_name_model_type='llama3', attribute_model_type='None')
    
    # LLAMA1.extract_events(sentences=msentences, event_names=mevent_names)
    # print("LLAMA_no_evidence_events:",LLAMA1.event_list)
 
    LLAMA2 = EventExtractor(event_name_model_type='llama3', attribute_model_type='None')   
    LLAMA2.extract_events(sentences=msentences, event_names=mevent_names, 
                                event_descriptions={"Eating":"To take food into the body by mouth",
                                                    "Exretion":"The process by which waste matter is discharged from the body",
                                                    "Family":"A domestic group, or a number of domestic groups linked through descent (demonstrated or stipulated) from a common ancestor, marriage, or adoption.",
                                                    "Pain":"The sensation of discomfort, distress, or agony, resulting from the stimulation of specialized nerve endings.",
                                                    "Sleep":"A natural and periodic state of rest during which consciousness of the world is suspended."},
                                prompt_evidence={'keywords':DICT.keywords, 
                                                 'event_names':DICT.predicted_events, 
                                                 'similarities':BIOLORD.similarities_dict},
                                get_phrase=True, get_keyword=True)
    print("LLAMA_all_evidence_events:",LLAMA2.event_list)
    # sudo kill -9 $(nvidia-smi | awk 'NR>8 {print $5}' | grep -E '^[0-9]+$')
    


