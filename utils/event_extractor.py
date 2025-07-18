import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import subprocess
import ollama
import regex as re
import pandas as pd
import sys, os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

class EventExtractor:
    def __init__(self, is_event_model_type=None, event_name_model_type=None, attribute_model_type=None):
        self.event_name_model_type = event_name_model_type
        self.event_name_known = False
        self.is_event_model_type = is_event_model_type
        self.attribute_model_type = attribute_model_type
        self.error_logs = []
        if event_name_model_type == "biolord":
            self.model = SentenceTransformer('FremyCompany/BioLORD-2023')
        elif event_name_model_type == "dictionary":
            self.dictionary_df = pd.read_excel("../resources/keyword_dict_annotated.xlsx")
            self.dictionary = {i:j for i,j,p in zip(self.dictionary_df['label'], self.dictionary_df['class'], self.dictionary_df['positive']) if p==1}
            
            self.lemma_data = self.get_lemma_data()
            
            #update self.dictionary with all other forms of the key of the same lemma
            for forms in self.lemma_data['all_forms']:
                for form in forms:
                    if form in self.dictionary.keys():
                        event_name = self.dictionary[form]
                        self.dictionary.update({i:event_name for i in forms if i not in self.dictionary.keys()})
                        break
            #write self.dictionary to a file "resources/keyword_dict_annotated_with_medication_expanded.xlsx"
            self.dictionary_expanded_df = pd.DataFrame(list(self.dictionary.items()), columns=['label', 'class'])
            #soring the dataframe by class
            self.dictionary_expanded_df.sort_values(by=['class','label'],ascending = True, inplace=True)
            self.dictionary_expanded_df.to_excel("../resources/keyword_dict_annotated_expanded.xlsx", index=False)
            self.dictionary = {i:j for i,j in zip(self.dictionary_expanded_df['label'], self.dictionary_expanded_df['class'])}
                             
        self.is_event_cache = {}
        self.event_name_cache = {}
        self.attribute_cache = {}
        self.event_attribute_cache = {}
        
    def extract_events(self, sentences, event_names, threshold=0.2):

        self.event_list = []
        self.sentences = sentences
        self.similarities_dict = [{}]*len(self.sentences)
        self.keywords = [""]*len(self.sentences)
        self.predefined_event_names = event_names
        self.threshold = threshold
        self.extract_is_event()
        self.extract_event_names()
        self.extract_attributes()
        for sentence, event, similarity_dict, keyword, attribute_dict in zip(self.sentences, self.predicted_events, self.similarities_dict, self.keywords, self.attribute_dict_list ):
            if self.event_name_model_type == "biolord":
                self.event_list.append({"sentence":sentence, "event":event, "similarity":similarity_dict, "attributes": attribute_dict})
            elif self.event_name_model_type == "dictionary":
                self.event_list.append({"sentence":sentence, "event":event, "keyword":keyword, "attributes": attribute_dict})
            else:
                self.event_list.append({"sentence":sentence, "event":event, "attributes": attribute_dict})
        return self.event_list
    
    
    def extract_attributes_given_event(self, sentence, event_name, skip_others=True):
        attributes_dict = {}
        if skip_others and event_name == "Unknown":
            return attributes_dict
        if self.attribute_model_type != "llama3":
            return attributes_dict
        if sentence in self.event_name_cache:
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
        json_response = self.get_json_response(prompt)

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
            json_response = self.get_json_response(prompt)
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
            self.extract_event_names_dictionary()
            
    
    def extract_event_names_dictionary(self):
        self.predicted_events = []
        self.keywords = []
        for sentence in self.sentences:
            sentence_events = []
            sentence_keywords = []
            if sentence in self.event_name_cache:
                self.predicted_events.append(self.event_name_cache[sentence])
                continue
            for index,(keyword, event_name) in enumerate(self.dictionary.items()):
                if re.search(rf'\b{re.escape(keyword)}\b', sentence, re.IGNORECASE):
                    num_of_occurrence = len(re.findall(rf'\b{re.escape(keyword)}\b', sentence, re.IGNORECASE))
                    for i in range(num_of_occurrence):
                        sentence_events.append(event_name)
                        sentence_keywords.append(keyword)
            if len(sentence_events)==0 and index==len(self.dictionary)-1:
                self.predicted_events.append(["Unknown"])
                self.keywords.append([])
            else:
                self.predicted_events.append(sentence_events)
                self.keywords.append(sentence_keywords)
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
            # Compute cosine similarities manually
            self.similarities = np.dot(query_vecs, label_vecs.T)
            self.indices = np.argmax(self.similarities, axis=1).reshape(-1, 1)
            

            self.similarities_dict = [{k: round(v, 2) for k, v in zip(self.predefined_event_names, row)}
                        for row in self.similarities]
            self.predicted_events = [
                self.predefined_event_names[i[0]] if (i[0] != -1 and s[i[0]] > self.threshold) else "Unknown"
                for i,s in zip(self.indices, self.similarities)
            ]
            self.event_name_known = True



    
    def get_event_names(self, threshold):
        indices = [
            [int(idx[0]),sim[0]] if sim[0] > threshold else [-1,sim]
            for idx, sim in zip(self.indices, self.similarities)
        ]
        events = [[self.predefined_event_names[i],sim] if i != -1 else ["Unknown",sim] for i,sim in indices]
        return events

    def get_json_response(self, prompt):
        response = ollama.generate(model='llama3', prompt=prompt, options={"temperature": 0}, format='json')
        raw_output = response['response'].strip()
        # json_response = re.search(r'\{.*?\}', raw_output)
        json_response = re.search(r'\{(?:[^{}]|(?R))*\}', raw_output, re.DOTALL)
        # json_response = re.search(r'```(?:json)?\s*({.*?})\s*```', raw_output, re.DOTALL)
        if json_response: 
            try:      
                return json_response.group(0)
            except:
                return "{}"
        else:
            json_response="{}"
            return json_response
    
    
    def extract_is_event_llama(self):
        self.is_events = []
        for sentence in self.sentences:
            if sentence in self.is_event_cache:
                self.is_events.append(self.is_event_cache[sentence])
            else:
                prompt = f"""Does this sentence describe an activity related to any of {self.predefined_event_names}?. 
                Output ONLY a JSON: {{"is_activity":<boolean>}}
                Sentence: {sentence}"""
                json_response = self.get_json_response(prompt)
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
            
            
    def extract_event_names_llama(self):
        self.predicted_events = []
        
        for sentence in self.sentences:
            if sentence in self.event_name_cache:
                self.predicted_events.append(self.event_name_cache[sentence])
            else:                
                prompt = f"""Does this sentence describe an activity related to any of {self.predefined_event_names}?. 
                Output ONLY a JSON: {{"is_activity":<boolean>}}
                Sentence: {sentence}"""
                json_response = self.get_json_response(prompt)
                
                if json_response:
                    try:
                        event = json.loads(json_response)
                        event_name = event.get("event_name", "Unknown")
                    except json.JSONDecodeError:
                        event_name = "Unknown"
                
                self.predicted_events.append(event_name)
                self.event_name_known = True
                self.event_name_cache[sentence]=event_name
    
   
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
    # msentences = ["My right knee hurts", "The patient is sleeping well", "He had acetaminophen 500mg", "He met his family"]
    # ["ALL NIGHT.","NAPS COMFORTABLY", "SLEEPS COMFORTABLY" ]
    msentences = ["He slept well","He slept well","Trigonometry"]
    mevent_names = ["Pain", "Sleep", "Alert And Oriented"]
    # BIOLORDLLAMA = EventExtractor(is_event_model_type='llama3', attribute_model_type='llama3')
    # BIOLORDLLAMA = EventExtractor(event_name_model_type='dictionary', attribute_model_type='None')
    BIOLORDLLAMA = EventExtractor(event_name_model_type='dictionary', attribute_model_type='None')
    BIOLORDLLAMA.extract_events(sentences=msentences, event_names=mevent_names)
    # print(BIOLORDLLAMA.lemma_data)
    print(BIOLORDLLAMA.event_list)

