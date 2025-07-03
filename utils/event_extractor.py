import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import subprocess
import ollama
import regex as re

class EventExtractor:
    def __init__(self, event_name_model_type, attribute_model_type):
        self.event_name_model_type = event_name_model_type
        self.attribute_model_type = attribute_model_type
        if event_name_model_type == "biolord":
            self.model = SentenceTransformer('FremyCompany/BioLORD-2023')
    
    def extract_attributes(self):
        self.attribute_dict_list = []
        for event_name, sentence in zip(self.predicted_events, self.sentences):
            self.attribute_dict_list.append(self.extract_attributes_given_event(sentence,event_name))
    
    def extract_attributes_given_event(self, sentence, event_name, skip_others=True):
        attributes_dict = {}
        if skip_others and event_name == "Others":
            return attributes_dict
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
        code_match = re.search(r'```(?:json)?\s*({.*?})\s*```', json_response, re.DOTALL)
        if code_match:
            try:
                json_str = code_match.group(1)
                attributes_dict = json.loads(json_str)
            except Exception as e:
                attributes_dict={"ERROR":e, "input":json_response}
        else:
            attributes_dict = {}
        return attributes_dict

    def extract_events(self, sentences, event_names, threshold=0.2):
        self.event_list = []
        self.sentences = sentences
        self.event_names = event_names
        self.threshold = threshold
        self.extract_event_names()
        self.extract_attributes()
        for sentence, event, attribute_dict in zip(self.sentences, self.predicted_events, self.attribute_dict_list):
            self.event_list.append({sentence:{"event":event, "attributes": attribute_dict}})
        return self.event_list

    def extract_event_names(self):
        if self.event_name_model_type == "biolord":
            self.extract_event_names_biolord()
        if self.event_name_model_type == "llama3":
            self.extract_event_names_llama()

    def extract_event_names_biolord(self):
        self.predicted_events = []
        self.sentence_embeddings = self.model.encode(self.sentences, normalize_embeddings=True)
        self.label_embeddings = self.model.encode(self.event_names, normalize_embeddings=True)

        # FAISS expects float32
        query_vecs = np.array(self.sentence_embeddings).astype('float32')
        label_vecs = np.array(self.label_embeddings).astype('float32')

        # Build FAISS index for cosine similarity (via inner product after normalization)
        dim = label_vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine if vectors are normalized
        self.index.add(label_vecs)

        # Perform search: top-1 similar label per sentence
        self.similarities, self.indices = self.index.search(query_vecs, 1)

        self.predicted_events = self.get_event_names(self.threshold)
    
    def get_event_names(self, threshold):
        # self.similarities is (n_sentences, 1)
        indices = [
            int(idx[0]) if sim[0] > threshold else -1
            for idx, sim in zip(self.indices, self.similarities)
        ]
        events = [self.event_names[i] if i != -1 else "Others" for i in indices]
        return events

    def get_json_response(self, prompt):
        response = ollama.generate(model='llama3', prompt=prompt)
        raw_output = response['response'].strip()
        return raw_output

    def extract_event_names_llama(self):
        self.predicted_events = []
        for sentence in self.sentences:
            prompt = F"""# CONTEXT
            You are trying to understand the sentences written by a nurse. 
            Your task involves looking for specific event names within the text. 
            Accurate detection of these event names is crucial for a clear understanding of the note. 
            # TASK DESCRIPTION
            1. Read the following sentence: "{sentence}"
            2. Select the appropriate event name from the event name list based on its content. 
            Event name list: {self.event_names}
            3. Evaluate the sentence carefully to determine the most appropriate event name that best matches the context and content of the sentence. 
            
            # RESTRICTIONS
            ## No matching event names
            - If none of {self.event_names} accurately describes the content of the sentence, classify it as "Other".
            # OUTPUT FORMAT
            Format the output as ONLY a valid JSON using the format {{"event_name": "<event_name>"}}, where <event_name> is the determined classification from the provided list or "Other" if no match is found.
            """
            json_response = self.get_json_response(prompt)
            json_response = re.search(r'\{.*?\}', json_response)
            if json_response:
                try:
                    event = json.loads(json_response.group())
                    event_name = event.get("event_name", "Other")
                except json.JSONDecodeError:
                    event_name = "Other"
            self.predicted_events.append(event_name)

if __name__ == "__main__":
    msentences = ["My right knee hurts", "The patient is sleeping well", "He had acetaminophen 500mg"]
    mevent_names = ["Pain Complaint", "Sleep"]
    # BIOLORD = EventExtractor("biolord")
    # BIOLORD.extract_event_names(sentences=msentences, event_names=mevent_names, threshold=0.2)
    # print("BioLORD:", BIOLORD.predicted_events)


    BIOLORDLLAMA = EventExtractor(event_name_model_type="biolord", attribute_model_type="llama3")
    BIOLORDLLAMA.extract_events(sentences=msentences, event_names=mevent_names)
    print(BIOLORDLLAMA.event_list)

