import spacy
import re

class TextLib:
    def __init__(self, language="en_core_web_sm"):
        self.nlp = spacy.load(language)
        self.nlp.add_pipe("set_custom_boundaries", before="parser")
        
    def remove_error_strings(self,text):
        error_strings = ["\x13"]
        for e in error_strings:
            text=text.replace(e,"")
        return text
    
    def split_headers(self,text):
        pattern = r'(?<![\d\]]):\s+'
        header_split = re.split(pattern, text)
        if len(header_split) >= 2:
            header1 = header_split[0]
            remaining_text = text.lstrip(header1+":\n")
            header2, body = self.split_headers(remaining_text)
            header = [header1]  
            header.extend(header2)           
        else:
            if len(text.split(" "))<=3 and text.strip().endswith(":"):
                header = [text]
                body = ""
            else:
                header = []
                body = text
        return header, body
    
    def get_abbreviation_variants(self,abbreviation):
        """
        Returns all variants of an abbriviation. for example pt -> Pt, PT
        """
        variants = [abbreviation, abbreviation.upper(), abbreviation.capitalize()]
        return variants
        
    
    def replace_abbreviations(self, text, abbreviation_dict=None):
        for expansion,abbreviation_list in abbreviation_dict.items():
            for abbr_key in abbreviation_list:
                for abbr in self.get_abbreviation_variants(abbr_key):    
                    if abbr.endswith('.'):
                        abbr=abbr.replace(".","\\.")
                        pattern = rf'\b{abbr}'
                    if "/" in abbr:
                        pattern = rf"(?<=\s){abbr}(?=\s)"
                    else:
                        pattern = rf'\b{abbr}\b'                
                    text = re.sub(pattern, expansion, text)
        return text

    @staticmethod
    @spacy.Language.component('set_custom_boundaries')
    def set_custom_boundaries(doc):
        for token in doc[:-1]:
            if token.text.strip(' ') =='\n': #\n Title : BlaBla
                if (token.i + 2) < len(doc) and doc[token.i + 2].text.strip(' ') == ':':
                    doc[token.i + 1].is_sent_start = True
                    doc[token.i + 2].is_sent_start = False
                    if (token.i + 3) < len(doc) and doc[token.i + 3].text.strip(' ') == "\n": #\n Title : \n BlaBla
                        if (token.i + 4) < len(doc):
                            doc[token.i + 4].is_sent_start = False
                elif (token.i + 3) < len(doc) and doc[token.i + 3].text.strip(' ') == ':': #\n Title Title : BlaBla
                    doc[token.i + 1].is_sent_start = True
                    doc[token.i + 2].is_sent_start = False
                elif (token.i + 4) < len(doc) and doc[token.i + 4].text.strip(' ') == ':': #\n Title Title Title : BlaBla
                    doc[token.i + 1].is_sent_start = True
                    doc[token.i + 2].is_sent_start = False
            
            if token.text.strip(' ') =='\n\n' or token.text.strip(' ') =='\n\n\n':
                if (token.i + 1) < len(doc):
                    doc[token.i + 1].is_sent_start = True
            
            if token.text == ';':
                if (token.i + 1) < len(doc):
                    doc[token.i + 1].is_sent_start = True
        return doc

    def sentence_splitter(self, text, span=True):
        doc = self.nlp(text)
        sentences = []
        headers = []
        start_line = 1
        end_line = 0
        start_col = 0
        end_col = 0
        for sent in doc.sents:
            txt_frag = sent.text_with_ws
            sent_text = txt_frag.strip()
            if span:    
                lines = txt_frag.split("\n")
                num_lines = (len(lines))
                num_char_last = len(lines[-1])
                last_char = lines[-1]
                end_line = start_line + num_lines -1
                end_col = num_char_last if "\n" in txt_frag else (num_char_last + start_col)
                span = f"{start_line}.{start_col}_{end_line}.{end_col}"
                start_line = end_line
                start_col = end_col
                sentences.append({
                    'text': sent_text,
                    'span': span})
            else:
                headers,body = self.split_headers(sent_text)
                sentences.append({
                    'text': body,
                    'span': span,
                    'headers':headers})
        # sentences = [i for i in sentences if (self.is_sentence(i['text']) or i['headers'])]
        return sentences
    
    def is_sentence(self,text):
        if len(text) <=5:
            return False
        else:
            return True
    
    def get_token_count(self,text):
        doc = self.nlp(text)
        return len(doc)
    
    def display_token(self,text):
        for i in self.nlp(text):
            print(repr(i.text),i.is_sent_start)
    