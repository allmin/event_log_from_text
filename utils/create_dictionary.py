import pandas as pd
import os
import nltk
annotated_dict = "resources/keyword_dict_annotated.xlsx"
labels_main = ['Sleep','Eating','Excretion','Pain','Family']
if os.path.exists(annotated_dict):
    df_dict = pd.read_excel(annotated_dict)
    df_dict = df_dict[df_dict.positive==1]
    print(df_dict)
else:
    print("repopulating the dictionary")
    from nltk.corpus import wordnet as wn
    model_file = "resources/keyword_dict.xlsx"
    keyword_dict = {}
    labels_simple = labels_main
    rows = []
    for label in labels_simple:
            keyword_dict[label]=[]
            syns = wn.synsets(label)
            print(syns)   
            for syn in syns:
                if syn.lemmas()[0].name() not in keyword_dict[label]:
                    keyword_dict[label].append(syn.lemmas()[0].name()) 
                    row = pd.DataFrame([{"class":label,"label":syn.lemmas()[0].name().lower(),"definition":syn.definition(),"example":syn.examples(),"positive":1}])
                    rows.append(row)
                print(f"name:{syn.name()}, lemma names:{syn.lemmas()[0].name()}, \ndef:{syn.definition()}, \neg:{syn.examples()}")
                hypernyms = syn.hypernyms()
                hyponyms = syn.hyponyms()
                for hyper in hypernyms:
                    if hyper.lemmas()[0].name() not in keyword_dict[label]:
                        keyword_dict[label].append(hyper.lemmas()[0].name()) 
                        row = pd.DataFrame([{"class":label,"label":hyper.lemmas()[0].name().lower(),"definition":hyper.definition(),"example":hyper.examples(),"positive":1}])
                        rows.append(row)
                # print(f"\tHypernym: {hyper.name()}, lemma names:{hyper.lemmas()[0].name()}, - {hyper.definition()}")
                for hypo in hyponyms:
                    if hypo.lemmas()[0].name() not in keyword_dict[label]:
                        keyword_dict[label].append(hypo.lemmas()[0].name()) 
                        row = pd.DataFrame([{"class":label,"label":hypo.lemmas()[0].name().lower(),"definition":hypo.definition(),"example":hypo.examples(),"positive":1}])
                        rows.append(row)
                # print(f"\tHyponyms: {hypo.name()}, lemma names:{hypo.lemmas()[0].name()}, - {hypo.definition()}")

    keyword_dict
    df_dict = pd.concat(rows,ignore_index=True)
    df_dict.to_excel(model_file,index=False)