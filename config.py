event_types = ["Pain", "Sleep", "Excretion", "Eating", "Family"]
event_description_dict_embedder = {"Eating":"To take food into the body by mouth",
                              "Excretion":"Waste matter discharged from the body s feces or urine",
                              "Family":"A visit, call or communication with a member of the family", #Interaction with a family member
                              "Pain":"The reporting of pain or an observation of pain signals by the doctor/nurse",
                              "Sleep":"The act of sleeping, possibly mentioning its quality or quantity"}
event_description_dict_llm = {
                            "Eating": "The patient takes food into their body by mouth.",
                            "Excretion": "The patient discharges waste matter from their body.",
                            "Family": "The patient has a visit, call, or communication with a family member.",
                            "Pain": "The patient reports or shows signs of pain.",
                            "Sleep": "The patient is sleeping, or the sleep’s quality or quantity is described."
                            }