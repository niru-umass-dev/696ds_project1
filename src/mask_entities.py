import en_core_web_trf
nlp = en_core_web_trf.load()
doc = nlp("The United Nations is headquartered in the Hotel Geneva in Switzerland, where five of its member nations - the United States, Russia, China, France, and the United Kingdom - have diplomatic missions. The Secretary General is Armando Gutierrez.")
print([(w.text, w.pos_) for w in doc])
matcher = "l'Hotel de Geneva"
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)