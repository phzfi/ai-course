import spacy
nlp = spacy.load("fi_core_news_sm")
import fi_core_news_sm
nlp = fi_core_news_sm.load()
doc = nlp("No text available yet")
print([(w.text, w.pos_) for w in doc])
