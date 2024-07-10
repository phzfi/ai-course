# source https://spacy.io/usage/spacy-101
# https://realpython.com/natural-language-processing-spacy-python/
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

print("simppeli lause")
print("")
# Testataan simppelillä lauseella.
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        token.shape_, token.is_alpha, token.is_stop)



# Entäs hassuttelulla?
print("------------------")
print ("hassuttelua")
print("")
doc_funny = nlp("This better might be better at betting that that better better")
# Elikkä suomeksi "Uhkapelaaja on parempi uhkapelaamaan kuin parempi uhkapelaaja"
# Lemmojen pitäisi olla bettereissä bet, good, bet, good, bet
for token in doc_funny:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        token.shape_, token.is_alpha, token.is_stop)
# Vaan meneekö oikein?
# No ei tainnut mennä, voi ei.
# Eipä mennyt sanasijaintikaan...

# Palataan takaisin siihen alkuperäiseen Apple-lauseeseen...

print ("--------------------")
print("Mitkä ovat erisnimiä?")
print("")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


# Kokeillaan sitten toisenlaista tilannetta - ovatko sanat samankaltaisia?

nlp = spacy.load("en_core_web_lg")
catdog_tokens = nlp("dog mutt cat lion leopard banana afskfsd")
print ("--------------------")
print("Voidaan tulostaa, että miltä sanat \"näyttävät\" ja ovatko ne tunnettuja?")
print("")

for token in catdog_tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)


print ("--------------------")
print("Voidaan tulostaa, ovatko sanat lähellä toisiaan?")
print("")

nlp = spacy.load("en_core_web_lg")  # make sure to use larger package!
doc1 = nlp("I like salty fries and hamburgers.")
doc2 = nlp("Fast food tastes very good.")

# Similarity of two documents
print(doc1, "<->", doc2, doc1.similarity(doc2))
# Similarity of tokens and spans
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))


# Voidaan myös selvittää sanojen yleisyyttä.


import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")
complete_text = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech company. He is"
    " interested in learning Natural Language Processing."
    " There is a developer conference happening on 21 July"
    ' 2019 in London. It is titled "Applications of Natural'
    ' Language Processing". There is a helpline number'
    " available at +44-1234567891. Gus is helping organize it."
    " He keeps organizing local Python meetups and several"
    " internal talks at his workplace. Gus is also presenting"
    ' a talk. The talk will introduce the reader about "Use'
    ' cases of Natural Language Processing in Fintech".'
    " Apart from his work, he is very passionate about music."
    " Gus is learning to play the Piano. He has enrolled"
    " himself in the weekend batch of Great Piano Academy."
    " Great Piano Academy is situated in Mayfair or the City"
    " of London and has world-class piano instructors."
)
complete_doc = nlp(complete_text)

words = [
    token.text
    for token in complete_doc
    if not token.is_stop and not token.is_punct
]

print(Counter(words).most_common(5))
# Pitäisi tulla
#[('Gus', 4), ('London', 3), ('Natural', 3), ('Language', 3), ('Processing', 3)]

# Nyt lisätään Language jossain muodossa (languages), niin nähdääen että mitä tapahtuu.

complete_text_2 = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech company. He is"
    " interested in learning Natural Language Processing."
    " There is a developer conference happening on 21 July"
    ' 2019 in London. It is titled "Applications of Natural'
    ' Language Processing". There is a helpline number'
    " available at +44-1234567891. Gus is helping organize it."
    " He keeps organizing local Python meetups and several"
    " internal talks at his workplace. Gus is also presenting"
    ' a talk. The talk will introduce the reader about "Use'
    ' cases of Natural Language Processing in Fintech".'
    " Apart from his work, he is very passionate about music."
    " Gus is learning to play the Piano. He has enrolled"
    " himself in the weekend batch of Great Piano Academy."
    " Great Piano Academy is situated in Mayfair or the City"
    " of London and has world-class piano instructors. Also Language languages."
)
complete_doc_2 = nlp(complete_text_2)
# nollataan sanalista.
words=[]
words = [
    token.text
    for token in complete_doc_2
    if not token.is_stop and not token.is_punct
]

print(Counter(words).most_common(5))
# lisättiin language ja languages, joten languagen pitäsi kasvaa kahdella?
#[('Language', 5), ('Gus', 4), ('London', 3), ('Natural', 3), ('Processing', 3)]
#Hmmmm... miksi ei tullutkaan language 5

#Koeponnistetaan uudestaan, nyt lemmalla.
print("")
print("-----------")
print ("kokeillaan lemmoilla")
print("")
lemmat1 = [
    token.lemma_
    for token in complete_doc
    if not token.is_stop and not token.is_punct
]
print(Counter(lemmat1).most_common(5))

lemmat2 = [
    token.lemma_
    for token in complete_doc_2
    if not token.is_stop and not token.is_punct
]
print(Counter(lemmat2).most_common(5))

# Miksi ei toimi (vinkki iso alkukirjain...)

print("")
print("-----------")
print ("kokeillaan lemmoilla, nyt lowercasessa")
print("stripataan lemmat kanssa turhista merkeistä")
print("")
lemmat1 = [
    token.lemma_.strip().lower()
    for token in complete_doc
    if not token.is_stop and not token.is_punct
]
print(Counter(lemmat1).most_common(5))

lemmat2 = [
    token.lemma_.strip().lower()
    for token in complete_doc_2
    if not token.is_stop and not token.is_punct
]
print(Counter(lemmat2).most_common(5))
# ja näin saatiin viisi language-sanaa.


# Voidaan kaivaa myöskin nimiä etunimi+sukunimi-pareja esiin.


about_text = (
    "Jimbo Jones likes to fish."
    "Jumbo Jones is his brother."
    "Jumbo brings the bait."
    "On the other side of the city four friends"
    " are standing in an alleyway."
    "Their names are: Bank Bill, Bale Bribble, Beff Boomhauer"
    " and Bill Bauterive."
    "+35840123123 is Jimbo's phone number."
)
about_doc = nlp(about_text)

from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# Määritetään pattern matcheri, jolla voidaan hakea määritelmän mukaisia asioita.
# Tässä tapauksessa koko nimi.
def extract_full_name(nlp_doc):
    pattern = [{"POS": "PROPN"}, {"POS": "PROPN"}]
    matcher.add("FULL_NAME", [pattern])
    matches = matcher(nlp_doc)
    for _, start, end in matches:
        span = nlp_doc[start:end]
        yield span.text


for name in extract_full_name(about_doc) :
    print (name)

# Matchereilla voidaan hakea muitankin käyttäen ylläolevaa patterneja,
# esim. puh.numeroita, s-postiosoitteita jne.
# Onkos meillä jotain jännää tekstissä?

print("")
print("---------------")
print("Osaako se suomea? Myös, miten sanat liittyvät toisiinsa?")
print("lisää luettavaa aiheesta, jos kiinnostaa https://nlp.stanford.edu/software/dependencies_manual.pdf")
print("")


nlp = spacy.load("fi_core_news_sm")
piano_text = "Unelias Pekka opettelee hiljaa punaisen pianon soittamista."
piano_doc = nlp(piano_text)
for token in piano_doc:
    print(
        f"""
TOKEN: {token.text}
=====
{token.tag_ = }
{token.head.text = }
{token.dep_ = }"""
    )
#voidaan piirtää käppyrä, jos tahtoo
#displacy.serve(piano_doc, style="dep")



print("")
print("---------------")
print("Snips snaps sanoo sakset?")
print("")

# Entäs, saadaanko pilkottua tekstia jollakin tavalla?

nlp = spacy.load("en_core_web_sm")
one_line_about_text = (
    "Gus Proto is a Python developer"
    " currently working for a London-based Fintech company"
)
one_line_about_doc = nlp(one_line_about_text)

# Extract children of `developer`
print([token.text for token in one_line_about_doc[5].children])


# Extract previous neighboring node of `developer`
print (one_line_about_doc[5].nbor(-1))


# Extract next neighboring node of `developer`
print (one_line_about_doc[5].nbor())


# Extract all tokens on the left of `developer`
print([token.text for token in one_line_about_doc[5].lefts])


# Extract tokens on the right of `developer`
print([token.text for token in one_line_about_doc[5].rights])


# Print subtree of `developer`
print (list(one_line_about_doc[5].subtree))


# Joskus meitä kiinnostaa parsia tekstistä toisiinsa liittyviä osia

print("")
print("---------------")
print("Erilaisia osia lauseesta...")
print("")


conference_text = (
    "There is a developer conference happening on 21 July 2019 in London."
)
conference_doc = nlp(conference_text)

# Extract Noun Phrases
for chunk in conference_doc.noun_chunks:
    print (chunk)



# Haetaan sitten vielä tekstistä kaikki verbifraasit.
# Tarvitsee asentaa textacy
#pip install textacy 


print("")
print("---------------")
print("Erilaisia verbifraaseja...")
print("")


import textacy

about_talk_text = (
    "The talk will introduce reader about use"
    " cases of Natural Language Processing in"
    " Fintech, making use of"
    " interesting examples along the way."
)

patterns = [{"POS": "AUX"}, {"POS": "VERB"}]
about_talk_doc = textacy.make_spacy_doc(
    about_talk_text, lang="en_core_web_sm"
)
verb_phrases = textacy.extract.token_matches(
    about_talk_doc, patterns=patterns
)

# Print all verb phrases
for chunk in verb_phrases:
    print(chunk.text)



# Extract noun phrase to explain what nouns are involved
for chunk in about_talk_doc.noun_chunks:
    print (chunk)


# Voidaan piirtää käppyröitä tekstin sisällöstä. Ks. https://spacy.io/api/top-level#displacy_options
#displacy.serve(doc, style="dep")