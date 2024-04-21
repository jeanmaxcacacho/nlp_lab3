import pickle
import pandas as pd
import torchtext

haiku1 = "[SOH] Tranquil waters flow, Whispering secrets of time, Embraced by the night."
haiku2 = "[SOH] Moonlight dances soft, Through branches of ancient oak, Embraced by the night."
haiku3 = "[SOH] Serene silence reigns, Stars shimmer in the night sky, Embraced by the night."
haiku4 = "[SOH] Shadows dance gently, Across fields of golden wheat, Embraced by the night."
haiku5 = "[SOH] Fireflies flicker bright, Illuminating the dark, Embraced by the night."

# , suffices as end of line and . suffices as end of haiku. this is the haiku structure anyway

haikus = [haiku1, haiku2, haiku3, haiku4, haiku5]

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

tokenized_haikus = [tokenizer(haiku) for haiku in haikus]

# flatten tokenized_haikus in place
i = 0
while i < len(tokenized_haikus):
    if isinstance(tokenized_haikus[i], list):
        tokenized_haikus[i:i+1] = tokenized_haikus[i]
    else:
        i += 1

vocabulary = torchtext.vocab.build_vocab_from_iterator([tokenized_haikus])

with open('vocab.pkl','wb') as f:
    pickle.dump(vocabulary, f)

indexed_tokens = [vocabulary[token] for token in tokenized_haikus]

# the last token is to be predicted
context_size = 4

df = pd.DataFrame(columns=['x'+str(i) for i in range(context_size)])

for i in range(len(indexed_tokens) - context_size + 1):
    data = indexed_tokens[i:i+context_size]
    df = pd.concat([df, pd.DataFrame([data], columns=df.columns)], ignore_index=True)

df.to_csv("dataset.csv")