# Abhishek Dutta, Copyright 2024, MIT License.

import streamlit as st; import os
import keras; from keras import ops
import keras_hub as kh; import kagglehub

st.title('Abhi Micro Med LM')
os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle_username']
os.environ['KAGGLE_KEY'] = st.secrets['kaggle_key']

# initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# load the model once and use it across all users and sessions
@st.cache_resource
def load_model():
    return kagglehub.model_download('abhionic/medcon/keras/3m')

template = 'Patient: {patient} Doctor: {doctor}'
path = load_model()
model = keras.saving.load_model(f'{path}/model.keras')
vocab = f'{path}/vocab.txt'; seq_len = 256
tokenizer = kh.tokenizers.WordPieceTokenizer(vocab, seq_len, lowercase=True)
sampler = kh.samplers.TopPSampler(p=0.1, k=5)
def next(prompt, cache, index):
    logits = model(prompt)[:, index-1, :]; hidden_states = None
    return logits, hidden_states, cache

# react to user input
if prompt := st.chat_input('please enter your symptoms'):
    # add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    # display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)

    text = template.format(patient=prompt, doctor='')
    tokens = tokenizer(text); tokens = ops.expand_dims(tokens, axis=0)
    ct = ops.count_nonzero(tokens)
    outokens = sampler(next=next, prompt=tokens, index=ct)
    padidx = ops.where(ops.equal(outokens, 0))
    if ops.size(padidx)>0: outokens = outokens[0, :padidx[1][0]]
    outext = tokenizer.detokenize(outokens)
  
    def stream_data():
        for word in outext.split(' '):
            yield word + ' '
            time.sleep(0.02)

    # display assistant response in chat message container
    with st.chat_message('assistant'):
        response = st.write_stream(stream_data)
    # add assistant response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response})
