# Abhishek Dutta, Copyright 2024, MIT License.

import streamlit as st; import os; import time
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
    return kagglehub.model_download('abhionic/medcon/keras/14m')

template = '<user> {user} <model> '
path = load_model()
model = keras.saving.load_model(f'{path}/model.keras')
vocab = f'{path}/vocab.txt'; seq_len = 256
reserved_tokens = ['[PAD]', '[UNK]', '<user>', '<model>', '<preliminary diagnosis>',
                   '<indicators>', '<causes>', '<treatment>', '<prevention>']
tokenizer = kh.tokenizers.WordPieceTokenizer(vocab, seq_len, lowercase=True, strip_accents=True,
                        special_tokens=reserved_tokens, special_tokens_in_strings=True)
sampler = kh.samplers.TopPSampler(temperature=1, p=0.1, k=5)

generated_token_probs = []; last_logits = None
def next_with_prob(prompt, cache, index):
    global last_logits
    if last_logits is not None:
        selected_token_id = prompt[0, index - 1]
        probs = ops.softmax(last_logits, axis=-1)
        selected_prob = probs[0, selected_token_id]
        generated_token_probs.append(selected_prob)
    logits = model(prompt)[:, index-1, :]
    last_logits = logits; hidden_states = None
    return logits, hidden_states, cache

# react to user input
if prompt := st.chat_input('please enter your symptoms'):
    # add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    # display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)

    text = template.format(user=prompt); tokens = tokenizer(text) 
    tokens = ops.expand_dims(tokens, axis=0); ct = ops.count_nonzero(tokens)
    outokens = sampler(next=next_with_prob, prompt=tokens, index=ct)
    padidx = ops.where(ops.equal(outokens, 0)) # remove padding
    if ops.size(padidx)>0: outokens = outokens[0, :padidx[1][0]]
    try:
        start_marker_id = tokenizer.token_to_id('<preliminary diagnosis>')
        end_marker_id = tokenizer.token_to_id('<indicators>')
        start_idx = ops.where(outokens == start_marker_id)[0][0]
        end_idx = ops.where(outokens == end_marker_id)[0][0]
        target_probs = generated_token_probs[start_idx + 1 : end_idx]
        average_prob = ops.mean(target_probs) if target_probs else 0
    except: print('could not find markers in the output.')
    outext = tokenizer.detokenize(outokens)
    outext = outext + f' confidence={average_prob}'
  
    def stream_data():
        for word in outext.split(' '):
            yield word + ' '
            time.sleep(0.02)

    # display assistant response in chat message container
    with st.chat_message('assistant'):
        response = st.write_stream(stream_data)
    # add assistant response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response})
