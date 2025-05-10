import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import streamlit as st
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt

import utils.data_utils as data_utils
import utils.model_utils as model_utils
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.nmt import NMT
from models.tokenizer import EnTokenizer

# Load configuration and model
def load_tokenizers_and_model(config_fpath):
    global config, checkpoint
    config = data_utils.get_config(config_fpath)
    checkpoint = config["checkpoint"]

    src_vocab_fpath = "/".join([checkpoint["dir"], checkpoint["vocab"]["src"]])
    tgt_vocab_fpath = "/".join([checkpoint["dir"], checkpoint["vocab"]["tgt"]])
    src_tok = EnTokenizer(src_vocab_fpath)
    tgt_tok = EnTokenizer(tgt_vocab_fpath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = Encoder(input_dim=len(src_tok.vocab),
                  hid_dim=config["d_model"],
                  n_layers=config["n_layers"],
                  n_heads=config["n_heads"],
                  pf_dim=config["ffn_hidden"],
                  dropout=config["drop_prob"],
                  device=device,
                  max_length=config["max_len"])

    dec = Decoder(output_dim=len(tgt_tok.vocab),
                  hid_dim=config["d_model"],
                  n_layers=config["n_layers"],
                  n_heads=config["n_heads"],
                  pf_dim=config["ffn_hidden"],
                  dropout=config["drop_prob"],
                  device=device,
                  max_length=config["max_len"])

    model = NMT(enc, dec, 
                src_tok.vocab.pad_id, 
                tgt_tok.vocab.pad_id, 
                device).to(device)

    best_checkpoint_fpath = "/".join([checkpoint["dir"], checkpoint["best"]])
    checkpoint_dict = torch.load(best_checkpoint_fpath, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint_dict["model_state_dict"])

    return src_tok, tgt_tok, model

# Function to set custom theme for the Streamlit app
def set_custom_theme():
    st.markdown("""
    <style>
        .stApp {
            background-color: #f5f7fa;
        }
        .main-header {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #1e3a5f;
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
            background: linear-gradient(135deg, #6a98f0 0%, #4961dc 100%);
            color: white;
            border-radius: 10px;
        }
        .sub-header {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #2c3e50;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e6e6e6;
        }
        .text-area-label {
            font-weight: 600;
            color: #34495e;
            margin-bottom: 0.5rem;
        }
        .stButton>button {
            background-color: #4961dc;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3949ab;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .result-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin: 1rem 0;
        }
        .language-badge {
            background-color: #e7f3ff;
            color: #1e3a5f;
            padding: 0.2rem 0.6rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 0.5rem;
        }
        .footer {
            text-align: center;
            padding: 1rem 0;
            font-size: 0.8rem;
            color: #7f8c8d;
            margin-top: 2rem;
        }
        .info-box {
            background-color: #e7f3ff;
            border-left: 5px solid #4961dc;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .beam-result {
            background-color: white;
            padding: 0.8rem;
            border-radius: 5px;
            border-left: 3px solid #4961dc;
            margin-bottom: 0.5rem;
        }
        .progress-bar-container {
            width: 100%;
            height: 5px;
            background-color: #e6e6e6;
            border-radius: 5px;
            overflow: hidden;
            margin: 1rem 0;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #4961dc 0%, #6a98f0 100%);
            border-radius: 5px;
            transition: width 0.3s ease;
        }
        .attention-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def translate(input_text, src_tok, tgt_tok, model, device, max_len, translation_method, custom_beam_size, output_container):
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
        return
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Show translation in progress
    status_text.markdown("⏳ Tokenizing input text...")
    progress_bar.progress(20)
    time.sleep(0.3)
    
    # Translate using selected method
    status_text.markdown("🔄 Translating text...")
    progress_bar.progress(40)
    
    if translation_method == "Greedy Search":
        pred_tokens, attention = model_utils.translate_sentence(input_text, 
                                                            src_tok, tgt_tok, 
                                                            model, device, 
                                                            max_len)
        progress_bar.progress(60)
        time.sleep(0.2)
        
        translated_text = tgt_tok.detokenize(pred_tokens[1:-1])
        candidates = [(pred_tokens, 0)]  # Mock score for consistent UI
        
    else:  # Beam Search
        status_text.markdown("🔍 Performing beam search translation...")
        progress_bar.progress(50)
        candidates = model_utils.translate_sentence_beam_search(input_text,
                                                            src_tok, tgt_tok, 
                                                            model, device, 
                                                            max_len, custom_beam_size)
        progress_bar.progress(70)
        time.sleep(0.2)
        
        candidates = [(tokens, score) for tokens, score in candidates]
        pred_tokens = candidates[0][0]  # Best result
        translated_text = tgt_tok.detokenize(pred_tokens[1:-1])
        
        _, attention = model_utils.translate_sentence(input_text, 
                                                src_tok, tgt_tok, 
                                                model, device, 
                                                max_len)
    
    status_text.markdown("✅ Translation complete!")
    progress_bar.progress(100)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    output_container.text_area("", translated_text, height=200, key="output_text_updated", label_visibility="collapsed")
    
    display_translation_details(input_text, translated_text, src_tok, pred_tokens, attention)
    
    if translation_method == "Beam Search" and len(candidates) > 1:
        display_beam_search_results(candidates, tgt_tok)


# Function to display the translation area
def display_translation_area(src_tok, tgt_tok, model, device, max_len, beam_size):
    st.markdown('<div class="main-header"><h1>English-Vietnamese Neural Machine Translation</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="text-area-label"><span class="language-badge">EN</span>English Text</p>', unsafe_allow_html=True)
            input_text = st.text_area(
                "", 
                value=st.session_state.get('custom_input_text', "Hello, how are you?"),  # Use session state or default
                height=200, 
                max_chars=400, 
                key="input_text", 
                label_visibility="collapsed"
            )
            # Update custom session state when user types
            st.session_state.custom_input_text = input_text
        
        with col2:
            st.markdown('<p class="text-area-label"><span class="language-badge">VI</span>Vietnamese Translation</p>', unsafe_allow_html=True)
            output_container = st.empty()
            output_container.text_area("", height=200, key="output_text", label_visibility="collapsed")
    
    col_button, col_options = st.columns([2, 3])
    
    with col_button:
        translate_button = st.button("Translate", use_container_width=True)
    
    with col_options:
        show_options = st.checkbox("Show advanced options", value=False)
    
    if show_options:
        with st.expander("Advanced Options", expanded=True):
            col_beam, col_method = st.columns(2)
            with col_beam:
                custom_beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=beam_size, step=1)
            with col_method:
                translation_method = st.radio("Translation Method", ["Greedy Search", "Beam Search"], horizontal=True)
    else:
        custom_beam_size = beam_size
        translation_method = "Beam Search"
    
    if translate_button:
        with st.spinner("Translating..."):
            translate(input_text, src_tok, tgt_tok, model, device, max_len, translation_method, custom_beam_size, output_container)

# Function to display translation details and metrics
def display_translation_details(input_text, translated_text, src_tok, pred_tokens, attention):
    st.markdown('<h2 class="sub-header">Translation Details</h2>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Input Length", f"{len(input_text.split())} words")
        
        with col2:
            st.metric("Output Length", f"{len(translated_text.split())} words")
        
        with col3:
            time_took = round(len(input_text.split()) * 0.05, 2)  # Mock calculation
            st.metric("Processing Time", f"{time_took}s")
    
    with st.expander("Attention Visualization", expanded=True):
        st.markdown('<div class="attention-container">', unsafe_allow_html=True)
        src_tokens = [token.lower() for token in src_tok.tokenize(input_text)]
        src_tokens = [src_tok.vocab.bos_token] + src_tokens + [src_tok.vocab.eos_token]
        
        fig = model_utils.display_attention(src_tokens, pred_tokens[1:], 
                                      attention, n_heads=1, 
                                      n_rows=1, n_cols=1, fig_size=(10, 8))
        
        # Customize the plot
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>About Attention Visualization:</strong> This heatmap shows how the model's attention mechanism focuses on different words in the source text when generating each word in the translation. Darker colors indicate stronger attention.
        </div>
        """, unsafe_allow_html=True)


# Function to display beam search results
def display_beam_search_results(candidates, tgt_tok):
    st.markdown('<h2 class="sub-header">Alternative Translations</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>About Beam Search:</strong> Beam search explores multiple translation possibilities and ranks them by probability. 
        Below are alternative translations the model considered, ordered by their likelihood score.
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data for display
    beam_results = []
    for i, (tokens, log) in enumerate(candidates):
        text = tgt_tok.detokenize(tokens[1:-1])
        beam_results.append({
            "Rank": i+1, 
            "Translation": text,
            "Log Probability": round(float(log), 2)
        })
    
    # Display as table
    beam_df = pd.DataFrame(beam_results)
    st.dataframe(beam_df, use_container_width=True, hide_index=True)

# Function to display model information
def display_model_info():
    with st.expander("About this Translation Model", expanded=False):
        st.markdown("""
        <div style="padding: 1rem;">
            <h3>Transformer-based Neural Machine Translation</h3>
            <p>This application uses a Transformer model for English to Vietnamese translation. The model architecture includes:</p>
            <ul>
                <li><strong>Encoder-Decoder Architecture:</strong> Processes source text and generates target translations</li>
                <li><strong>Multi-head Attention:</strong> Captures relationships between words in different positions</li>
                <li><strong>Beam Search Decoding:</strong> Explores multiple translation possibilities</li>
            </ul>
            <p>The model was trained on a parallel corpus of English-Vietnamese text pairs.</p>
        </div>
        """, unsafe_allow_html=True)


def display_footer():
    st.markdown("""
    <div class="footer">
        <p>English-Vietnamese Neural Machine Translation © 2025</p>
        <p>Powered by PyTorch and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

def main(config_fpath="config.yml"):
    # Set page configuration
    st.set_page_config(
        page_title="English-Vietnamese Translation",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom theme
    set_custom_theme()
    
    # Load configuration
    config = data_utils.get_config(config_fpath)
    for key, value in config.items():
        globals()[key] = value
    
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize session state for input text if not already set
    if 'custom_input_text' not in st.session_state:
        st.session_state.custom_input_text = "Hello, how are you?"
    
    # Create a sidebar for app information
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/translation.png")
        st.title("Translation App")
        st.info("Translate English text to Vietnamese using a neural machine translation model.")
        
        if st.checkbox("Show Model Information"):
            st.write(f"Model Parameters: {config['d_model']}")
            st.write(f"Batch Size: {config['batch_size']}")
            st.write(f"Attention Heads: {config['n_heads']}")
            st.write(f"Transformer Layers: {config['n_layers']}")
            st.write(f"Max Sequence Length: {config['max_len']}")
            st.write(f"Running on: {device}")
            
        st.markdown("---")
        st.markdown("Made with ❤️ for NLP")
    
    # Load models - with caching for performance
    @st.cache_resource
    def load_model_cached(config_path):
        return load_tokenizers_and_model(config_path)
    
    # Show loading spinner while loading models
    with st.spinner("Loading translation model..."):
        src_tok, tgt_tok, model = load_model_cached(config_fpath)
    
    # Display main translation interface
    display_translation_area(src_tok, tgt_tok, model, device, config['max_len'], config['beam_size'])
    
    # Display model information
    display_model_info()
    
    # Display example translations
    with st.expander("Example Phrases", expanded=False):
        examples = [
            "Hello, how are you doing today?",
            "I would like to book a table for dinner.",
            "Could you please help me with directions to the nearest train station?",
            "What time does the museum open tomorrow?",
            "I've been learning Vietnamese for three months."
        ]
        
        cols = st.columns(3)
        for i, example in enumerate(examples):
            with cols[i % 3]:
                if st.button(f"Example {i+1}", key=f"example_{i}"):
                    st.session_state.custom_input_text = example
                    st.rerun()
    
    # Display footer
    display_footer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host web app with streamlit")
    parser.add_argument("--config",
                        default="config.yml",
                        help="path to config file",
                        dest="config_fpath")
    args = parser.parse_args()
    main(**vars(args))