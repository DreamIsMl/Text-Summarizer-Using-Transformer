import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import streamlit as st

# Load T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

def generate_summary(text):
    # Preprocess the input text
    preprocessed_text = text.strip().replace('\n', '')
    t5_input_text = 'summarize: ' + preprocessed_text
    
    # Tokenize the text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512).to(device)
    
    # Generate summary
    summarize_test = model.generate(tokenized_text, min_length=30, max_length=120)
    
    # Decode the summary
    summary = tokenizer.decode(summarize_test[0], skip_special_tokens=True)
    
    return summary

# Streamlit app
def main():
    st.title("Text Summarization App with T5 Model")
    
    # Input text area
    user_input = st.text_area("Enter text for summarization:", "")
    
    if st.button("Generate Summary"):
        if user_input:
            # Generate and display summary
            summary = generate_summary(user_input)
            st.subheader("Generated Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text for summarization.")

if __name__ == "__main__":
    main()
