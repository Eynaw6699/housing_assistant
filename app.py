import gradio as gr
import pandas as pd
import time
import re
import config
from rag import RAGPipeline
from utils import setup_logger

logger = setup_logger("app")

# Global RAG instance
rag_pipeline = None

def init_rag():
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Initializing RAG system for Gradio...")
        try:
            rag_pipeline = RAGPipeline()
            logger.info("RAG system initialized successfully.")
            return "RAG Ready"
        except Exception as e:
            return f"Init Failed: {e}"
    return "RAG Ready"

def highlight_text(text: str, query: str) -> str:
    """
    Highlights words in text that appear in query using HTML.
    """
    if not query:
        return text
    
    # Simple tokenization
    words = re.findall(r'\w+', query.lower())
    words = [w for w in words if len(w) > 3] # Ignore short stop words approximately
    
    if not words:
        return text

    # Compile regex pattern for all words
    pattern = re.compile(r'(' + '|'.join(map(re.escape, words)) + ')', re.IGNORECASE)
    
    # Replace with span
    highlighted = pattern.sub(r'<span style="background-color: #ffd700; color: black; font-weight: bold;">\1</span>', text)
    return highlighted

def format_retrieval_html(docs, query) -> str:
    """
    Formats retrieved documents as an HTML table with highlighting.
    """
    if not docs:
        return "<p>No documents retrieved.</p>"
    
    html = """
    <style>
        .doc-card { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        .doc-score { font-weight: bold; color: #2c3e50; }
        .doc-source { font-size: 0.8em; color: #7f8c8d; margin-bottom: 5px; }
        .doc-content { font-size: 0.9em; line-height: 1.4; }
    </style>
    <div style="height: 600px; overflow-y: scroll; padding-right: 10px;">
    """
    
    for d in docs:
        source = d.metadata.get('source', 'Unknown')
        title = d.metadata.get('title', 'Document')
        
        # Highlight content
        snippet = d.page_content
        snippet = highlight_text(snippet, query)
        
        html += f"""
        <div class="doc-card">
            <div class="doc-score">Title: {title}</div>
            <div class="doc-source">Source: {source}</div>
            <div class="doc-content">{snippet}</div>
        </div>
        """
    
    html += "</div>"
    return html

def process_query_stream(message, history):
    global rag_pipeline
    
    if not message.strip():
        yield "", history, "Make sure to type a question!"
        return

    # Check Init
    if rag_pipeline is None:
        init_rag()
        if rag_pipeline is None:
            yield "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Error: System not initialized"}], ""
            return

    # 1. Update User Message
    history = history + [{"role": "user", "content": message}]
    # Add placeholder for assistant
    history = history + [{"role": "assistant", "content": ""}] 
    
    yield "", history, "Searching..."

    try:
        accumulated_response = ""
        docs = []
        
        # Stream!
        for token, retrieved_docs in rag_pipeline.stream_query(message):
            if retrieved_docs and not docs:
                docs = retrieved_docs
                # Render docs immediately
                retrieval_html = format_retrieval_html(docs, message)
                yield "", history, retrieval_html
            
            if token:
                accumulated_response += token
                # Update last message
                history[-1]['content'] = accumulated_response
                yield "", history, format_retrieval_html(docs, message) # Keep updating HTML implies refresh, maybe optimizable but fine for local
                
    except Exception as e:
        logger.error(f"Stream Error: {e}")
        history[-1]['content'] = config.GENERIC_ERROR_MESSAGE
        yield "", history, ""

# --- App ---
logger.info("Starting Gradio App...")
init_rag()

with gr.Blocks(title="HDB Assistant") as demo:
    gr.Markdown("# Public Housing Assistant")
    
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### Retrieved Context")
            # Using HTML for rich text highlighting
            retrieval_display = gr.HTML(label="Context", value="<p>Context will appear here...</p>")

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=600, label="Chat Conversation")
            msg = gr.Textbox(placeholder="Ask about HDB eligibility...", label="Your Question")
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear")

    # Bindings
    msg.submit(
        process_query_stream, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, retrieval_display]
    )
    submit_btn.click(
        process_query_stream, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, retrieval_display]
    )
    
    def clear_all():
        return [], "<p>Context cleared.</p>"
        
    clear_btn.click(clear_all, inputs=None, outputs=[chatbot, retrieval_display])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
