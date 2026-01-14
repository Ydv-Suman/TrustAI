from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
from typing import List
from loader import load_document

def split_into_sentences(text: str) -> List[str]:
    # Pattern to match sentence endings (. ! ?) followed by space or end of string
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


def create_sentence_based_chunks(
    documents: List,
    target_chunk_size: int = 400,
    chunk_overlap: int = 40,
    min_chunk_size: int = 100
) -> List:
    """Create sentence-based chunks from documents with variable chunk sizes."""
    all_chunks = []
    
    for doc in documents:
        text = doc.page_content
        sentences = split_into_sentences(text)
        
        if not sentences:
            continue
        
        current_chunk = []
        current_length = 0
        chunk_metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed target size and we have content
            if current_length + sentence_length > target_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**chunk_metadata, 'chunk_index': len(all_chunks)}
                ))
                
                # Handle overlap: include last few sentences in next chunk
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                # Start new chunk with overlap sentences
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            # Add current sentence to chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Add remaining sentences as final chunk (if it meets minimum size)
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_chunk_size or not all_chunks:
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**chunk_metadata, 'chunk_index': len(all_chunks)}
                ))
    
    return all_chunks


file = "../data/large_llm_hallucination_document.pdf"
print(f"Loading document from: {file}")
cleaned_text = load_document(file=file)

# Convert cleaned text to Document object
document = Document(
    page_content=cleaned_text,
    metadata={"source": file}
)

# Create chunks
print("Creating sentence-based chunks...")
chunks = create_sentence_based_chunks(
    documents=[document]
)