import re
from typing import List
from plagiarism_ai.schemas.plagiarism import TextChunk

class TextChunker:
    """
    Chunks text into smaller pieces for embedding generation.
    Uses overlapping windows to maintain context across chunk boundaries.
    Simple sentence splitting using common Tamil and English punctuation.
    """
    @staticmethod
    def chunk_text(text: str, max_chunk_size: int = 200, overlap: int = 50) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        # Replace multiple spaces/newlines with a single space
        normalized_text = re.sub(r'[\r\n\s]+', ' ', text).strip()
        
        # Split by sentence terminators (periods, question marks, exclamation marks, etc.)
        # Includes English and Tamil typical punctuation
        # Using a regex that splits keeping the punctuation is complex, so we split by boundary
        sentences = re.split(r'(?<=[.?!।])\s+', normalized_text)
        
        chunks: List[TextChunk] = []
        current_chunk = ""
        start_index = 0

        for sentence in sentences:
            if not sentence:
                continue

            # If a single sentence is larger than max_chunk_size, split by words
            if len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(TextChunk(
                        id=f"chunk-{len(chunks)}",
                        text=current_chunk.strip(),
                        start_index=start_index,
                        end_index=start_index + len(current_chunk)
                    ))
                    start_index += len(current_chunk) + 1
                    current_chunk = ""
                
                words = sentence.split(' ')
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk + ' ' + word) > max_chunk_size:
                        chunks.append(TextChunk(
                            id=f"chunk-{len(chunks)}",
                            text=temp_chunk.strip(),
                            start_index=start_index,
                            end_index=start_index + len(temp_chunk)
                        ))
                        start_index += len(temp_chunk) + 1
                        
                        overlap_count = max(1, overlap // 10)
                        overlap_words = temp_chunk.split(' ')[-overlap_count:]
                        temp_chunk = ' '.join(overlap_words) + ' ' + word
                    else:
                        temp_chunk = f"{temp_chunk} {word}" if temp_chunk else word
                
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                if len(current_chunk + ' ' + sentence) > max_chunk_size:
                    chunks.append(TextChunk(
                        id=f"chunk-{len(chunks)}",
                        text=current_chunk.strip(),
                        start_index=start_index,
                        end_index=start_index + len(current_chunk)
                    ))
                    start_index += len(current_chunk) + 1
                    current_chunk = sentence
                else:
                    current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence

        if current_chunk:
            chunks.append(TextChunk(
                id=f"chunk-{len(chunks)}",
                text=current_chunk.strip(),
                start_index=start_index,
                end_index=start_index + len(current_chunk)
            ))

        return chunks
