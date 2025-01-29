import os
import logging
from dotenv import load_dotenv
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.auth import default
import json
import datetime
from vertexai.preview.generative_models import GenerationConfig
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def init_vertex_ai():
    try:
        # Check if GOOGLE_APPLICATION_CREDENTIALS is set
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        logger.info(f"Credentials path: {creds_path}")
        
        if not creds_path:
            logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
            return False
            
        # Initialize Vertex AI
        project_id = "future-footing-423417-b1"
        location = "us-central1"
        
        credentials, project = default()
        vertexai.init(
            project=project_id, 
            location=location,
            credentials=credentials
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {str(e)}")
        return False

def chunk_text(text, chunk_size=4000):
    """Split text into chunks of approximately chunk_size characters."""
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        if current_length + len(para) > chunk_size and current_chunk:
            # Join and add the current chunk
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = len(para)
        else:
            current_chunk.append(para)
            current_length += len(para)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def count_qa_pairs(text):
    """Count the number of Q&A pairs in the text"""
    # Count occurrences of "Q:" as a proxy for Q&A pairs
    return text.count("Q:")

def generate_qa(text_chunk, model, chunk_num, total_chunks):
    """Generate Q&A for a single chunk of text."""
    print(f"\n[Chunk {chunk_num}/{total_chunks}] Generating Q&A pairs...")
    print(f"Processing chunk of {len(text_chunk)} characters")
    
    prompt = f"""
    Your task is to create questions and answers from the provided document content. Important guidelines:

    1. Focus on the text content only - do not create Q&As about URLs themselves
    2. When relevant, include URLs in answers as "For more information, visit: [URL]"
    3. Break down content into clear, focused Q&As
    4. Create questions about the actual information, not about where to find it
    5. Include both high-level and detailed questions
    6. Keep answers factual and based on the content

    Content:
    {text_chunk}

    For each Q&A item, include:
    1. Question: Make it clear and specific about the content
    2. Answer: Provide a helpful response that addresses the question. If relevant, end with "For more information: [URL]"
    3. Source: Document section
    4. Tag: Categorize the Q&A (e.g., benefits, policies, working hours, leadership, culture, development, wellness, etc.)
    5. Title: Create an engaging, clickable title

    Example Q&A:
    {{
        "qa_items": [
            {{
                "question": "What are the key wellness benefits available?",
                "answer": "Employees receive comprehensive wellness benefits including mental health support and fitness programs. For more details: benefits.company.com/wellness",
                "source": "Benefits Overview",
                "tag": "benefits",
                "title": "Understanding Your Wellness Benefits"
            }}
        ]
    }}
    """
    
    # Define the response schema for Q&A
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "qa_items": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "question": {"type": "STRING"},
                        "answer": {"type": "STRING"},
                        "source": {"type": "STRING"},
                        "tag": {"type": "STRING"},
                        "title": {"type": "STRING"}
                    },
                    "required": ["question", "answer", "source", "tag", "title"]
                }
            }
        },
        "required": ["qa_items"]
    }
    
    print("Waiting for AI response...", end='', flush=True)
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            max_output_tokens=8192,
            temperature=0.7
        )
    )
    
    try:
        qa_data = json.loads(response.text)
        qa_count = len(qa_data['qa_items'])
        print(f"\r✓ Generated {qa_count} Q&A pairs for chunk {chunk_num}")
        return qa_data['qa_items']
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse response for chunk {chunk_num}")
        logger.error(f"Error details: {str(e)}")
        raise

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Vertex AI using the same method as crawler.py
    print("\nInitializing Vertex AI...")
    if not init_vertex_ai():  # Use the existing init_vertex_ai() function
        print("Failed to initialize Vertex AI")
        sys.exit(1)
    
    # Get the input file from command line argument
    import sys
    if len(sys.argv) != 2:
        print("Usage: python qa_generator.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    # Change output file naming pattern
    output_file = f"{os.path.splitext(input_file)[0]}_qa{os.path.splitext(input_file)[1]}"
    
    print("✓ Vertex AI initialized\n")
    
    # Load the content
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into chunks
    chunks = chunk_text(content)
    print(f"Content split into {len(chunks)} chunks")
    
    # Initialize the model
    model = GenerativeModel('gemini-1.5-pro')
    
    # Track metrics
    total_qa_pairs = 0
    successful_chunks = 0
    failed_chunks = 0
    
    # Process each chunk and collect results
    all_qa_pairs = []
    total_chunks = len(chunks)
    
    print(f"\nStarting Q&A generation for {total_chunks} chunks...")
    print("="*80)
    
    for i, chunk in enumerate(chunks, 1):
        try:
            qa_result = generate_qa(chunk, model, i, total_chunks)
            qa_count = count_qa_pairs(qa_result)
            total_qa_pairs += qa_count
            successful_chunks += 1
            all_qa_pairs.append(qa_result)
            
            # Show progress
            progress = (i / total_chunks) * 100
            print(f"Progress: {progress:.1f}% | Total Q&As: {total_qa_pairs}")
            print("-"*40)
            
        except Exception as e:
            print(f"\n❌ Error processing chunk {i}: {str(e)}")
            failed_chunks += 1
            continue
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Q&A Generation Results\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Statistics\n")
        f.write(f"- Total chunks processed: {total_chunks}\n")
        f.write(f"- Successful chunks: {successful_chunks}\n")
        f.write(f"- Failed chunks: {failed_chunks}\n")
        f.write(f"- Total Q&A pairs generated: {total_qa_pairs}\n\n")
        f.write("## Q&A Content\n\n")
        
        # Write each Q&A item in a structured format
        for qa_items in all_qa_pairs:  # all_qa_pairs now contains lists of qa items
            for item in qa_items:
                f.write(f"## {item['title']}\n\n")
                f.write(f"**Question:** {item['question']}\n\n")
                f.write(f"**Answer:** {item['answer']}\n\n")
                f.write(f"**Source:** {item['source']}\n")
                f.write(f"**Tag:** {item['tag']}\n\n")
                f.write("---\n\n")
    
    # Print final statistics
    print("\n" + "="*80)
    print("Generation Complete!")
    print(f"✓ Successfully processed: {successful_chunks}/{total_chunks} chunks")
    print(f"✗ Failed chunks: {failed_chunks}")
    print(f"📊 Total Q&A pairs generated: {total_qa_pairs}")
    print(f"💾 Results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main() 