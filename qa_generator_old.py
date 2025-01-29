# This is the mac mini version of the qa_generator.py file ff
import os
import logging
from dotenv import load_dotenv
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.auth import default
import json
import datetime
from vertexai.preview.generative_models import GenerationConfig

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

def generate_qa(filename):
    """Generate Q&A from a markdown file"""
    try:
        print(f"\nProcessing file: {filename}")
        print("="*80)
        
        # Read the markdown file
        print("Reading file contents...")
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Read {len(content)} characters")
        
        # Initialize Vertex AI
        print("\nInitializing Vertex AI...")
        if not init_vertex_ai():
            raise Exception("Failed to initialize Vertex AI")
        print("✓ Vertex AI initialized")
        
        # Create the prompt
        print("\nPreparing prompt...")
        prompt = f"""
        Your task is to create an extensive set of questions and answers from the provided document content. Be thorough and detailed:

        1. Break down EVERY section, paragraph, bullet point, and detail into separate Q&As
        2. For lists or bullet points, create individual questions for each item
        3. For complex topics, create multiple questions approaching the topic from different angles
        4. Look for implied information that employees might want to know
        5. Include both high-level and detailed questions
        6. Don't skip any information - if it's in the document, create a Q&A for it

        Document Name: {filename}
        Content:
        {content}

        For each Q&A item, include:
        1. Question: Make it clear and specific
        2. Answer: Provide a detailed, helpful response that fully addresses the question. Write as if explaining to an employee.
        3. Source: {filename}
        4. Tag: Categorize the Q&A (e.g., benefits, policies, working hours, leadership, culture, development, wellness, etc.)
        5. Title: Create an engaging, clickable title that makes employees want to learn more. Start with phrases like:
           - "Find out how..."
           - "Discover why..."
           - "Learn about..."
           - "Everything you need to know about..."
           - "The complete guide to..."
           - "Important updates on..."

        IMPORTANT: Be exhaustive in your analysis. Every piece of information should be converted into at least one Q&A. Aim for maximum coverage of the content.

        Format the output as JSON with this structure:
        {{
            "qa_items": [
                {{
                    "question": "What specific question is being answered?",
                    "answer": "Detailed, helpful answer that fully explains the topic...",
                    "source": "filename.md",
                    "tag": "relevant category",
                    "title": "Engaging, clickable title..."
                }}
            ]
        }}
        
        Wrap the JSON in ```json``` code blocks.
        """
        print("✓ Prompt prepared")
        
        # Generate Q&A with better error handling
        try:
            print("\nGenerating Q&A content...")
            print("This may take a few minutes. Please wait...")
            
            # Use gemini-1.5-flash for faster processing
            model = GenerativeModel('gemini-1.5-flash')
            
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
            
            # Reduce chunk size and add overlap
            max_chunk_size = 15000  # Reduced from 25000
            overlap = 1000  # Add overlap to maintain context
            
            # Better chunking with overlap
            content_chunks = []
            start = 0
            while start < len(content):
                end = min(start + max_chunk_size, len(content))
                # Find the last period to avoid cutting mid-sentence
                if end < len(content):
                    last_period = content[start:end].rfind('.')
                    if last_period != -1:
                        end = start + last_period + 1
                chunk = content[start:end]
                content_chunks.append(chunk)
                start = end - overlap
            
            print(f"\nSplit content into {len(content_chunks)} chunks")
            
            all_qa_items = []
            for i, chunk in enumerate(content_chunks, 1):
                print(f"\nProcessing chunk {i}/{len(content_chunks)}...")
                
                chunk_prompt = prompt.replace(content, chunk)
                
                # Generate without streaming for structured output
                response = model.generate_content(
                    chunk_prompt,
                    generation_config=GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        max_output_tokens=8192,
                        temperature=0.7
                    )
                )
                
                try:
                    # Parse response directly
                    chunk_data = json.loads(response.text)
                    if chunk_data and 'qa_items' in chunk_data:
                        print(f"\nGenerated {len(chunk_data['qa_items'])} Q&As in this chunk")
                        for qa in chunk_data['qa_items'][:2]:
                            print(f"\n  Q: {qa['question'][:100]}...")
                        print("  ...")
                        
                        all_qa_items.extend(chunk_data['qa_items'])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse chunk {i}")
                    logger.error(f"Error details: {str(e)}")
                    logger.error(f"Response text: {response.text[:500]}...")
                    continue  # Skip failed chunk instead of failing completely
            
            # Combine all Q&As
            qa_data = {"qa_items": all_qa_items}
            print(f"\n\nTotal Q&As generated: {len(all_qa_items)}")
            
            # Save Q&A to a new file
            output_filename = filename.replace('.md', '_qa.md')
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(f"# Q&A Generated from {filename}\n\n")
                f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Total Q&As: {len(qa_data['qa_items'])}\n\n")
                f.write("---\n\n")
                
                for item in qa_data['qa_items']:
                    f.write(f"## {item['title']}\n\n")
                    f.write(f"**Question:** {item['question']}\n\n")
                    f.write(f"**Answer:** {item['answer']}\n\n")
                    f.write(f"**Source:** {item['source']}\n")
                    f.write(f"**Tag:** {item['tag']}\n\n")
                    f.write("---\n\n")
            
            print("\n✓ Successfully generated Q&A content")
            print(f"✓ Found {len(qa_data['qa_items'])} Q&A items")
            print(f"✓ Saved to {output_filename}")
            
            return qa_data
            
        except Exception as e:
            print("\n✗ Error generating Q&A content")
            logger.error(f"Error in Q&A generation: {str(e)}")
            if 'response_text' in locals() and response.text:
                logger.error("Full response:")
                logger.error(response.text)
            raise
            
    except Exception as e:
        print("\n✗ Process failed")
        logger.error(f"Error generating Q&A: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Q&A from markdown files')
    parser.add_argument('file', help='Markdown file to process')
    args = parser.parse_args()
    
    generate_qa(args.file) 