import os
import logging
from dotenv import load_dotenv
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.auth import default
import json
import datetime
import requests
from urllib.parse import urlparse
import re

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

def download_pdf(url, company_name, filename=None):
    """Download a PDF file and save it to the company folder"""
    try:
        # Create company folder if it doesn't exist
        os.makedirs(f"pdfs/{company_name}", exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            # Extract filename from URL or generate one
            parsed_url = urlparse(url)
            url_filename = os.path.basename(parsed_url.path)
            if url_filename and url_filename.lower().endswith('.pdf'):
                filename = url_filename
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"document_{timestamp}.pdf"
        
        filepath = os.path.join('pdfs', company_name, filename)
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Verify it's a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type:
            raise Exception(f"URL does not point to a PDF file: {content_type}")
        
        # Save the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✓ Downloaded: {filename}")
        return filepath
        
    except Exception as e:
        print(f"✗ Failed to download {url}: {str(e)}")
        return None

def extract_pdfs(filename):
    """Extract and download employee-related PDFs from markdown content"""
    try:
        print(f"\nProcessing file: {filename}")
        print("="*80)
        
        # Extract company name from filename
        company_name = filename.split('_')[0]
        print(f"Company name: {company_name}")
        
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
        Analyze the following content and extract all PDF URLs that are related to:
        - Employee engagement
        - Employee benefits
        - Career development
        - Company culture
        - Diversity and inclusion
        - Employee policies
        - Employee handbooks
        - Candidate information
        - Onboarding materials
        - Training documents
        - Any other employee or candidate-related materials

        Document Content:
        {content}

        Format the output as JSON with this structure:
        {{
            "pdf_files": [
                {{
                    "url": "https://example.com/document.pdf",
                    "title": "Descriptive title of the PDF",
                    "category": "benefits/culture/policy/etc",
                    "description": "Brief description of the document content"
                }}
            ]
        }}
        
        Only include URLs that end in .pdf
        Wrap the JSON in ```json``` code blocks.
        """
        print("✓ Prompt prepared")
        
        # Generate PDF list with better error handling
        try:
            print("\nAnalyzing content for PDFs...")
            print("This may take a few minutes. Please wait...")
            print("Progress: ", end='', flush=True)
            
            model = GenerativeModel('gemini-1.5-pro')
            
            # Generate content with progress updates
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "candidate_count": 1,
                    "max_output_tokens": 8192,
                },
                stream=True
            )
            
            # Collect the response while showing progress
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    print(".", end='', flush=True)
            
            response_text = full_response
            print(" Done!")
            
            if not response_text:
                raise Exception("Empty response from Vertex AI")
            
            # Extract JSON from response
            json_start = response_text.find('```json')
            if json_start == -1:
                raise Exception("No JSON block found in response")
            
            json_end = response_text.find('```', json_start + 7)
            if json_end == -1 or json_end <= json_start + 7:
                raise Exception("Unclosed JSON block in response")
            
            # Extract just the JSON content between the first set of backticks
            json_str = response_text[json_start + 7:json_end].strip()
            
            try:
                pdf_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON response")
                logger.error(f"JSON content: {json_str}")
                logger.error(f"Error details: {str(e)}")
                raise Exception("Failed to parse JSON response") from e
            
            if not pdf_data.get('pdf_files'):
                print("\n✓ No PDF files found")
                return
            
            # List PDFs before downloading
            print(f"\nFound {len(pdf_data['pdf_files'])} PDF files:")
            print("="*80)
            for i, pdf in enumerate(pdf_data['pdf_files'], 1):
                print(f"\n{i}. {pdf['title']}")
                print(f"   URL: {pdf['url']}")
                print(f"   Category: {pdf['category']}")
                print(f"   Description: {pdf['description']}")
            print("="*80)
            
            # Ask for confirmation
            response = input("\nWould you like to download these PDFs? (y/n): ")
            if response.lower() != 'y':
                print("Download cancelled")
                return
            
            # Download PDFs
            print("\nDownloading PDFs...")
            
            downloaded_files = []
            for pdf in pdf_data['pdf_files']:
                url = pdf['url']
                if url.lower().endswith('.pdf'):
                    filepath = download_pdf(url, company_name)
                    if filepath:
                        downloaded_files.append({
                            'filepath': filepath,
                            'title': pdf['title'],
                            'category': pdf['category'],
                            'description': pdf['description']
                        })
            
            # Save metadata
            if downloaded_files:
                metadata_file = os.path.join('pdfs', company_name, 'pdf_metadata.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'downloaded_on': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'source_file': filename,
                        'pdfs': downloaded_files
                    }, f, indent=2)
                print(f"\n✓ Saved metadata to {metadata_file}")
            
            print(f"\n✓ Downloaded {len(downloaded_files)} PDF files")
            
        except Exception as e:
            print("\n✗ Error processing content")
            logger.error(f"Error: {str(e)}")
            if 'response_text' in locals() and response_text:
                logger.error("Full response:")
                logger.error(response_text)
            raise
            
    except Exception as e:
        print("\n✗ Process failed")
        logger.error(f"Error extracting PDFs: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and download employee-related PDFs from markdown files')
    parser.add_argument('file', help='Markdown file to process')
    args = parser.parse_args()
    
    extract_pdfs(args.file) 