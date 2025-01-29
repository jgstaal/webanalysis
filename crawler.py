import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import datetime
import json
import os
import logging
from dotenv import load_dotenv
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.auth import default
import sys
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def init_vertex_ai():
    try:
        # Check if GOOGLE_APPLICATION_CREDENTIALS is set
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        logger.info(f"Credentials path: {creds_path}")
        
        if not creds_path:
            logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
            return False
            
        # Check if the credentials file exists
        if not os.path.exists(creds_path):
            logger.error(f"Credentials file not found at: {creds_path}")
            return False
            
        # Try to load and validate the credentials file content
        try:
            with open(creds_path, 'r') as f:
                cred_data = json.loads(f.read())
                logger.info(f"Credential type: {cred_data.get('type')}")
                if cred_data.get('type') != 'service_account':
                    logger.error("Invalid credentials file: not a service account key")
                    return False
        except json.JSONDecodeError:
            logger.error("Invalid credentials file: not valid JSON")
            return False
        except Exception as e:
            logger.error(f"Error reading credentials file: {str(e)}")
            return False
            
        # Try to load the credentials
        credentials, project = default()
        logger.info(f"Loaded credentials for project: {project}")
        
        # Initialize Vertex AI
        project_id = "future-footing-423417-b1"
        location = "us-central1"
        
        vertexai.init(
            project=project_id, 
            location=location,
            credentials=credentials
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {str(e)}")
        return False

def invoke_vertex_ai(prompt, model_name='gemini-1.5-pro'):
    try:
        logger.info(f"Starting Vertex AI invocation with model: {model_name}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.debug(f"Prompt content: {prompt[:500]}...")
        
        if not init_vertex_ai():
            raise Exception("Vertex AI initialization failed")
            
        model = GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        logger.info(f"Received response from Vertex AI. Response length: {len(response.text) if response.text else 0}")
        logger.debug(f"Response content: {response.text[:500]}...")
        
        return response.text if response.text else ""
    except Exception as e:
        logger.error("Error invoking Vertex AI with model %s: %s", model_name, str(e))
        return ""

async def process_content_with_ai(content, url):
    prompt = f"""
    Analyze the following webpage content from {url}. Please provide:
    1. A brief summary of the main topics
    2. Key information points
    3. Important links identified
    4. Any notable sections or categories
    5. Create a JSON list of all URLs that have a relation with employee engagement. Think about culture, inclusion, belonging, wellness, mental health, shifts, leadership development etc.
    The JSON should be in this format:
    {{
        "employee_related_urls": [
            {{
                "url": "https://example.com/careers",
                "category": "Careers",
                "description": "Main careers page"
            }},
            ...
        ]
    }}
    Make sure to wrap the JSON in ```json``` code blocks.

    Content:
    {content}
    """
    
    response = invoke_vertex_ai(prompt)
    
    # Try to extract and parse the JSON from the response
    try:
        # Find JSON block in the response
        json_start = response.find('```json')
        json_end = response.find('```', json_start + 7)
        if json_start != -1 and json_end != -1:
            json_str = response[json_start + 7:json_end].strip()
            urls_data = json.loads(json_str)
            
            # Print employee-related URLs to console
            print("\nEmployee-related URLs found:")
            print("="*80)
            for item in urls_data.get('employee_related_urls', []):
                print(f"\nURL: {item['url']}")
                print(f"Category: {item['category']}")
                print(f"Description: {item['description']}")
            print("="*80)
    except Exception as e:
        logger.error(f"Error parsing JSON from AI response: {str(e)}")
    
    return response

async def crawl_employee_urls(urls_data, timestamp, company_name):
    """Crawl all employee-related URLs and save to a separate file"""
    print("\nStarting to crawl employee-related URLs...")
    
    async with AsyncWebCrawler() as crawler:
        browser_config = BrowserConfig(
            headless=True,
            viewport_width=1280,
            viewport_height=720
        )
        
        crawler_config = CrawlerRunConfig(
            word_count_threshold=200,
            extraction_strategy=None,
            markdown_generator=None,
            cache_mode=CacheMode.BYPASS,
            wait_for="css:body",
            screenshot=False,
            pdf=False,
            verbose=True
        )
        
        # Create a file for employee-related content with company name
        filename = f'{company_name}_{timestamp}_employee.md'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Employee-Related Pages Content\n")
            f.write(f"Crawled on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Crawl each URL
            for item in urls_data.get('employee_related_urls', []):
                url = item['url']
                print(f"\nCrawling employee URL: {url}")
                
                result = await crawler.arun(
                    url=url,
                    browser_config=browser_config,
                    run_config=crawler_config
                )
                
                if result.success:
                    print(f"✓ Successfully crawled: {url}")
                    
                    # Write to file
                    f.write(f"\n{'='*80}\n")
                    f.write(f"## {item['category']}\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"Description: {item['description']}\n\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(result.markdown)
                    f.write("\n\n")
                else:
                    print(f"✗ Failed to crawl: {url}")
                    f.write(f"\nFailed to crawl: {url}\n\n")
        
        print(f"\nSaved employee pages content to {filename}")

def parse_url_argument():
    parser = argparse.ArgumentParser(description='Web crawler for company websites')
    parser.add_argument('domain', help='Domain to crawl (e.g., bbinsurance.com)')
    args = parser.parse_args()
    
    # Clean and standardize the domain name for the company name
    domain = args.domain.lower().replace('http://', '').replace('https://', '').replace('www.', '')
    company_name = domain.split('.')[0].replace('-', '')  # Remove hyphens
    base_url = f"https://www.{domain}"
    
    return base_url, company_name  # Return both URL and standardized company name

async def main():
    # Replace the hardcoded BASE_URL with the parsed argument
    try:
        BASE_URL, company_name = parse_url_argument()  # Get both values
    except Exception as e:
        print(f"Error: Please provide a valid domain name. Example: python crawler.py bbinsurance.com")
        print(f"Error details: {str(e)}")
        sys.exit(1)
    
    print("\nStarting crawler with base URL:", BASE_URL)
    print("="*80)
    
    async with AsyncWebCrawler() as crawler:
        browser_config = BrowserConfig(
            headless=True,
            viewport_width=1280,
            viewport_height=720
        )
        
        # JavaScript to handle dynamic content
        js_code = """
        // Debug helper
        function debugLog(msg) {
            console.log('[Debug] ' + msg);
            return msg;
        }

        debugLog('Starting page analysis...');
        await new Promise(r => setTimeout(r, 5000));

        // Get all links and return them as a string (JSON)
        const links = Array.from(document.querySelectorAll('a[href]')).map(link => {
            try {
                return {
                    url: link.href,
                    text: link.textContent.trim(),
                    path: link.pathname
                };
            } catch (e) {
                debugLog(`Error processing link: ${e.message}`);
                return null;
            }
        }).filter(Boolean);

        debugLog(`Found ${links.length} links`);
        return JSON.stringify(links);
        """
        
        crawler_config = CrawlerRunConfig(
            word_count_threshold=200,
            extraction_strategy=None,
            markdown_generator=None,
            cache_mode=CacheMode.BYPASS,
            js_code=js_code,
            wait_for="css:body",
            screenshot=False,
            pdf=False,
            verbose=True
        )
        
        # Crawl the base URL
        result = await crawler.arun(
            url=BASE_URL,
            browser_config=browser_config,
            run_config=crawler_config
        )
        
        if result.success:
            print(f"✓ Successfully crawled: {BASE_URL}")
            
            try:
                # Parse the extracted links
                links = json.loads(result.extracted_content) if result.extracted_content else []
                print(f"\nFound {len(links)} links on the page")
                
                # Process content with Vertex AI
                ai_analysis = await process_content_with_ai(result.markdown, BASE_URL)
                
                # Save the content
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'{company_name}_{timestamp}_generic.md'
                
                # Try to extract and parse the JSON from the AI response
                try:
                    json_start = ai_analysis.find('```json')
                    json_end = ai_analysis.find('```', json_start + 7)
                    if json_start != -1 and json_end != -1:
                        json_str = ai_analysis[json_start + 7:json_end].strip()
                        urls_data = json.loads(json_str)
                        
                        # Crawl the employee-related URLs
                        await crawl_employee_urls(urls_data, timestamp, company_name)
                except Exception as e:
                    logger.error(f"Error processing employee URLs: {str(e)}")
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# Brown & Brown Insurance Website Content\n")
                    f.write(f"Crawled on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"URL: {BASE_URL}\n\n")
                    f.write("="*80 + "\n\n")
                    
                    # Add AI Analysis section
                    f.write("# AI Analysis\n\n")
                    f.write(ai_analysis)
                    f.write("\n\n" + "="*80 + "\n\n")
                    
                    # Original content
                    f.write("# Original Content\n\n")
                    f.write(result.markdown)
                    
                    # Add list of links at the end
                    f.write("\n\n" + "="*80 + "\n")
                    f.write("# All Links Found\n\n")
                    for link in links:
                        f.write(f"- [{link['text']}]({link['url']})\n")
                
                print(f"\nSaved content with AI analysis to {filename}")
                
            except json.JSONDecodeError:
                print("Failed to parse extracted links")
            except Exception as e:
                print(f"Error processing content: {str(e)}")
        else:
            print(f"✗ Failed to crawl: {BASE_URL}")

if __name__ == "__main__":
    asyncio.run(main()) 