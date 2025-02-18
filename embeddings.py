import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client
load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)


@dataclass
class ProcessedChunk:
    url: str
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    date: str
    category: str
    location: str

async def get_title_summary_link_location_date(content: str) -> Dict[str, Any]:
    """Extract title, summary, date, location, and category using GPT-4 with one-shot prompting."""
    system_prompt = """
    You are an AI designed to extract structured information from text documents for an e-commerce platform.
    Your task is to return a JSON object with the following fields:
    - 'title': A concise title for the content.
    - 'summary': A brief summary of the content.
    - 'url': The URL of the experience.
    - 'date': The relevant date or availability window mentioned in the content.
    - 'location': The physical location (e.g., city, country) mentioned in the content.
    - 'category': One of the following predefined categories: 
        [Arts and Culture, Sports, Culinary, Travel, Shopping, Entertainment, Health and Wellness]. 
      Choose the closest match based on the content. Do not invent new categories.
    Provide clear and concise values for each field.

    Here is an example:
    Content:
    Experience the magic of Venice with a guided gondola tour. 
    Discover the city's history, enjoy breathtaking views, and savor local cuisine.
    Date: Available from March 2024.
    Location: Venice, Italy.
    URL: https://www.example.com/venice-gondola-tour.

    Output:
    {
        "title": "Venice Guided Gondola Tour",
        "summary": "A guided gondola tour in Venice, exploring history, views, and local cuisine.",
        "url": "https://www.example.com/venice-gondola-tour",
        "date": "Available from March 2024",
        "location": "Venice, Italy",
        "category": "Travel"
    }

    Follow the example format as closely as possible.
    """

    try:
        # Call OpenAI API with the prompt and content
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Content:\n{content[:1500]}..."}  # Truncate to avoid token limits
            ]
        )
        
        # Extract and validate the response
        extracted_json = response.choices[0].message.content.strip()
        if not extracted_json:
            raise ValueError("Empty response from API.")
        
        print("This is the correct response that is working:")
        print(extracted_json)

        # Parse the JSON response
        return json.loads(extracted_json)
    except json.JSONDecodeError as decode_err:
        print(f"Error decoding JSON response: {decode_err}")
        print(f"Raw response: {response.choices[0].message.content if 'response' in locals() else 'No response available.'}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary",
            "url": "Unknown",
            "date": "Unknown",
            "location": "Unknown",
            "category": "Unknown"
        }
    except Exception as e:
        print(f"Error getting data from OpenAI API: {e}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary",
            "url": "Unknown",
            "date": "Unknown",
            "location": "Unknown",
            "category": "Unknown"
        }
    
#need to change the embedding fucntion so that the summary and other important information is also stored in the embeddings
async def get_embedding(content: str) -> List[float]:
    """Get embedding vector for the content from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=content
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


import json

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into the database."""
    try:
        # Print the processed chunk to inspect its structure
        print(f"Processed chunk details:\n")
        print(f"URL: {chunk.url}")
        print(f"Title: {chunk.title}")
        print(f"Summary: {chunk.summary}")
        print(f"Content: {chunk.content}")
        print(f"Metadata: {json.dumps(chunk.metadata, indent=2)}")  # Pretty print metadata
        print(f"Date: {chunk.date}")
        print(f"Category: {chunk.category}")
        print(f"Location: {chunk.location}")
        
        # Prepare the data for insertion
        data = {
            "url": chunk.url,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": json.dumps(chunk.metadata),  # Convert metadata dict to JSON string
            "embedding": chunk.embedding,
            "date": chunk.date,
            "category": chunk.category,
            "location": chunk.location
        }

        # Insert the chunk into the database
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk for {chunk.url}")
        
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None


async def process_file(file_path: str) -> ProcessedChunk:
    """Process a single file by reading its content, extracting data, and generating embeddings."""
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Extract URL from the content
        first_line = content.splitlines()[0]
        if first_line.startswith("URL: "):
            url = first_line[5:].strip()
        else:
            raise ValueError(f"No valid URL found in the first line of {file_path}")

        # Extract data (title, summary, date, category, location, etc.)
        extracted_data = await get_title_summary_link_location_date(content)

        # Generate embedding for the content
        embedding = await get_embedding(content)

        # Metadata generation
        metadata = {
            "source": "scraped_experiences",
            "chunk_size": len(content),
            "crawled_at": datetime.now(timezone.utc).isoformat()
        }

        # Create ProcessedChunk instance
        processed_chunk = ProcessedChunk(
            url=extracted_data.get("url", url),
            title=extracted_data.get("title", "Untitled"),
            summary=extracted_data.get("summary", "No summary available"),
            content=content,
            metadata=metadata,
            embedding=embedding,
            date=extracted_data.get("date", "Unknown"),
            category=extracted_data.get("category", "Unknown"),
            location=extracted_data.get("location", "Unknown")
        )

        # Insert the processed chunk into the database
        await insert_chunk(processed_chunk)

        return processed_chunk

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


async def process_directory(directory_path: str) -> List[ProcessedChunk]:
    """Process all files in the given directory."""
    processed_chunks = []
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".txt")]
    for idx, file_path in enumerate(files, start=1):
        processed_chunk = await process_file(file_path)
        if processed_chunk:
            processed_chunks.append(processed_chunk)
    return processed_chunks


# Example Usage:
# Directory with scraped experience files
async def main():
    directory_path = "scraped_experiences"
    processed_chunks = await process_directory(directory_path)

    # Output the processed chunks for debugging or save to a database
    print(f"Processed {len(processed_chunks)} files.")
    for chunk in processed_chunks[:5]:  # Print the first 5 chunks for review
        print(chunk)

# Run the script
if __name__ == "__main__":
    asyncio.run(main())
