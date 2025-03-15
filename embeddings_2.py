from __future__ import annotations
import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client

# Load environment variables from .env (for local testing) 
# On Streamlit Cloud or other hosts, use their secrets configuration instead.
load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ExtraAttributes:
    url: str
    deadline_date: str
    event_dates: str
    discount: str

async def get_additional_attributes(content: str) -> Dict[str, str]:
    """
    Extract additional attributes (deadline_date, event_dates, discount) 
    from the content using GPT.
    """
    prompt = f"""
    You are an AI designed to extract additional event attributes from the content of a Priceless experience description.
    Your task is to return EXACTLY a JSON object with the following keys:
    - "deadline_date": The booking deadline or expiration date and end date of the experience will also be considered as the deadline (if not mentioned, use "No deadline").
    - "event_dates": The dates when the event will take place this can include multiple dates too or ranges if that is what is in the content  (if not mentioned, use "No specific event date").
    - "discount": Any discount information (if not available, use "No discount").

    Do not include any text other than the JSON object.

    Example:
    {{
        "deadline_date": "2025-03-15",
        "event_dates": "2025-03-20 to 2025-03-22",
        "discount": "10% off"
    }}

    Content:
    {content}
    """

    print("the prompt is ")
    print(prompt)
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are an AI that extracts event attributes."},
                {"role": "user", "content": prompt}
            ]
        )
        extracted_json = response.choices[0].message.content.strip()
        print(f"Extracted JSON: {extracted_json}")
        return json.loads(extracted_json)
    except Exception as e:
        print(f"Error extracting additional attributes: {e}")
        return {
            "deadline_date": "No deadline",
            "event_dates": "No specific event date",
            "discount": "No discount"
        }

async def process_file(file_path: str) -> Optional[ExtraAttributes]:
    """
    Process a single file: read its content, extract the extra attributes,
    and update the corresponding row in the Supabase 'site_pages' table.
    """
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        print(f"Processing file: {file_path}")
        print(f"Content: {content}")
        # Extract URL from the first line (assuming format: "URL: <actual URL>")
        first_line = content.splitlines()[0]
        if first_line.startswith("URL: "):
            url = first_line[5:].strip()
        else:
            raise ValueError(f"No valid URL found in the first line of {file_path}")
        
        # Extract additional attributes using GPT
        attributes = await get_additional_attributes(content)
        print(f"Extracted attributes for {url}: {attributes}")
        
        # Create an instance of ExtraAttributes
        extra = ExtraAttributes(
            url=url,
            deadline_date=attributes.get("deadline_date", "No deadline"),
            event_dates=attributes.get("event_dates", "No specific event date"),
            discount=attributes.get("discount", "No discount")
        )
        
        # Prepare data for updating the record
        data = {
            "deadline_date": extra.deadline_date,
            "event_dates": extra.event_dates,
            "discount": extra.discount
        }
        
        # Update the record in the 'site_pages' table based on the URL
        update_result = supabase.table("site_pages").update(data).eq("url", url).execute()
        # print(f"Updated extra attributes for {url}: {update_result}")
        
        return extra
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

async def process_directory(directory_path: str) -> List[ExtraAttributes]:
    """
    Process all files in the specified directory and update their extra attributes in the database.
    """
    extra_attributes_list = []
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".txt")]
    for file_path in files:
        extra = await process_file(file_path)
        if extra:
            extra_attributes_list.append(extra)
    return extra_attributes_list
# async def process_directory(directory_path: str) -> List[ExtraAttributes]:
#     """
#     Process the first 5 files in the specified directory and update their extra attributes in the database.
#     """
#     extra_attributes_list = []
#     files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".txt")]
    
#     # Limit to the first 5 files
#     files_to_process = files[:20]  # This takes only the first 5 files
    
#     for file_path in files_to_process:
#         extra = await process_file(file_path)
#         if extra:
#             extra_attributes_list.append(extra)
#     return extra_attributes_list

async def main():
    directory_path = "scraped_experiences"
    extra_attributes = await process_directory(directory_path)
    print(f"Processed extra attributes for {len(extra_attributes)} files.")
    for attr in extra_attributes:
        print(attr)

if __name__ == "__main__":
    asyncio.run(main())
