from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
from typing import Dict
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List
from openai import OpenAI
import streamlit as st
import datetime
import json
from typing import Optional, Tuple

# load_dotenv()

client_work = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],  
)

llm = st.secrets.get("LLM_MODEL", "gpt-4o-mini")
model = OpenAIModel(llm)

# load_dotenv()

# client_work = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
# )

# llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
# model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

# Memory list for storing the latest 3 user inputs and 3 model outputs
memory: List[str] = []
show_more_match_count = 5  # Initialize with the default value


def update_memory(user_message: str, model_response: str) -> None:
    """
    Update the memory list with the latest user input and model output.

    Args:
        user_message (str): The user's input.
        model_response (str): The model's output.
    """
    global memory

    # Append user input and model response with appropriate labels
    memory.append(f"User: {user_message}")
    memory.append(f"Model: {model_response}")

    # Ensure memory contains only the latest 6 messages
    if len(memory) > 6:
        memory = memory[-6:]

def print_system_prompt():
    memory_context = get_memory_context()
    formatted_prompt = system_prompt_template.format(memory_context=memory_context)
    print(formatted_prompt)



def get_memory_context() -> str:
    """
    Get the memory context from the memory list.

    Returns:
        str: A formatted string containing the memory messages.
    """
    return "\n".join(memory)

system_prompt_template = """
You are an expert chatbot designed to provide detailed information about Priceless experiences.

You have access to a database of experiences that includes their titles, summaries, URLs, dates, locations, types of events, and additional metadata. Your role is to assist users by retrieving and presenting this information based on their queries.

You also have access to memory, which includes the user's past queries and your responses. Use the following memory context to enhance your responses if necessary like for example if the user asks for more related results( for example give more similar results) or a follow up like i want only in italy then you can use the memory to get the previous queries and then use that to get the results.:
{memory_context}
When responding:
1. Always include the **title**, **summary**, **URL**, **date**, **location**, **type of event**, and any other relevant details available for the experience.
2. Use the embeddings to find the most relevant matches based on the user's query.
3. If the user asks a general query which is very general, then refine the query to get better results .Only do this when you think is necessary
4. If you cannot find an exact match or if the information is not available, let the user know honestly and provide alternative suggestions if possible.
5. If the user refers to previous responses, refine your answer accordingly using memory.

Your responses should be clear, concise, and informative, always prioritizing the user’s intent. For example:

You are not designed to answer unrelated questions or perform tasks outside your domain.
Always aim to provide the most relevant and actionable insights based on the user's request.

"""
chatbot = Agent(
    model,
    system_prompt=system_prompt_template.format(memory_context=get_memory_context()),
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        # print("the response from the get_embedding function is", response)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error
    
async def classify_query(openai_client: AsyncOpenAI, user_query: str) -> str:
    """
    Use ChatGPT to classify the query into one of the predefined categories.

    Args:
        openai_client: The OpenAI client instance.
        user_query: The user's general query.

    Returns:
        A string representing the category or None if classification fails.
    """
    categories = [
        "Arts and Culture", "Sports", "Culinary", "Travel",
        "Shopping", "Entertainment", "Health and Wellness"
    ]

    # Log the input query
    print(f"Classifying query: {user_query}")

    prompt = f"""
    Classify the following query into one of these categories: {', '.join(categories)} also the answer should be one of these only no extra characters or words.
    Query: "{user_query}"
    """

    try:
        # Log before making the API call
        print(f"Sending classification request to OpenAI with prompt: {prompt}")

        # OpenAI API call
        response =  client_work.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        # Log the response from OpenAI
        print(f"Received response from OpenAI: {response}")

        classification = response.choices[0].message.content.strip()

        # Log the classification result
        print(f"Classification result: {classification}")

        if classification in categories:
            print(f"Query classified as: {classification}")
            return classification

        print(f"Classification not valid, returning None.")
        return None

    except Exception as e:
        print(f"Error during classification: {e}")
        return None

    
def retrieve_summaries(supabase_client: Client, category: str) -> List[str]:
    """
    Retrieve summaries of experiences from Supabase for a specific category.
    
    Args:
        supabase_client: The Supabase client instance.
        category: The category to filter by.
        
    Returns:
        A list of summaries or an empty list if no data is found.
    """
    print(f"Retrieving summaries for category: {category}")
    
    try:
        response = supabase_client.table('site_pages').select('summary').eq('category', category).execute()
        
        # Log the response from Supabase
        # print(f"Received response from Supabase: {response}")
        
        if response.data:
            summaries = [item['summary'] for item in response.data]
            print(f"Found {len(summaries)} summaries.")
            return summaries
        
        print(f"No summaries found for category '{category}'.")
        return []
    
    except Exception as e:
        print(f"Error retrieving summaries: {e}")
        return []

    
async def refine_with_summaries(openai_client: AsyncOpenAI, user_query: str, summaries: List[str]) -> Tuple[str, bool]:
    """
    Use ChatGPT to refine the query based on retrieved summaries and decide if follow-up is needed.
    
    Args:
        openai_client: The OpenAI client instance.
        user_query: The user's original query.
        summaries: Summaries retrieved from Supabase.
        
    Returns:
        A tuple containing the refined query and a boolean indicating if follow-up is needed.
    """
    print(f"Refining query based on summaries for query: {user_query}")
    
    # Limit the summaries to the first 10 for brevity
    summaries_text = "\n".join(summaries[:70])
    print(f"Summaries being sent to OpenAI: {summaries_text[:200]}...")  # Print a preview (first 200 characters) of the summaries for debugging
    
    prompt = f"""
    The user asked: "{user_query}"
    Here are some relevant summaries based on the category of the query:
    {summaries_text}
    
    Refine the query so it better matches the available information you have been given many summaries based on this you should refine the query into a comprehensive query atleast 5 lines which can retrieve the best results by adapting with the words that it sees through summaries.It should not use the specific names of the of the places in summaries but should be a general query that can be used to retrieve the best results. 
    If the query cannot be refined without asking for more details, suggest a follow-up question this follow up question should only be question that you see best and nothing else and should contain some information that you want to ask from the user (note that follow up should asked in worst condition when the query is very vague).
    If the refined query is given then the follow up will be false but if the follow up is asked only then the follow up will be true.
    It cannot be that you give a refined query and and follow up is true . Follow up will only be true if a question is asked and nothing else
    Return the result in the following format:
    - Refined Query: <refined_query>
    - Needs Follow-Up: <true/false>
    """
    
    try:
        # Log the prompt being sent to OpenAI
        print(f"Sending refinement request to OpenAI with prompt: {prompt[:200]}...")  # Print a preview of the prompt
        
        response = client_work.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a query refinement assistant."},
                      {"role": "user", "content": prompt}]
        )
        
        # Log the response from OpenAI
        print(f"Received response from OpenAI: {response}")
        
        response_text = response.choices[0].message.content.strip()
        print(f"Response text from OpenAI: {response_text[:200]}...")  # Log a preview (first 200 characters) of the response
        
        # Extract refined query and follow-up status
        refined_query = response_text.split("- Refined Query:")[1].split("- Needs Follow-Up:")[0].strip()
        needs_follow_up = response_text.split("- Needs Follow-Up:")[1].strip().lower() == "true"
        
        print(f"Refined query: {refined_query}")
        print(f"Needs follow-up: {needs_follow_up}")
        
        return refined_query, needs_follow_up
    
    except Exception as e:
        print(f"Error during query refinement: {e}")
        return user_query, True  # Default to needing a follow-up


    

@chatbot.tool
async def retrieve_relevant_experiences(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant experiences based on the user's query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant experiences
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant experiences
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5
                
            }
        ).execute()
        print("the result is ", result)
 

        formatted_experiences = []
        for experience in result.data:
            experience_text = f"""
            **Title**: {experience['title']}
            **Summary**: {experience['summary']}
            **Date**: {experience.get('date', 'N/A')}
            **Location**: {experience.get('location', 'N/A')}
            **Type of Event**: {experience.get('category', 'N/A')}
            **URL**: {experience['url']}
            """
            formatted_experiences.append(experience_text)
        model_response = "\n\n".join(formatted_experiences)
        update_memory(user_query, model_response)
               # Check if any relevant experiences were found

        # print("the retrieved results by the retrieve_relevant_experiences function is", result.data)   
        # Format the results
        # print("the relevant experiences returned the following result")
        # # print(result.data)
        print("the formatted experiences are ", formatted_experiences)
        if not result.data:
            print("No relevant experiences found. Refining the query...")

            try:
                refined_result = await refine_general_query(ctx, user_query)
                print("Refined query results:", refined_result)
                return refined_result
            except Exception as e:
                print("Error in refine_general_query:", str(e))
                return "No relevant experiences found, and refinement failed."
        
        return "\n\n".join(formatted_experiences)

    except Exception as e:
        return f"An error occurred: {str(e)}"

@chatbot.tool
async def refine_general_query(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Handle general queries by classifying them, retrieving relevant summaries, 
    and refining the query for better results.
    
    Args:
        ctx: Context including the Supabase client and OpenAI client.
        user_query: The user's general query.
        
    Returns:
        A response with refined query results or a follow-up question.
    """
    try:
        # Step 1: Classify the query into one of the categories
        print(f"Classifying user query: {user_query}")
        category = await classify_query(ctx.deps.openai_client, user_query)
        
        if not category:
            print("Unable to classify the query.")
            return "I'm unable to classify your query. Could you provide more specific details?"
        
        print(f"Query classified under category: {category}")

        # Step 2: Retrieve summaries from Supabase based on the category
        print(f"Retrieving summaries for category: {category}")
        summaries = retrieve_summaries(ctx.deps.supabase, category)
        
        if not summaries:
            print(f"No experiences found under the category '{category}'.")
            return f"No experiences found under the category '{category}'. Please refine your query."
        
        print(f"Retrieved {len(summaries)} summaries from the database.")

        # Step 3: Refine the query using ChatGPT
        print("Refining query using retrieved summaries...")
        refined_query, needs_follow_up = await refine_with_summaries(
            ctx.deps.openai_client, user_query, summaries
        )
        
        if needs_follow_up:
            print(f"Follow-up needed for refined query: {refined_query}")
            return f"The retrieved information suggests that more details are needed. Could you clarify: {refined_query}"
        
        print(f"Refined query: {refined_query}")

        # Step 4: Retrieve results based on the refined query
        print("Fetching query embedding for refined query...")
        query_embedding = await get_embedding(refined_query, ctx.deps.openai_client)
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5
            }
        ).execute()
        
        if not result.data:
            print("No relevant experiences found based on refined query.")
            return "No relevant experiences found based on your refined query."
        
        print(f"Found {len(result.data)} matching experiences.")

        # Format the results
        formatted_experiences = []
        for experience in result.data:
            experience_text = f"""
            **Title**: {experience['title']}
            **Summary**: {experience['summary']}
            **Date**: {experience.get('date', 'N/A')}
            **Location**: {experience.get('location', 'N/A')}
            **Type of Event**: {experience.get('category', 'N/A')}
            **URL**: {experience['url']}
            """
            formatted_experiences.append(experience_text)
        model_response = "\n\n".join(formatted_experiences)
        update_memory(user_query, model_response)

        print("Returning formatted experiences to the user.")
        return "\n\n".join(formatted_experiences)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"
    
# ok now we are going to try to add memory to the chatbot so that it can remember the user's previous queries and then use that to answer
@chatbot.tool
async def handle_memory_based_queries(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Handle memory-based queries by classifying the user query into scenarios:
    1. Request for more results (e.g., "Show me more options similar to these or give more results that are not the same but similar").
    2. Follow-up queries for refinement (e.g., "Filter this to Italy or like tell for india ").
    3. Repeated requests for specific details (e.g., "What are the timings for this? or what is the detail of the first experience").

    Args:
        ctx: Context including dependencies like OpenAI client and Supabase.
        user_query: The user's current query.

    Returns:
        A response based on the query classification.
    """
    global memory
    print("here is the system prompt template ")
    # Call the function to print the system prompt template
    print_system_prompt()

    try:
        # Print user query received
        print(f"Received user query: {user_query}")

        # Prepare the prompt for OpenAI classification
        prompt = f"""
        You are a query refinement assistant. Your job is to classify the user's current query into one of three categories:
        
        1) "Show more" for cases where the user wants additional results based on prior responses, such as:
           - "Show me more options similar to these."
           - "This is great, but I'm looking for something more luxurious."
        
        2) "Extend" for cases where the user refines or corrects the query with additional details, such as:
           - "Can you filter this to only show experiences in Italy?"
           - "Actually, I meant family-friendly experiences, not solo ones."
        
        3) "Specific" for cases where the user asks for additional details on a specific result, such as:
           - "What are the timings for the New York wine tasting?"
           - "Give me details about the second option."

        User query: "{user_query}"
        Based on this input, classify the query and return one of these three labels: "show more", "extend", or "specific".
        """

        # Print the prompt before making the OpenAI call
        print(f"Prepared prompt for classification: {prompt}")

        # OpenAI call to classify the query
        response = client_work.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a query refinement assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Print the raw response from OpenAI
        print(f"OpenAI response: {response}")

        # Parse the classification result
        classification = response.choices[0].message.content.strip().lower()

        # Print classification result
        print(f"Classification result: {classification}")
        print(f"Classification result (raw): {repr(classification)}")
        print("is the classification equal to show more", classification == "show more")
        print("is the classification equal to extend", classification == "extend")
        print("is the classification equal to specific", classification == "specific")
        # Handle classifications using nested if-else
        if classification.strip() == '"show more"':
            print("Handling 'show more' classification.")
            global show_more_match_count  # Access the global variable
            # Increment match count for more embeddings
            show_more_match_count += 5
            print(f"Updated match count: {show_more_match_count}")

            query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
            print(f"Query embedding for 'show more': {query_embedding}")

            result = ctx.deps.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding,
                    'match_count': show_more_match_count
                }
            ).execute()
            print(f"Supabase query result for 'show more': {result.data}")
            
            if not result.data:
                return "No more results available."

            # Format and return the last 5 results
            formatted_experiences = []
            for experience in result.data[-5:]:
                experience_text = f"""
                **Title**: {experience['title']}
                **Summary**: {experience['summary']}
                **Date**: {experience.get('date', 'N/A')}
                **Location**: {experience.get('location', 'N/A')}
                **Type of Event**: {experience.get('category', 'N/A')}
                **URL**: {experience['url']}
                """
                formatted_experiences.append(experience_text)
            
            # Print formatted experiences before returning
            print(f"Formatted experiences for 'show more': {formatted_experiences}")
            return "\n\n".join(formatted_experiences)

        else:
            if classification == '"extend"':
                print("Handling 'extend' classification.")
                
                # Use the last 2 memory entries and append the new query
                context_query = f"{memory[2]} {memory[4]} {user_query}"
                print(f"Context query for 'extend': {context_query}")

                query_embedding = await get_embedding(context_query, ctx.deps.openai_client)
                print(f"Query embedding for 'extend': {query_embedding}")

                result = ctx.deps.supabase.rpc(
                    'match_site_pages',
                    {
                        'query_embedding': query_embedding,
                        'match_count': 5
                    }
                ).execute()
                print(f"Supabase query result for 'extend': {result.data}")
                
                if not result.data:
                    return "No relevant experiences found for the refined query."

                # Format and return results
                formatted_experiences = []
                for experience in result.data:
                    experience_text = f"""
                    **Title**: {experience['title']}
                    **Summary**: {experience['summary']}
                    **Date**: {experience.get('date', 'N/A')}
                    **Location**: {experience.get('location', 'N/A')}
                    **Type of Event**: {experience.get('category', 'N/A')}
                    **URL**: {experience['url']}
                    """
                    formatted_experiences.append(experience_text)
                
                # Print formatted experiences before returning
                print(f"Formatted experiences for 'extend': {formatted_experiences}")
                return "\n\n".join(formatted_experiences)

            else:
                # Handle "specific"
                try:
                    print("Handling 'specific' classification.")

                    # Step 1: Extract the last model response from memory
                    last_model_response = memory[-1]
                    print(f"Last model response: {last_model_response}")
                    # Step 2: Use OpenAI to determine the referenced URL
                    prompt = f"""
                    You are assisting with identifying a referenced experience. Below is the last model response that contains multiple experiences, and a user's query about it. 
                    Your task is to extract the URL of the experience the user is referring to, based on their query. 
                    If the user refers to a specific number (e.g., "second one", "third one") or by name (e.g., "wine tasting"), extract the matching URL.

                    Last model response:
                    {last_model_response}

                    User query:
                    {user_query}

                    Please extract and return the exact URL referenced by the user's query. If no match is found, return "No matching URL found."
                    """
                    print(f"Prompt for URL extraction: {prompt}")
                    # Call OpenAI to process the prompt
                    url_extraction_response = client_work.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an assistant helping extract URLs from text based on user queries."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    extracted_url = url_extraction_response.choices[0].message.content.strip()
                    print(f"Extracted URL: {extracted_url}")
                    # Handle case where no URL is found
                    if "No matching URL found" in extracted_url:
                        return "I couldn't find the specific experience you're referring to. Could you clarify?"

                    # Step 3: Fetch content from Supabase using the extracted URL
                    result = ctx.deps.supabase.table("site_pages").select("content").eq("url", extracted_url).execute()

                    if not result.data:
                        return "No content was found for the specified experience. Please try again."

                    content = result.data[0]['content']
                    print(f"Content for the extracted URL: {content}")
                    # Step 4: Use OpenAI to generate a specific answer based on content and query
                    detail_prompt = f"""
                    You are an assistant providing detailed answers based on a user's query and the content of a specific experience. 
                    Below is the full content of the experience and the user's query.

                    Content:
                    {content}

                    User query:
                    {user_query}

                    Please provide a detailed and specific answer to the user's query based on the content provided.
                    """
                    print(f"Prompt for detailed response: {detail_prompt}")
                    detail_response = client_work.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a detailed response assistant."},
                            {"role": "user", "content": detail_prompt}
                        ]
                    )
                    final_response = detail_response['choices'][0]['message']['content'].strip()

                    return final_response

                except Exception as e:
                    return f"An error occurred while processing your request: {str(e)}"

    except Exception as e:
        # Print the exception message
        print(f"Error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"
#making the filter based tool

import json

async def analyze_query(user_query: str) -> dict:
    """
    Determines which filters should be applied (date, location, discount) and returns a JSON with boolean values for each filter.
    """
    prompt = f"""
    Analyze the following query and determine if each filter should be applied:
    Query: "{user_query}"

    Check for the following filters:
    - A date (return true if found, false otherwise) this may various values for example Today, Tomorrow, This week, Next week, Next 2 weeks, Next month, Specific months (e.g., August), Next year, This year, Specific dates (e.g., 25th August), 
    - A location or a specific place or said that places surrounding or around some place then this filter will also be made active (return true if found, false otherwise)
    - A discount preference (yes/no) (return true if "yes" or "no" is found, false otherwise)
    
    Output the result in JSON format:
    {{"date": true/false, "location": true/false, "discount": true/false}}
    """

    print("Sending request to the model...")
    response = client_work.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": "You are an advanced query parser."},
                  {"role": "user", "content": prompt}]
    )

    # Check if response exists
    if not response or not response.choices or not response.choices[0].message.content.strip():
        print("Error: Empty response received from model!")
        return {"date": False, "location": False, "discount": False}  # Default values

    raw_response = response.choices[0].message.content.strip()
    
    # Debugging print statements
    print("Response received from model (raw):")
    print(raw_response)

    try:
        # Remove potential Markdown formatting (e.g., ```json ... ```)
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()  # Remove ```json and ```

        print("Converting response to dictionary...")
        result = json.loads(raw_response)  # Convert string to dict

        # Debugging final output
        print("Final result (JSON format):")
        print(result)

        return result
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("Raw response was:")
        print(raw_response)  # Log raw response for debugging
        return {"date": False, "location": False, "discount": False}  # Default values



CITY_LIST = [
    "Maipú, Chile", "Noord, Aruba", "Gouves, Greece", "Överstbyn, Sweden",
    "Scottsdale, Arizona", "Brits, South Africa", "Duluth, Georgia", "Panjim, India",
    "Chiang Mai, Thailand", "Veghel, Netherlands", "St Andrews, United Kingdom",
    "Quito, Ecuador", "Bordeaux, France", "Lutz, Florida", "Lawrence Township, New Jersey",
    "Otterlo, Netherlands", "Bridgetown, Barbados", "Doetinchem, Netherlands",
    "Toronto, Canada", "Freeport, Bahamas", "San Antonio, Texas", "Brummen, Netherlands",
    "Dubai, United Arab Emirates", "Kajiado, Kenya", "Murrells Inlet, South Carolina",
    "Cape Town, South Africa", "Adriatic Sea, Croatia", "Croatia", "Watamu, Kenya",
    "Saint-Germain-des-Prés, Paris, France", "Firenze, Italy", "Delhi, India",
    "Four Roads, Barbados", "Voorburg, Netherlands", "Potomac, Maryland",
    "Porto San Giorgio, Italy", "Korčula, Croatia", "Sabana Westpunt, Curaçao",
    "Colona, Illinois", "Mexico City, Mexico", "Benaulim, India", "Buenos Aires, Argentina",
    "Breskens, Netherlands", "Adamuz, Spain", "Ponte Vedra Beach, Florida", "Newport Beach, California",
    "Vari, Greece", "Lyon, France", "Berlin, Germany", "Online", "Knossos, Greece", "Olbia, Italy",
    "Mumbai, India", "Ireland", "Greenwich Village, New York, USA", "Singapore", "Lantau Island, Hong Kong",
    "Colombia", "Yarra River, Australia", "Royal Birkdale, United Kingdom", "Metropolitan City of Naples, Italy",
    "Frankfurt, Germany", "Šibenik, Croatia", "Genoa, Italy", "Catania, Italy", "Northern Ireland",
    "Dortmund, Germany", "Napoli, Italy", "Socorro, Brazil", "Urla, Turkey", "Ubud, Bali, Indonesia",
    "Thyssen-Bornemisza National Museum, New York City", "Speightstown, Barbados", "Ponce, Puerto Rico",
    "Cannes, France", "Girim, India", "Medellin, Colombia", "Hong Kong, Hong Kong", "Peñalolén, Chile",
    "Dubrovnik, Croatia", "Hoedspruit, South Africa", "Cesenatico, Province of Forlì-Cesena, Italy",
    "Paris, France", "Blitar City, Indonesia", "San Vicente de Tagua Tagua, Chile", "Girona, Spain", "Venice, Italy",
    "Alto Jahuel, Chile", "Australia", "Vitacura, Chile", "Eindhoven, Netherlands", "Cuelim, India", "Clare, Ireland",
    "Miami Beach, Florida", "New York, New York", "Bondi Beach, Australia", "Auckland, New Zealand", "Nassau, Bahamas",
    "Cataño, Puerto Rico", "United Arab Emirates", "Central Hong Kong, Hong Kong", "Santa Cruz, Chile", "Prague, Czech Republic",
    "Toledo, Spain", "Belgium", "Venezia, Italy", "Ballydehob, Ireland", "Sakvorde, India", "Johannesburg, South Africa",
    "Naples, Italy", "Kabupaten Bintan, Indonesia", "Vega Alta, Puerto Rico", "Turin, Italy", "Montego Bay, Jamaica",
    "Thailand", "At home", "Kuala Lumpur, Malaysia", "Hamburg, Germany", "Palm Coast, Florida and Various North American Cities",
    "Sani Pass, South Africa", "Blitar, Indonesia", "Metropolitan City of Florence, Italy", "Urca, Rio de Janeiro, Brazil",
    "Orlando, Florida", "Bicester, United Kingdom", "United Kingdom", "Global (over 300,000 stores)", "Saint-Nicolas-de-la-Haie, France",
    "Hardenberg, Netherlands", "Germany", "Gulpen, Netherlands", "Naples, Florida", "Cappadocia, Nevşehir, Turkey", "Valparaíso, Chile",
    "Bogotá, Colombia", "Athens, Greece", "Vilassar de Dalt, Spain", "Global", "Stockholm, Sweden", "Kabupaten Badung, Indonesia",
    "Hoog Soeren, Netherlands", "Weert, Netherlands", "Korcula, Croatia", "Saint-Étienne-de-Lisse, France", "Austin, Texas",
    "Montreal, Canada", "Boca Raton, Florida", "Sint Michiel, Curaçao", "El Manzano, Chile", "São Paulo, Brazil", "New Zealand",
    "Willemstad, Curaçao", "Lille, France", "Blackrock, Ireland", "United States (U.S. retailers)", "Hormigos, Spain", "Batam City, Indonesia",
    "Blaine, Minnesota", "Cromwell, Connecticut", "Varese, Italy", "Europe, Turkey, UAE & Asia", "Las Vegas, Nevada", "Amsterdam",
    "Bad Teinach-Zavelstein, Germany", "Selected countries of card issuance", "Seville, Spain", "Tsim Sha Tsui, Hong Kong",
    "Royal Portrush, Northern Ireland", "Egypt and Jordan", "Amsterdam, Netherlands", "Berthoud, Colorado", "San Antonio de Areco, Argentina",
    "Mexico", "Caledon, Canada", "Warrane (Sydney Cove), Sydney, Australia", "Slavonia, Croatia", "New Delhi, India", "Sarasota, Florida",
    "Maineville, Ohio", "Cuncolim, India", "Sydney, Australia", "Onderdendam, Netherlands", "Enniskerry, Ireland", "Digital content",
    "Curaçao, Curaçao", "Twickenham, United Kingdom", "Tiaong, Philippines", "Velim, India", "Lo Barnechea, Chile", "Lagos, Nigeria",
    "Ayutthaya, Thailand", "Gurugram, India", "Milan, Italy", "Khong Chiam, Thailand", "Zagreb, Croatia", "San Francisco, California",
    "Europe, Maldives, India, Middle-East & Africa", "Lane Cove National Park, Australia", "Polientes, Spain", "N/A", "London, United Kingdom",
    "Kreuzberg, Berlin, Germany", "Île-de-France, France", "Broken Bay, Australia", "Roxas, Philippines", "Kepulauan Anambas Regency, Indonesia",
    "Istria, Croatia", "Condrieu, France", "Hilton Head Island, South Carolina", "Gubbio, Italy", "Grampians, Australia", "France",
    "Zarate, Argentina", "Monesterio, Spain", "Madrid, Spain", "Tuye, Goa, India", "Global (800+ hotels in 100 countries)", "Roermond, Netherlands",
    "Nairobi, Kenya", "Serranillos Playa, Spain", "Naas, Ireland", "United States", "Narok Town, Kenya", "Middle East", "Bergen op Zoom, Netherlands",
    "Munich, Germany", "Lipa, Philippines", "Leeuwarden, Netherlands", "Petroto, Greece", "Netherlands", "Worldwide", "Rome, Italy", "Sandur, India",
    "Río Grande, Puerto Rico", "Norton, Massachusetts", "Krugersdorp, South Africa", "UK", "Dublin, Ireland", "San Juan, Puerto Rico",
    "Maastricht, Netherlands", "Spain", "Online service", "Salamanca, Spain", "Agra, India", "Papendrecht, Netherlands", "Beşiktaş, Turkey",
    "Las Palmas de Gran Canaria, Spain", "Golden Slavonia, Croatia", "Rio de Janeiro, Brazil", "Lumban, Philippines", "Falmouth, Jamaica",
    "München, Germany", "Royal Portrush Golf Club, Northern Ireland", "Bangkok, Thailand", "Grasse, France", "Valkenburg, Netherlands",
    "Italy", "Kowloon, Hong Kong", "Istanbul, Turkey", "Koewacht, Netherlands", "Panjim, Goa, India", "Mainz, Germany", "Marseille, France",
    "Dublin 8, Ireland", "Erkelenz, Germany", "El Nido, Philippines", "New York City, United States", "Dubai, UAE", "Pieve d'Alpago, Italy",
    "UAE Airports", "Oranjestad, Aruba", "Bali, Indonesia", "Nevşehir, Turkey", "Amersfoort, Netherlands", "Maipú, Chile", "Noord, Aruba",
    "Gouves, Greece", "Överstbyn, Sweden", "Scottsdale, Arizona", "Brits, South Africa", "Duluth, Georgia", "Panjim, India", "Chiang Mai, Thailand",
    "Veghel, Netherlands", "St Andrews, United Kingdom", "Quito, Ecuador", "Bordeaux, France", "Lutz, Florida", "Lawrence Township, New Jersey",
    "Otterlo, Netherlands", "Bridgetown, Barbados", "Doetinchem, Netherlands", "Toronto, Canada", "Freeport, Bahamas", "San Antonio, Texas",
    "Brummen, Netherlands", "Dubai, United Arab Emirates", "Kajiado, Kenya", "Murrells Inlet, South Carolina", "Cape Town, South Africa",
    "Adriatic Sea, Croatia", "Croatia", "Watamu, Kenya", "Saint-Germain-des-Prés, Paris, France", "Firenze, Italy", "Delhi, India", "Four Roads, Barbados",
    "Voorburg, Netherlands", "Potomac, Maryland", "Porto San Giorgio, Italy", "Korčula, Croatia", "Sabana Westpunt, Curaçao"
]

import json

async def extract_location(user_query: str) -> str:
    """
    Extracts the exact city or cities from the query, ensuring they exist in CITY_LIST.
    Also handles cases like "near me" or "around {city}" by finding the closest match.

    Returns:
        - A single city from CITY_LIST if one is mentioned.
        - Multiple cities if a broader location is referenced.
        - "None" if no valid city is found.
    """

    # Define the prompt for extracting the location
    prompt = f"""
    You are an advanced location extraction expert. Your task is to extract an exact city or location 
    from the following user query and match it to a predefined list.

    Query: "{user_query}"

    The predefined list of valid cities is:
    {CITY_LIST}

    **Rules:**
    - If the query mentions a specific city, return that city only if it exists in the list.
    - If the query says "near me" or "around "city", return the closest matching cities from the list.
    - If multiple relevant cities exist, return them all (comma-separated).
    - If no valid city is found, return "None".

    **Output Format (JSON):**
    {{"matched_cities": ["city1", "city2"]}} or {{"matched_cities": []}}
    """

    print("Sending request to the model...")
    response = client_work.chat.completions.create(
        model="gpt-4o", messages=[
            {"role": "system", "content": "You are a precise location extractor."},
            {"role": "user", "content": prompt}
        ]
    )

    # Check for an empty response
    if not response or not response.choices or not response.choices[0].message.content.strip():
        print("Error: Empty response received from model!")
        return "None"

    raw_response = response.choices[0].message.content.strip()

    # Debugging print statement
    print("Response received from model (raw):")
    print(raw_response)

    try:
        # Remove Markdown formatting if present (e.g., ```json ... ```)
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()  # Remove ```json and ```

        # Parse JSON safely
        print("Attempting to parse the response into JSON...")
        extracted_data = json.loads(raw_response)

        # Extract matched cities
        matched_cities = extracted_data.get("matched_cities", [])
        print("Matched cities found:", matched_cities)

        if matched_cities:
            return ", ".join(matched_cities)  # Return cities as a comma-separated string

        return "None"  # If no cities were found

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("Raw response was:")
        print(raw_response)  # Log raw response for debugging
        return "None"


async def extract_date( query: str) -> Optional[dict]:
    """Uses GPT to extract a structured date from a natural language query."""
    # Get the current date in YYYY-MM-DD format
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"Current date: {current_date}")

    prompt = f"""
    You are an advanced AI that can extract a date from natural language queries.
    Given a query, you need to extract a structured date that can be compared with the database values (YYYY-MM-DD).

    Current date: {current_date}

    Please extract the date from the following query:
    Query: "{query}"

    The possible date types include:
    - Specific dates (e.g., 2024-12-31)
    - Relative dates (e.g., today, tomorrow, next week, next 2 weeks, this week, next month, August, etc.)

    Output the extracted date in the format: YYYY-MM-DD and give no other text other then the date in formatYYYY-MM-DD
    """

    try:
        print(f"Sending request to GPT model with query: {query}")
        response =  client_work.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an advanced date parser."},
                      {"role": "user", "content": prompt}]
        )

        # Extract the result and clean up any formatting
        extracted_date = response.choices[0].message.content.strip()
        print(f"Extracted date from response: {extracted_date}")

        # Return the extracted date, assuming the GPT model returns it in the expected format
        return {"date": extracted_date}

    except Exception as e:
        print(f"Error extracting date: {e}")
        return None


@chatbot.tool
async def filter_experiences(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Filters experiences based on date, location, and discount, retrieving relevant results.
    """
    try:
        print(f"Received user query: {user_query}")
        
        # Step 1: Identify applicable filters
        print("Identifying applicable filters...")
        filters = await analyze_query(user_query)
        print(f"Identified filters: {filters}")

        # Step 2: Extract values for active filters
        date_filter = None
        location_filter = []
        discount_filter = False

        if filters["date"]:
            print("Extracting date filter...")
            date_filter = await extract_date(user_query)  # Returns YYYY-MM-DD or "None"
            print(f"Extracted date filter: {date_filter}")

        if filters["location"]:
            print("Extracting location filter...")
            location_str = await extract_location(user_query)  # Returns comma-separated cities
            location_filter = [city.strip() for city in location_str.split(",") if city.strip()]
            print(f"Extracted location filter: {location_filter}")

        if filters["discount"]:
            print("Extracting discount filter...")
            discount_filter = True
            print(f"Discount filter: {discount_filter}")

        # Step 3: Construct Supabase filtering conditions
        query_params = {}
        print("Constructing query parameters...")

        if date_filter and date_filter != "None":
            print("Adding date filter to query parameters...")
            print(f"Original Date filter: {date_filter}")

            # Ensure date_filter is a dictionary and extract the date value
            if isinstance(date_filter, dict) and "date" in date_filter:
                date_filter = date_filter["date"]  # Extract the actual date string

            # Now, ensure it's a proper string before further processing
            if isinstance(date_filter, str):
                # Remove 'date:' if it exists
                if date_filter.lower().startswith("date:"):
                    date_filter = date_filter[5:].strip()  # Remove "date:" and strip spaces

                # Remove curly braces if they exist
                if date_filter.startswith("{") and date_filter.endswith("}"):
                    date_filter = date_filter[1:-1].strip()  # Remove outer {}

            # Print after modification
            print(f"Processed Date filter (YYYY-MM-DD format): {date_filter}")

            # Now set the query parameter
            query_params["deadline_date_gte"] = date_filter  # Ensures deadline_date > extracted date

            print(f"Added date filter to query parameters: {query_params['deadline_date_gte']}")

        if location_filter:
            query_params["location_filter"] = location_filter  # Filter using multiple cities
            print(f"Added location filter to query parameters: {query_params['location_filter']}")

        if discount_filter:
            query_params["discount_filter"] = True
            print("Added discount filter to query parameters.")

        # Step 4: Query Supabase with improved filtering
        print(f"Querying Supabase with parameters: {query_params}")
        result = ctx.deps.supabase.rpc('filter_experiences', {"filter_data": query_params}).execute()

        if not result.data:
            print("No relevant experiences found.")
            return "No relevant experiences found."

        # Step 5: Format the retrieved experiences
        print("Formatting the retrieved experiences...")
        formatted_experiences = []
        for experience in result.data:
            experience_text = f"""
            **Title**: {experience['title']}
            **Summary**: {experience['summary']}
            **Deadline Date**: {experience.get('deadline_date', 'N/A')}
            **Location**: {experience.get('location', 'N/A')}
            **URL**: {experience['url']}

            """
            formatted_experiences.append(experience_text)

        print(f"Formatted experiences: {formatted_experiences}")
        return "\n\n".join(formatted_experiences)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"
