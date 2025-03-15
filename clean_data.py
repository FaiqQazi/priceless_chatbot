# import os

# # Define the directory and text to remove
# directory = "scraped_experiences"
# unwanted_text = """All Experiences
# Entertainment
# Arts and Culture
# Sports
# Culinary
# Travel
# More...
# Shopping
# Health and Wellness
# Less...
# All benefits
# Travel benefits
# Hotel benefits
# Lifestyle benefits
# Shopping benefits
# More...
# Insurance benefits
# Less...
# Argentina
# Australia
# Brazil
# Canada
# Chile
# More...
# Colombia
# Croatia
# Czechia
# France
# Germany
# Greece
# Hong Kong
# India
# Indonesia
# Ireland
# Italy
# Kenya
# Mexico
# Netherlands
# Philippines
# Singapore
# South Africa
# Spain
# Sweden
# Thailand
# Turkey
# United Arab Emirates
# United Kingdom
# United States
# Clear Selection
# All Experiences
# Entertainment
# Arts and Culture
# Sports
# Culinary
# Travel
# More...
# Shopping
# Health and Wellness
# Less...
# All benefits
# Travel benefits
# Hotel benefits
# Lifestyle benefits
# Shopping benefits
# More...
# Insurance benefits
# Less...
# Argentina
# Australia
# Brazil
# Canada
# Chile
# More...
# Colombia
# Croatia
# Czechia
# France
# Germany
# Greece
# Hong Kong
# India
# Indonesia
# Ireland
# Italy
# Kenya
# Mexico
# Netherlands
# Philippines
# Singapore
# South Africa
# Spain
# Sweden
# Thailand
# Turkey
# United Arab Emirates
# United Kingdom
# United States
# Clear Selection
# Questions?
# Call
# Find tickets
# Monday through Friday, 9am–6pm PT
# Send a Message
# Thank you for your message
# So glad you like this! If this item becomes available again, we will be sure to let you know.
# So we are sure to reach you, please confirm your email address below:
# How to buy gift
# Item added to cart. Go to cart now"""

# for filename in os.listdir(directory):
#     file_path = os.path.join(directory, filename)
    
#     # Read file content
#     with open(file_path, "r", encoding="utf-8") as file:
#         content = file.read()
    
#     # Remove unwanted text
#     cleaned_content = content.replace(unwanted_text, "").strip()
    
#     # Write cleaned content back to the file
#     with open(file_path, "w", encoding="utf-8") as file:
#         file.write(cleaned_content)

# print("Cleaning complete.")


import os

# Define the directory
directory = "scraped_experiences"

# Define the unwanted block as a single string
unwanted_block = """Email
Phone
(opens in new tab)
(opens in new tab)
Contact us
(opens in new tab)
Terms of Use
(opens in new tab)
About priceless™
(opens in new tab)
Privacy Notice
(opens in new tab)
Mastercard.com"""

# Loop through each file in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    # Read file content
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Find the first occurrence of the unwanted block
    removal_index = content.find(unwanted_block)

    # If we found the unwanted block, clean the content
    if removal_index != -1:
        # Keep only the content before the unwanted block
        cleaned_content = content[:removal_index].strip()
    else:
        cleaned_content = content.strip()  # If no unwanted block, keep the content as is

    # Write cleaned content back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(cleaned_content)

print("Cleaning complete.")
