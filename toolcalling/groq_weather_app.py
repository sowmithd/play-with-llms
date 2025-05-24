# # Function Calling with OpenAI APIs

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)


# 1. Define Function to fetch context

# Get the current weather
def get_current_weather(location):
    """Get the current weather in a given location"""
    # First get coordinates using Geocoding API
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    
    # Geocoding API to get lat/lon
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
        geo_response = requests.get(geo_url)
        geo_data = geo_response.json()
        
 
        # If still no results, return error
        if not geo_data:
            return json.dumps({"error": f"Location '{location}' not found"})
        
        # Extract coordinates
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        
        # Use the free Current Weather API endpoint
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        
        # Check if we got a valid response
        if weather_data.get('cod') != 200:
            error_message = weather_data.get('message', 'Unknown error')
            print(f"API Error: {error_message}")
            return json.dumps({
                "error": "Weather API error",
                "message": error_message,
                "status_code": weather_data.get('cod')
            })
        
        # Extract relevant information
        weather = {
            "location": location,
            "current": {
                "temperature": weather_data["main"]["temp"],
                "feels_like": weather_data["main"]["feels_like"],
                "humidity": weather_data["main"]["humidity"],
                "wind_speed": weather_data["wind"]["speed"],
                "weather_description": weather_data["weather"][0]["description"]
            }
        }
        
        return json.dumps(weather)
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {str(e)}")
        return json.dumps({"error": "Network error", "message": str(e)})
    except KeyError as e:
        print(f"Data error: {str(e)}")
        return json.dumps({"error": "Data parsing error", "message": f"Missing field: {str(e)}"})
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return json.dumps({"error": "Unexpected error", "message": str(e)})


# ### Define Functions
# 
# As demonstrated in the OpenAI documentation, here is a simple example of how to define the functions that are going to be part of the request. 
# 
# The descriptions are important because these are passed directly to the LLM and the LLM will use the description to determine whether to use the functions or how to use/call.


# Define a function for LLM to use as a tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },   
    }
]

# Only run this section when the script is executed directly
if __name__ == "__main__":
    print("Weather Information Assistant")
    print("-----------------------------")
    print("Ask about weather in any location, or type 'exit' to quit.\n")
    
    while True:
        user_query = input("Your question: ")
        
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        try:
            # Clear system prompt to guide the LLM's behavior
            system_prompt = """You are a helpful weather assistant. 
            - ONLY answer questions related to current weather.
            - If the user asks about something unrelated to weather, respond with: "I can only answer questions about current weather. Please ask me about the weather in a specific location."
            - If there's an error fetching weather data, clearly communicate that to the user without making up information.
            - Never hallucinate or make up weather information.
            - Don't provide forecasts unless that data is explicitly available.
            """
            
            # First response to determine if it's a weather question and call the function
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0,
                max_tokens=300,
                tools=tools,
                tool_choice="auto"  # Let the model decide whether to use the tool
            )
            
            groq_response = response.choices[0].message
            print("\nProcessing your question...")
            
            # Check if the model decided to use the tool
            if groq_response.tool_calls:
                # Extract the location from the function call
                tool_call = groq_response.tool_calls[0]
                if tool_call.function.name == "get_current_weather":
                    args = json.loads(tool_call.function.arguments)
                    location = args.get("location", "")
                    print(f"Getting weather for: {location}")
                    
                    # Call the weather function
                    weather_data = get_current_weather(**args)
                    
                    # Check if there was an error
                    weather_json = json.loads(weather_data)
                    if "error" in weather_json:
                        error_info = f"Error: {weather_json.get('message', 'Unknown error')}"
                        print(error_info)
                    
                    # Send the result back to the LLM
                    second_response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": groq_response.tool_calls
                            },
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": weather_data
                            }
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    
                    # Print the final response
                    print("\nAnswer:")
                    print(second_response.choices[0].message.content)
                else:
                    # Unknown tool call
                    print("\nAnswer:")
                    print("I'm not sure how to handle that request. Please ask about the weather in a specific location.")
            else:
                # Model chose not to use the tool - likely not a weather question
                print("\nAnswer:")
                print(groq_response.content)
        
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again with a different question.")
        
        print("\n" + "-" * 50 + "\n")
