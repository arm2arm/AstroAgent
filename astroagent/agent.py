"""
Core Agent implementation for AstroAgent.
Handles query processing, tool selection, and response generation.
"""

import json
from typing import List, Dict, Any, Optional
from openai import OpenAI


class AstroAgent:
    """
    An intelligent agent for processing astronomical queries.
    
    This agent uses OpenAI's API to understand natural language queries
    about astronomy and selects appropriate tools to answer them.
    """
    
    def __init__(self, api_key: str, tools: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the AstroAgent.
        
        Args:
            api_key: OpenAI API key
            tools: List of tool definitions in OpenAI function calling format
        """
        self.client = OpenAI(api_key=api_key)
        self.tools = tools or []
        self.tool_functions = {}
        self.conversation_history = []
        
    def register_tool(self, tool_definition: Dict[str, Any], function: callable):
        """
        Register a tool with its implementation function.
        
        Args:
            tool_definition: OpenAI function definition
            function: The actual function to call
        """
        tool_name = tool_definition["function"]["name"]
        self.tools.append(tool_definition)
        self.tool_functions[tool_name] = function
        
    def query(self, user_query: str, max_iterations: int = 5) -> str:
        """
        Process a user query and return a response.
        
        Args:
            user_query: Natural language query about astronomy
            max_iterations: Maximum number of tool calls to prevent infinite loops
            
        Returns:
            String response to the query
        """
        # Add user message to conversation
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })
        
        iterations = 0
        while iterations < max_iterations:
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are AstroAgent, an expert astronomical assistant. 
                        You help users with queries about astronomy, space, celestial objects, 
                        and astronomical events. Use the available tools to provide accurate, 
                        informative responses. When appropriate, include relevant data, dates, 
                        and scientific explanations."""
                    }
                ] + self.conversation_history,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None
            )
            
            response_message = response.choices[0].message
            
            # Check if the model wants to call a function
            if response_message.tool_calls:
                # Add assistant's response to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Call the function
                    if function_name in self.tool_functions:
                        function_response = self.tool_functions[function_name](**function_args)
                        
                        # Add function response to conversation
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(function_response)
                        })
                    else:
                        # Tool not found
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({"error": f"Tool {function_name} not found"})
                        })
                
                iterations += 1
            else:
                # No tool calls, return the response
                final_response = response_message.content or "I couldn't process that query."
                
                # Add assistant's final response to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                
                return final_response
        
        return "I apologize, but I reached the maximum number of tool calls. Please try rephrasing your query."
    
    def reset_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
