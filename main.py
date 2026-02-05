#!/usr/bin/env python3
"""
Main CLI interface for AstroAgent.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from astroagent import AstroAgent, get_available_tools


def main():
    """Main entry point for the AstroAgent CLI."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="AstroAgent - An Agentic System for Astronomical Queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What is the moon phase today?"
  python main.py "Show me today's NASA picture"
  python main.py "Where is Mars visible from New York?"
  python main.py -i  # Interactive mode
        """
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Astronomical query to process"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive mode"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (can also be set via OPENAI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided.")
        print("Set OPENAI_API_KEY environment variable or use --api-key option.")
        sys.exit(1)
    
    # Initialize console for rich output
    console = Console()
    
    # Initialize agent
    tool_definitions, tool_functions = get_available_tools()
    agent = AstroAgent(api_key=api_key, tools=tool_definitions)
    
    # Register all tools
    for tool_def in tool_definitions:
        tool_name = tool_def["function"]["name"]
        agent.register_tool(tool_def, tool_functions[tool_name])
    
    # Display welcome message
    console.print(Panel.fit(
        "[bold cyan]🌟 AstroAgent[/bold cyan]\n"
        "Your Intelligent Assistant for Astronomical Queries",
        border_style="cyan"
    ))
    
    if args.interactive:
        # Interactive mode
        console.print("\n[yellow]Interactive mode - Type 'exit' or 'quit' to end[/yellow]\n")
        
        while True:
            try:
                query = console.input("[bold green]You:[/bold green] ")
                
                if query.lower() in ["exit", "quit", "q"]:
                    console.print("\n[cyan]Thank you for using AstroAgent! 🚀[/cyan]")
                    break
                
                if not query.strip():
                    continue
                
                # Process query
                console.print("\n[bold blue]AstroAgent:[/bold blue]", end=" ")
                response = agent.query(query)
                
                # Display response as markdown
                console.print(Markdown(response))
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n\n[cyan]Thank you for using AstroAgent! 🚀[/cyan]")
                break
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {str(e)}\n")
    
    elif args.query:
        # Single query mode
        console.print(f"\n[bold green]Query:[/bold green] {args.query}\n")
        console.print("[bold blue]AstroAgent:[/bold blue]", end=" ")
        
        try:
            response = agent.query(args.query)
            console.print(Markdown(response))
            console.print()
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}\n")
            sys.exit(1)
    
    else:
        # No query provided
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
