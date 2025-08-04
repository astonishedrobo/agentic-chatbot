import os
import json
import yaml
import asyncio
import argparse
from pathlib import Path
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.console import Group

from mem0 import Memory
from agent.graph import AgenticChatBot
from dotenv import load_dotenv
import docker

# Configuration
CONFIG_DIR = Path.home() / ".agentic_chatbot"
CONFIG_FILE = CONFIG_DIR / "config.json"
load_dotenv(".env")

COMMANDS = {
    "/exit": "Exit the chatbot session.",
    "/new": "Start a new, fresh conversation.",
    "/clear": "Clear the long-term memory.",
    "/commands": "Display this list of available commands."
}

def handle_user_login() -> str:
    parser = argparse.ArgumentParser(description="Agentic Chatbot CLI")
    parser.add_argument(
        "--relogin",
        action="store_true",
        help="Force re-login to switch to a different user.",
    )
    args = parser.parse_args()
    console = Console()
    if args.relogin or not CONFIG_FILE.exists():
        console.print(Panel("[bold yellow]Welcome! Let's set up your user profile.[/bold yellow]", border_style="yellow"))
        user_id = Prompt.ask("[cyan]Please enter a username[/cyan]")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump({"user_id": user_id}, f)
        console.print(Panel(f"[bold green]✓ Welcome, {user_id}! Your profile is saved.[/bold green]", border_style="green"))
        return user_id
    else:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            user_id = config["user_id"]
        console.print(Panel(f"[bold green]✓ Welcome back, {user_id}![/bold green]", border_style="green"))
        return user_id

def initialize_bot(user_id: str, console: Console) -> AgenticChatBot:
    with console.status("[bold blue]Warming up the engines...[/bold blue]", spinner="dots"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        mem_config = {
            "vector_store": {
                "provider": config["vector_store"]["provider"],
                "config": {
                    "host": config["vector_store"]["config"]["host"],
                    "port": config["vector_store"]["config"]["ports"][0],
                }
            },
            "graph_store": {
                "provider": config["graph_store"]["provider"],
                "config": {
                    "url": config["graph_store"]["config"]["url"],
                    "username": config["graph_store"]["config"]["username"],
                    "password": config["graph_store"]["config"]["password"]
                }
            }
        }
        mem = Memory.from_config(mem_config)
        console.log("Long-Term Memory initialized.")
        
        bot = AgenticChatBot(user_id=user_id, lt_mem=mem, config=config)
        console.log("Agentic Chatbot is ready.")
    return bot

async def chat_loop(bot: AgenticChatBot, console: Console):
    """
    The main interactive chat loop with an aesthetic UI and new command system.
    """
    header_text = f"Logged in as: [b]{bot._user_id}[/b] | Thread: [b]{bot._thread_id}[/b] | Type [b]/commands[/b] for help"
    last_response_panel = Panel("[bold cyan]Chat session started. Let's talk![/bold cyan]", border_style="cyan")

    while True:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            console.print(Panel(header_text, style="dim", border_style="dim"))
            console.print(last_response_panel)

            user_input = console.input("\n[bold green]>[/bold green] ")

            # Command Handling
            if user_input.lower() in COMMANDS:
                command = user_input.lower()
                if command == "/exit":
                    break
            
                if command == "/new":
                    bot.start_new_conversation()
                    header_text = f"Logged in as: [b]{bot._user_id}[/b] | Thread: [b]{bot._thread_id}[/b] | Type [b]/commands[/b] for help"
                    console.print(Panel(header_text, style="dim", border_style="dim"))
                    last_response_panel = Panel("[bold yellow]New conversation started.[/bold yellow]", border_style="yellow")
                    continue

                if command == "/clear":
                    bot.clear_memory()
                    last_response_panel = Panel("[bold green]Long-term memory cleared![/bold green]", border_style="green")
                    continue

                if command == "/commands":
                    # Build a single string containing all the markup
                    command_markup = "\n"
                    for cmd, desc in COMMANDS.items():
                        command_markup += f"  [bold cyan]{cmd}[/bold cyan]: {desc}\n"
                    
                    # Pass the complete string to the Panel for correct rendering
                    last_response_panel = Panel(
                        command_markup, 
                        title="[bold]Available Commands[/bold]", 
                        border_style="blue"
                    )
                    continue
            
            # Chatbot Interaction
            with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                response = await bot.run(user_input)
                bot_message = response['messages'][-1].content
            
            user_panel = Panel(
                user_input, 
                title="[bold green]User[/bold green]", 
                border_style="green",
                expand=True
            )
            bot_panel = Panel(
                Markdown(bot_message),
                title="[bold blue]Assistant[/bold blue]",
                border_style="blue",
                expand=True
            )

            last_response_panel = Group(user_panel, bot_panel)


        except KeyboardInterrupt:
            break
        except Exception as e:
            last_response_panel = Panel(f"[bold red]An error occurred: {e}[/bold red]", border_style="red")

def initialize_containers():
    """
    Initialize Docker containers for Neo4j and Qdrant containers.
    """
    client = docker.from_env()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Start Neo4j container
    try:
        neo4j_container = client.containers.run(
            image=config["graph_store"]["config"]["image"],
            detach=True,
            ports={'7474/tcp': config["graph_store"]["config"]["ports"][0], '7687/tcp': config["graph_store"]["config"]["ports"][1]},
            environment=[
                f"NEO4J_AUTH={config['graph_store']['config']['username']}/{config['graph_store']['config']['password']}"
            ],
            volumes={os.path.join(os.getcwd(), "neo4j", "data"): {'bind': '/data', 'mode': 'rw'}}
        )
    except Exception as e:
        print(f"Error starting Neo4j container: {e}")

    # Start Qdrant container
    try:
        qdrant_container = client.containers.run(
            image=config["vector_store"]["config"]["image"],
            detach=True,
            ports={'6333/tcp': config["vector_store"]["config"]["ports"][0], '6334/tcp': config["vector_store"]["config"]["ports"][1]},
            volumes={os.path.join(os.getcwd(), "qdrant_storage"): {'bind': '/qdrant/storage', 'mode': 'rw'}}
        )
    except Exception as e:
        print(f"Error starting Qdrant container: {e}")

    time.sleep(10)
    return [qdrant_container, neo4j_container]


if __name__ == "__main__":
    console = Console()
    with console.status("[bold blue]Initializing Database...[/bold blue]", spinner="dots"):
        containers = initialize_containers()

    try:
        user_id = handle_user_login()
        bot = initialize_bot(user_id, console)
        asyncio.run(chat_loop(bot, console))
    finally:
        console.print(Panel("[bold magenta]Goodbye![/bold magenta]", border_style="magenta"))
        for container in containers:
            try:
                container.stop()
                container.remove()
                console.log(f"Stopped and removed container: [bold green]{container.image.tags[0]}[/bold green]")
            except Exception as e:
                console.log(f"Error stopping/removing container {container.image.id}: {e}")