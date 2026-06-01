import requests
import json
from pydantic import BaseModel, Field
from typing import Any
from google import genai
from google.genai import types
import os
import time

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

def get_schedule(week: int, season: int = 2025) -> dict[str, Any]:
    """Fetch all games for a given NFL week."""
    response = requests.get(ESPN_SCOREBOARD, params={
        "dates": season,
        "seasontype": 2,
        "week": week
    })
    response.raise_for_status()
    return response.json()

def get_game_stats(game_id: str) -> dict[str, Any]:
    """Fetch full box score for a specific game."""
    response = requests.get(ESPN_SUMMARY, params={"event": game_id})
    response.raise_for_status()
    return response.json()

def flag_for_review(game_id: str, reason: str, flagged: list) -> None:
    """Log a game the agent isn't confident about."""
    flagged.append({"game_id": game_id, "reason": reason})
    print(f"Flagged game {game_id}: {reason}")

tools = [
    {
        "name": "get_schedule",
        "description": (
            "Fetch the list of NFL games for a specific week of the 2025 season. "
            "Returns game IDs, team names, scores, and game status. "
            "Always call this before get_game_stats to get valid game IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "week": {
                    "type": "integer",
                    "description": "NFL week number (1-18 for regular season)"
                }
            },
            "required": ["week"]
        }
    },
    {
        "name": "get_game_stats",
        "description": (
            "Fetch the full box score and team stats for a specific game. "
            "Requires a valid game ID from get_schedule. "
            "Returns passing, rushing, receiving, and defensive stats for both teams."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "game_id": {
                    "type": "string",
                    "description": "ESPN game ID obtained from get_schedule"
                }
            },
            "required": ["game_id"]
        }
    },
    {
        "name": "flag_for_review",
        "description": (
            "Flag a game that has missing, incomplete, or suspicious data. "
            "Use this instead of skipping silently when something looks wrong."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "game_id": {
                    "type": "string",
                    "description": "ESPN game ID to flag"
                },
                "reason": {
                    "type": "string",
                    "description": "Clear explanation of why this game is being flagged"
                }
            },
            "required": ["game_id", "reason"]
        }
    }
]

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

gemini_tools = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name=tool["name"],
        description=tool["description"],
        parameters=tool["input_schema"]
    )
    for tool in tools
])

def run_agent(goal: str, flagged: list) -> list[dict]:
    """Run the agent until it completes the goal or has no more tool calls to make."""

    messages = [types.Content(role="user", parts=[types.Part(text=goal)])]
    results = []

    while True:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=messages,
            config=types.GenerateContentConfig(tools=[gemini_tools])
        )

        time.sleep(4)  

        model_response_parts = response.candidates[0].content.parts
        messages.append(types.Content(
            role="model",
            parts=model_response_parts
        ))

        if not any(
            hasattr(part, "function_call") and part.function_call is not None
            for part in model_response_parts
        ):
            print("Agent finished.")
            break

        tool_results_for_model = []
        for part in model_response_parts:
            if not (hasattr(part, "function_call") and part.function_call is not None):
                continue

            tool_name = part.function_call.name
            tool_input = dict(part.function_call.args)
            print(f"Agent calling: {tool_name}({tool_input})")

            model_response_content = {"result": ""}

            if tool_name == "get_schedule":
                tool_output_actual = get_schedule(**tool_input)
                games_summary = []
                if 'events' in tool_output_actual:
                    games_summary = [{
                        "id": event["id"],
                        "name": event["name"]
                    } for event in tool_output_actual["events"]]
                model_response_content = {"result": f"Successfully fetched schedule for Week {tool_input['week']}. Found {len(games_summary)} games.", "games_summary": games_summary}

            elif tool_name == "get_game_stats":
                tool_output_actual = get_game_stats(**tool_input)
                results.append(tool_output_actual)
                model_response_content = {"result": f"Game stats for {tool_input['game_id']} fetched and stored."}

            elif tool_name == "flag_for_review":
                flag_for_review(**tool_input, flagged=flagged)
                model_response_content = {"result": f"Game {tool_input['game_id']} flagged for review with reason: {tool_input['reason']}."}

            tool_results_for_model.append(types.Part(
                function_response=types.FunctionResponse(
                    name=tool_name,
                    response=model_response_content
                )
            ))

        messages.append(types.Content(role="user", parts=tool_results_for_model))

    return results

GOAL = """
You are an NFL data collection agent. Your goal is to build a complete game log
dataset for the 2025-26 NFL regular season (weeks 1-18).

For each week:
1. Call get_schedule to get the list of games
2. For each game in that week, call get_game_stats to fetch the box score
3. If a game has missing or zero stats for any team, call flag_for_review
   with a clear reason
4. Move on to the next week only after processing all games in the current week

You must process all 18 weeks. Do not stop early. Do not skip games silently.
If something looks wrong, flag it and move on.
"""

def main():
    flagged = []

    print("Starting NFL data collection agent...")
    results = run_agent(goal=GOAL, flagged=flagged)

    with open("nfl_game_logs.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} game logs.")

    if flagged:
        with open("nfl_flagged_games.json", "w") as f:
            json.dump(flagged, f, indent=2)
        print(f"Flagged {len(flagged)} games for review.")

if __name__ == "__main__":
    main()
