import re, json
from scipy.stats import skewnorm
import Levenshtein as levenshtein

def count_tool_call_turns(text):
    # Pattern explanation:
    # assistant   : matches the literal string
    # \s+         : matches one or more whitespace characters (space, tab, newline, etc.)
    # <tool_call> : matches the literal string
    pattern = r'assistant\s+<tool_call>'
    
    # re.findall returns a list of all non-overlapping matches
    matches = re.findall(pattern, text)
    
    return len(matches)

def calculate_skewed_penalty(x, center=70, skewness=5, scale=200):
    """
    Calculate penalty value based on skewed normal distribution.

    Args:
        x: Input value(s) to evaluate
        center: Location parameter (center/mode of distribution)
        skewness: Controls skew direction and magnitude (negative = left skew)
        scale: Controls width of distribution

    Returns:
        y: Penalty value(s) corresponding to input x
    """
    max_y = 0.0036056853513317067  # base on center=70, skewness=5, scale=200
    return skewnorm.pdf(x, a=skewness, loc=center, scale=scale)/max_y

def parse_assistant_thoughts(text):
    """
    Parses content between assistant markers and tool calls.
    Handles the first tool call edge case by checking the start of the string.
    """
    # Regex Breakdown:
    # (?:^|</tool_response>\s*assistant) : Non-capturing group matching start of string 
    #                                       OR the specific response/assistant tail.
    # \s* : Consumes any leading whitespace/newlines.
    # (.*?)                               : Capturing group (non-greedy) for the actual text.
    # \s*<tool_call>                      : Stops at the next tool call tag.
    
    pattern = r'(?:^|</tool_response>\s*assistant)\s*(.*?)\s*<tool_call>'
    
    # re.DOTALL is crucial so the '.' matches newline characters
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    # Cleaning up the results to remove any lingering whitespace
    return [match.strip() for match in matches if match.strip()]


def parse_all_tool_calls(solution_str):
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, solution_str, flags=re.DOTALL)
    matches = [match.strip() for match in matches if match.strip()]
    tool_calls = []
    for match in matches:
        try:
            tool_calls.append(json.loads(match))
        except:
            return []
    return tool_calls

def compute_format_reward(solution_str):
    if "<answer>" not in solution_str and "</answer>" not in solution_str:
        return 0 
    if solution_str.count("<tool_call>") != solution_str.count("</tool_call>") and (solution_str.count("<tool_call>") > 0 or solution_str.count("</tool_call>") > 0):
        return 0.
    
    if solution_str.count("<tool_call>") == 0:
        return 0.
    
    tool_calls = parse_all_tool_calls(solution_str)
    if len(tool_calls) == 0:
        return 0.
    
    processed_tool_calls = []
    for tool_call in tool_calls:
        if "name" not in tool_call or "arguments" not in tool_call:
            return 0.
        if type(tool_call["arguments"]) != dict:
            return 0.
        tool = str(tool_call).lower().replace(" ", "")
        if tool in processed_tool_calls:
            return 0.
        processed_tool_calls.append(tool)

    

    assistant_thoughts = parse_assistant_thoughts(solution_str)
    for i, str1 in enumerate(assistant_thoughts):
        for j, str2 in enumerate(assistant_thoughts):
            if i != j:
                ratio = levenshtein.ratio(str2, str1)
                if ratio > 0.8:
                    return 0.0
                
    length = [len(x.split(" ")) for x in assistant_thoughts]
    if len(length):
        avg_length = float(sum(length))/len(length)
        score = calculate_skewed_penalty(avg_length)
        return score
    else:
        return 0.

def compute_bulk_tool_call_reward(solution_str):
    num_turns = count_tool_call_turns(solution_str)
    num_tools = solution_str.count("<tool_call>")
    if num_turns == 0 :
        return 0.0
    else:
        ratio = float(num_tools)/num_turns
        if ratio > 1:
            return ratio
        else:
            return 0.0