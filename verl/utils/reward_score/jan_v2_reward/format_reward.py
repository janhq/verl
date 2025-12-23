import re
from scipy.stats import skewnorm


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


def compute_format_reward(solution_str):
    if "<answer>" not in solution_str and "</answer>" not in solution_str:
        return 0 
    if solution_str.count("<tool_call>") != solution_str.count("</tool_call>") and (solution_str.count("<tool_call>") > 0 or solution_str.count("</tool_call>") > 0):
        return 0.
    
    if solution_str.count("<tool_call>") == 0:
        return 0.
    
    assistant_thoughts = parse_assistant_thoughts(solution_str)
    length = [len(x.split(" ")) for x in assistant_thoughts]
    if len(length):
        avg_length = float(sum(length))/len(length)
        score = calculate_skewed_penalty(avg_length)
        return score
    else:
        return 0.

