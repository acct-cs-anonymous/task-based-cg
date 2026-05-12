"""Store all constants and configuration defaults"""
VARIABLE_MAX_PROMPT_LENGTHS = {
    "direct": 40,
    "curriculum": 40,
    "step_by_step": 102
}

SPECIAL_TOKENS = {
    "START": "<START>",
    "SPACE": " ",
    "SEP": "<SEP>",
    "NULL": "<NULL>",
    "END": "<END>"
}
