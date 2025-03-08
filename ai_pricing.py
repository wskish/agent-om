
# updated 2024-10-28

def cost(model, input_tokens, output_tokens):
    # https://openai.com/pricing
    if model in ['gpt-3.5-turbo', 'gpt-3.5']:
        tokens = input_tokens + output_tokens
        dollars = 0.002 * tokens / 1000
    elif model == 'gpt-4':
        dollars =  (.03*input_tokens + .06*output_tokens) / 1000
    elif model in ['gpt-4-1106-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4-turbo']:
        dollars =  (.01*input_tokens + .03*output_tokens) / 1000
    elif model in ['chatgpt-4o-latest', 'gpt-4o-2024-05-13']:
        dollars =  (5*input_tokens + 15*output_tokens) / 1e6
    elif model in ['gpt-4o', 'gpt-4o-2024-08-06']:
        dollars =  (2.5*input_tokens + 10*output_tokens) / 1e6
    elif model in ['gpt-4o-mini', 'gpt-4o-mini-2024-07-18']:
        dollars =  (.15*input_tokens + .6*output_tokens) / 1e6
    elif model in ['o1-preview', 'o1-preview-2024-09-12']:
        dollars =  (15*input_tokens + 60*output_tokens) / 1e6
    elif model in ['o1-mini', 'o1-mini-2024-09-12']:
        dollars =  ( 3*input_tokens + 12*output_tokens) / 1e6

    # https://www.anthropic.com/pricing#anthropic-api        
    elif model in ['claude-3-5-sonnet-20241022', 'claude-3-7-sonnet-20250219']:
        dollars =  (3*input_tokens + 15*output_tokens) / 1e6
    else:
        dollars = -1
    return dollars
