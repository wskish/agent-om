
# updated 2025-03-08

def cost(model, input_tokens, output_tokens):
    # # https://platform.openai.com/docs/models#model-endpoint-compatibility
    if model in ['gpt-4o', 'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20']:
        dollars =  (2.5*input_tokens + 10*output_tokens) / 1e6
    elif model in ['gpt-4o-mini', 'gpt-4o-mini-2024-07-18']:
        dollars =  (.15*input_tokens + .6*output_tokens) / 1e6
    elif model in ['o3-mini', 'o3-mini-2025-01-31']:
        dollars = (1.1*input_tokens + 4.4*output_tokens) / 1e6
    elif model in ['o1', 'o1-2024-12-17']:
        dollars = (15*input_tokens + 60*output_tokens) / 1e6
    # https://www.anthropic.com/pricing#anthropic-api        
    elif model in ['claude-3-5-sonnet-20241022', 'claude-3-7-sonnet-20250219']:
        dollars =  (3*input_tokens + 15*output_tokens) / 1e6
    else:
        dollars = -1
    return dollars
