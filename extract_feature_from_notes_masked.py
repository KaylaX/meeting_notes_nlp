import openai
import pandas as pd
import json

data['next_steps'] = data['next_steps'].fillna('').str.strip()

competitor_map = {
    "PRODUCT_1": ["COMPETITOR_1"],
    "PRODUCT_2": ["COMPETITOR_2", "COMPETITOR_3"],
    "PRODUCT_3": ["COMPETITOR_4", "COMPETITOR_5", "COMPETITOR_6"],
    "PRODUCT_4": ["COMPETITOR_7", "COMPETITOR_8"],
    "PRODUCT_5": ["COMPETITOR_9"],
    "PRODUCT_6": ["COMPETITOR_4", "COMPETITOR_5", "COMPETITOR_10"]
}

results = []

for idx, row in data.iterrows():
    if not row['next_steps']:
        continue 

    competitor_prompt = "\n".join([
        f"{product}: {', '.join(competitors)}"
        for product, competitors in competitor_map.items()
    ])

    prompt = f"""
    Analyze the following sales call notes and provide the results in JSON format:
    Notes: "{row['next_steps']}"

    Offerings and their competitors:
    {competitor_prompt}

    Expected JSON output:
    {{
        "mentions": {{
            "PRODUCT_1": true/false,
            "PRODUCT_2": true/false,
            "PRODUCT_3": true/false,
            "PRODUCT_4": true/false,
            "PRODUCT_5": true/false,
            "PRODUCT_6": true/false
        }},
        "competitor_mentions": {{
            "mentioned": true/false,
            "competitors": ["List of competitors mentioned here"],
            "details": "Details about competitors here",
            "contract_end": "Date here, if mentioned"
        }},
        "customer_hesitations": {{
            "layoffs_budget_concern": true/false,
            "details": "Details about layoffs, budget concerns, or other hesitations"
        }}
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that analyzes sales call notes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.5
        )

        results.append({
            "id": idx,
            "analysis": response['choices'][0]['message']['content']
        })

    except Exception as e:
        continue

results_df = pd.DataFrame(results)

def parse_analysis_to_columns(results_df):
    columns = [
        "product_1_mention", "product_2_mention", "product_3_mention",
        "product_4_mention", "product_5_mention", "product_6_mention",
        "competitor_mention", "layoffs_budget_concern"
    ]
    
    for col in columns:
        results_df[col] = 0

    for idx, row in results_df.iterrows():
        try:
            if not row['analysis'] or not isinstance(row['analysis'], str):
                continue

            analysis = json.loads(row['analysis'])

            results_df.at[idx, 'product_1_mention'] = int(analysis["mentions"]["PRODUCT_1"])
            results_df.at[idx, 'product_2_mention'] = int(analysis["mentions"]["PRODUCT_2"])
            results_df.at[idx, 'product_3_mention'] = int(analysis["mentions"]["PRODUCT_3"])
            results_df.at[idx, 'product_4_mention'] = int(analysis["mentions"]["PRODUCT_4"])
            results_df.at[idx, 'product_5_mention'] = int(analysis["mentions"]["PRODUCT_5"])
            results_df.at[idx, 'product_6_mention'] = int(analysis["mentions"]["PRODUCT_6"])

            results_df.at[idx, 'competitor_mention'] = int(analysis["competitor_mentions"]["mentioned"])
            results_df.at[idx, 'layoffs_budget_concern'] = int(analysis["customer_hesitations"]["layoffs_budget_concern"])

        except json.JSONDecodeError as e:
            continue 
        except KeyError as e:
            continue  

    return results_df

results_df = parse_analysis_to_columns(results_df)

results_df.to_csv('masked_results.csv', index=False)
