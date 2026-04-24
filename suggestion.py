from groq import Groq
from model import flag

# Initialize Groq client
client = Groq(api_key="")
#print("flag:",len(flag))

def get_suggestion(flag):
    if len(flag)>0:
        word_list_str = "\n".join(f"{i+1}. {word}" for i, word in enumerate(flag))

        # Prompt for Groq
        prompt = f"""
        Words to analyze:
        {word_list_str}

        Based on the words, please provide:
        - Additional Words for Practice with similar pronounciation of the word list (with numbering starting from 1)
        - Suggested Practice Exercises (also numbered from 1)

        Do NOT include Weaknesses or Practice Tips or Word breakdowns.
        Do NOT include any other messages other than headings.

        Please ensure each section has its own numbering starting from 1, without continuing from the previous section.
        """

        # Get response from Groq
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an expert helping dyslexia students with pronunciation. Explain as a guide to dyslexics"},
                {"role": "user", "content": prompt}
            ]
        )
        # Full text response from Groq
        return response.choices[0].message.content
    else:
        return "No mispronounced word"



import re

def clean_groq_output(output):
    # Remove spaces around headers (like the "Additional Words for Practice")
    output = re.sub(r'\n\s*\n', '\n', output)  # Remove multiple newlines
    output = re.sub(r'(\*\*|__)(.*?)\1', r'\2', output)  # Remove ** for bolding
    output = re.sub(r'(Suggested Practice Exercises)', r'\n\n\1', output)
    return output
