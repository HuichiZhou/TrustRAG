from openai import OpenAI
import os

API_KEY = os.environ.get("OPENAI_API_KEY")
assert API_KEY is not None, "OPENAI_API_KEY is not set"


class GPT:
    def __init__(self, model_name):
        self.max_output_tokens = 1024
        self.client = OpenAI(api_key=API_KEY)
        self.model_name = model_name

    def query(self, msg, temperature=0.1):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg},
                ],
            )
            response = completion.choices[0].message.content

        except Exception as e:
            print(e)
            response = ""

        return response
