from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


class VLLM:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1))
    def prompt_text(self, text_prompt):
        kwargs = dict(
            model=self.args.gen_model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text_prompt},
            ],
            temperature=0.0,
            top_p=0.95,
            max_completion_tokens=self.args.gen_max_new_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response = self.client.chat.completions.create(**kwargs)
        generation = response.choices[0].message.content
        return generation
