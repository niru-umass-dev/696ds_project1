import os
import openai

openai.organization = "org-uqW82WXjsx1QYRwThjvVeQqQ"
# openai.api_key = "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"

openai.api_key = os.getenv("OPENAI_API_KEY")
# print(openai.Model.list())

# input_sentence = "The hotel has an enjoyable ambience and is great value for money. This location is great if you're looking to access the great fun and action close-by, but with that comes hustle-and-bustle noise from the street the hotel is located on. The rustic rooms here are a little small but really comfortable and clean while the beds are big. There was a thatched roof to the room that provided an airy feel but of course not soundproof. The hotel provides great breakfast to be eaten above the garden or to take away. The hotel does not provide private parking unfortunately."

def get_paraphrase(input_sentence: str):
    prompt = f"Paraphrase this: f'{input_sentence}'"
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      max_tokens=512,
    )
    return response['choices'][0]['text']