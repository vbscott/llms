from vertexai.language_models import TextGenerationModel
import vertexai


def interview(
    temperature: float,
    location: str,
    question: str,
) -> str:
    vertexai.init(location=location)
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 100,  # Token limit determines the maximum amount of text output.
        "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@002")
    response = model.predict(
        question,
        **parameters,
    )
    print(f"Response from Model: {response}")

    return response.text


done = False
while(not done):
    question = input("\n")
    if question == "done":
        done = True
        continue
    interview(0.9, "us-central1", question)
    print("\n")
