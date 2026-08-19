from openai import OpenAI

client = OpenAI()

question = input("Ask the AI a question: ")

try:
    response = client.responses.create(
        model="gpt-5.6",
        instructions="""
        You are an AI tutor.
        Explain technical concepts clearly.
        Assume the learner understands Python and basic machine learning.
        Give concise but technically accurate answers.
        """,
        input=question
    )

    print("\nAI RESPONSE:\n")
    print(response.output_text)

except Exception as error:
    print("\nThe request could not be completed.")
    print(error)
