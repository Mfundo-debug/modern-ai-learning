from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

# --------------------------------------------------
# STEP 1: DEFINE THE REQUIRED OUTPUT STRUCTURE
# --------------------------------------------------

class TopicExplanation(BaseModel):
    topic: str
    definition: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    keywords: list[str]


# --------------------------------------------------
# STEP 2: CREATE THE API CLIENT
# --------------------------------------------------

client = OpenAI()


# --------------------------------------------------
# STEP 3: GET A TOPIC FROM THE USER
# --------------------------------------------------

topic = input("Enter an AI topic: ")


# --------------------------------------------------
# STEP 4: SEND THE REQUEST
# --------------------------------------------------

try:

    response = client.responses.parse(

        model="gpt-5.6",

        input=[
            {
                "role": "developer",
                "content": """
                You are a technical AI tutor.

                The learner understands Python and basic
                machine learning.

                Explain the requested AI concept accurately.

                The difficulty must be one of:
                beginner, intermediate, or advanced.

                Include between 3 and 6 important keywords.
                """
            },

            {
                "role": "user",
                "content": f"Explain this AI concept: {topic}"
            }
        ],

        text_format=TopicExplanation
    )


    # --------------------------------------------------
    # STEP 5: ACCESS THE STRUCTURED RESULT
    # --------------------------------------------------

    result = response.output_parsed


    # --------------------------------------------------
    # STEP 6: DISPLAY INDIVIDUAL FIELDS
    # --------------------------------------------------

    print("\nTOPIC:")
    print(result.topic)

    print("\nDIFFICULTY:")
    print(result.difficulty)

    print("\nDEFINITION:")
    print(result.definition)

    print("\nKEYWORDS:")

    for keyword in result.keywords:
        print("-", keyword)


except Exception as error:

    print("\nThe request could not be completed.")
    print(error)
