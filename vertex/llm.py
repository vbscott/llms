import sys
from vertexai.language_models import TextGenerationModel
import vertexai

LOCATION = "us-central1"
EXIT_OPTIONS = ("done", "quit", "exit", "d", "q", "e")

class Class:
    """
    Object for the class taught.
    """
    def generate_grade(self, grade: str) -> str:
        """
        Takes in a grade and makes sure it's in an AI friendly format.
        Ex: 1 to 1st grade, 2 to 2nd grade, etc
        """
        grade_dict = {"1": "1st", "2": "2nd", "3": "3rd"}
        if grade in grade_dict:
            return grade_dict[grade]
        if grade.isnumeric():
            return grade + "th"
        return grade

    def save_to_file(self, info):
        """
        Saves info to filename given during class creation
        """
        if self.filename != "":
            f = open(self.filename, "a", encoding="utf-8")
            f.write(info)
            f.write("--------------------------------------------")
            f.close()


    def __init__(self, topic="", filename=""):
        if topic != "unknown":
            self.topic = input("What is the topic you would like to talk about?\n")
        else:
            self.subject = input("What is the subject of the class?\n")
        self.grade = self.generate_grade(input("What grade are you teaching?\n"))
        self.length = input("How long is your class in minutes?\n")
        self.filename = filename


def vertex_call(
        question: str,
) -> str:
    """
    Makes an API call to Google's Vertex endpoint
    """
    vertexai.init(location = 0.0)
    parameters = {
        "max_output_tokens": 400,
        "top_k": 1,
    }

    model = TextGenerationModel.from_pretrained("text-bison@002")
    return model.predict(
        question,
        **parameters,
    )


def generate_activity(c: Class):
    """
    Creates a well formed question asking to generate an activity from the LLM
    """
    question = f"Can you give me an activity for {c.topic} for {c.grade} grade?"
    response = vertex_call(question)
    print(response.text)
    c.save_to_file(response.text)

def activity_from_prompt(c: Class, prompt: str):
    """
    Takes in a prompt from another LLM response, parses it out
    then generates a question to generate an activity based on
    what the user wants to do.
    """
    question = input("Do you want to generate an activity based on the previous response?\n")
    while not question.lower() in EXIT_OPTIONS and "n" not in question.lower():
        question = input("Which topic?\n")
        if not question.isnumeric():
            question = input("Please give the number of the topic.\n")

        # Splits the prompt into lines to find the activity
        # the user would like to generate an activity for
        prompt_split = prompt.split("\n")

        for i in prompt_split:
            if question in i:
                # Each line is in the format 'Topic: Explination',
                # this parses it to just ask the topic
                question_split = i.split(":")
                choice = question_split[0]
                response = vertex_call(f"Activity for {choice} at the {c.grade} level")
                print(response.text)
                c.save_to_file(response.text)
                c.topic = choice
                break

        question = input("Do you want to create another activity?\n")

def generate_topics(c: Class):
    """
    Create and send a topics generation question. Then generate activities
    one or more of the responses if the user chooses to
    """
    question = f"Can you get me 4 topics for {c.subject} topic for {c.grade} grade?"
    response = vertex_call(question)

    print(response.text)
    c.save_to_file(response.text)
    activity_from_prompt(c, response.text)

def lesson_plan(c: Class):
    """
    Create the day's lesson plan based on the topic, grade, and length of the class
    """
    response = vertex_call(f"Can you generate a daily lesson plan about\
{c.topic} for {c.grade} that lasts {c.length} minutes?")

    print(response.text)
    c.save_to_file(response.text)

def user_input(filename=""):
    """
    Gathers user input and generates LLM qustions based on it.
    """
    question_prompt = "Which do you need?\n1) new topic\n\
2) lesson plan\n3) activity\n4) change class\n5) exit\n"
    question = input(question_prompt)

    # If there isn't a topic yet, set it to unknown, otherwise wait
    # to get user input
    topic = "unknown" if question.lower() in ("1", "topic") else ""
    if question in EXIT_OPTIONS or question == "5":
        return

    c = Class(topic=topic, filename=filename)
    while not question.lower() in EXIT_OPTIONS:
        if question.lower() in ("1", "topic"):
            generate_topics(c)
        elif question.lower() in ("2", "lesson"):
            lesson_plan(c)
        elif question.lower() in ("3", "activity"):
            generate_activity(c)
        elif question.lower() in ("4", "new", "class"):
            c = Class(filename=filename)
        elif question == "5":
            break
        else:
            print("Not one of the responses\n")
        question = input("Do you need something else?\n")
        if question.lower() in EXIT_OPTIONS or question == "5":
            break
        question = input(question_prompt)

def main():
    """
    Main function
    """
    filename = ""
    if len(sys.argv) >= 2:
        arg = sys.argv[1]

        if arg in ("-h", "--help"):
            print("This is a program to help you generate ideas for the classroom.\n\
You can run it as is or add -f/--file flag to save the output to the file specified.\n\
You can quit anytime by typing in 'quit', 'q', 'exit', or 'e'")
            sys.exit()
        elif arg in ("-f", "--file"):
            if len(arg) >= 2:
                filename = sys.argv[2]
            else:
                print("Missing file name")
        else:
            print(f"Invalid response: {arg}")
            sys.exit(1)
    user_input(filename)



if __name__ == "__main__":
    main()
