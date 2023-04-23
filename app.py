from flask import Flask, request, render_template
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

import os
os.environ["OPENAI_API_KEY"] = ""


app = Flask(__name__)

ENV = 'prod'

if ENV == 'dev':
    app.debug = True
else:
    app.debug = False


# set route to the home page using the decorator - @
@app.route('/')

# define a function to render the home page
def index():
    return render_template('index.html')

template = """What are the 3 recommended replies for "{human_input}"?

{history}

Human: {human_input}

Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0.9), 
    prompt=prompt, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=10),
)

@app.route('/reply-recommender', methods=['POST'])
def reply_recommender():
    if request.method == 'POST':
        user_input = request.form['user_input']
        if user_input == '':
            return render_template('index.html', message='Please enter the input')
        response = chatgpt_chain.predict(human_input=user_input)
        return render_template('index.html', output=response)
        

if __name__ == '__main__':
    app.run()
