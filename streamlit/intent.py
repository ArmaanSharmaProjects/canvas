
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image
import base64
from io import BytesIO
import os
import google.generativeai as genai
from flask import Flask, request, jsonify
import json # Temp
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app) 

os.environ['GOOGLE_API_KEY'] = "AIzaSyBn1t-SLcH0aQ39OQ5pQeBCe70Shk-yMCM"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')




class InputError(Exception):
    """Exception raised for errors in the input."""
    pass


def base64_to_image(base64_string: str) -> Image.Image:


    if not base64_string:
        raise InputError("Input string is empty")


    try:
        # Remove URL prefix if needed
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',', 1)[1]




        image_data = base64.b64decode(base64_string)
        image_stream = BytesIO(image_data)
        image = Image.open(image_stream)
       
        return image
   
    except base64.binascii.Error:
        raise InputError("Invalid base64-encoded string")
    except IOError:
        raise InputError("Unable to open image from provided data")


class ModelOutput(BaseModel):
    model_number: int = Field(description="The number of the selected model (1-6)")
    source_prompt: Optional[str] = Field(None, description="Source prompt for model 4")
    target_prompt: Optional[str] = Field(None, description="Target prompt for model 4")
    prompt: Optional[str] = Field(None, description="Prompt for models 1-3")


class ModelSelection(BaseModel):
    selected_models: List[ModelOutput] = Field(description="List of 3 selected models with their respective inputs")


parser = PydanticOutputParser(pydantic_object=ModelSelection)


template = '''
You are an AI model selector. Given the following models and their descriptions, select the 6 best prompts for each model for the given task. If an image is provided, consider its content in your selection.


Models:
1. Prompt-based model 1
 - Good for Object swapping, Object Insertion, moderate style changes, property changes, and general image editing features
2. Prompt-based model 2
 - Style Changes that retains image structure, image editing
3. Prompt-based model 3
 - Style Changes that doesn't necessarily retain image structure,  image editing
4. Prompt-based model 4 (requires source and target prompts)
 - Good for quick object swapping, moderate style changes, and property changes of objects
5. Mask-based model 5
 - Proficient at Object Swapping, Object Insertion, and Object Modification
6. Mask-based model 6
 - Proficient at Object Swapping and Object Modification


Guidance for Model Selection:


Model Rankings Based on Tasks
 
    General Image Editing:


        Top Choices:
            Model 1 (Prompt-based model 1)
            Model 4 (Prompt-based model 4, requires both source and target prompts)


        Secondary Choices:
            Model 2 (Prompt-based model 2)
            Model 3 (Prompt-based model 3)
           
    Style-Based Changes to Images:


        Top Choices:
            Model 3 (Prompt-based model 3)
            Model 2 (Prompt-based model 2)


        Secondary Choices:
            Model 1 (Prompt-based model 1)
            Model 4 (Prompt-based model 4, requires both source and target prompts)
       
    Individual Object Editing:


        Object Swapping:
       
            Top Choices:
                Model 5 (Mask-based model 5)
                Model 6 (Mask-based model 6)
           
            Secondary Choices:
                Model 1 (Prompt-based model 1)
                Model 4 (Prompt-based model 4, requires both source and target prompts)
       
        Object Insertion:
           
            Top Choices:
                Model 5 (Mask-based model 5)
                Model 6 (Mask-based model 6)
           
            Secondary Choices:
                Model 1 (Prompt-based model 1)
                Model 4 (Prompt-based model 4, requires both source and target prompts)
           
        Object Modification:
       
            Top Choices (Depends based on circumstance):
                Model 1 (Prompt-based model 1)
                Model 4 (Prompt-based model 4, requires both source and target prompts)
                Model 5 (Mask-based model 5)
                Model 6 (Mask-based model 6)


Rules:
- You are required to select 6 models in total
- For models 1-3, provide a suitable prompt for each model
- For model 4, provide both source (what is currently in the image) and target prompts (what the new image should look like)
- For models 5-6, do not provide any prompts
- For the given task, pick the 6 best models (1 through 6) with abilities that best match the task                
- Do not use Mask Models in your selection if the task instructs otherwise
- Prompt should request a change in the image.
- Prompt for Each model should be high quality and elaborate, but not too deep.  Must have simple to understand language


Your Task:


{task}


Output Format:


{format_instructions}


Output the selected models and their respective inputs.
'''




prompt = PromptTemplate(
    template=template,
    input_variables=["task"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)






def process_image(image_base64=None, task=None):


    image = base64_to_image(image_base64)
    response = model.generate_content([image, prompt.format(task=task)])
    parsed_output = parser.parse(response.text)
    return parsed_output.dict()


def filepath_to_base64(filepath: str) -> str:


    with open(filepath, "rb") as image_file:
        image_data = image_file.read()


    base64_encoded_str = base64.b64encode(image_data).decode('utf-8')


    return base64_encoded_str


#filepath = r"/home/azureuser/Intent_model/summer.jpg"
#base64_str = filepath_to_base64(filepath)
#task = "I want to make the image change seasons from summer to fall"


#try:
    #result = process_image(base64_str, task)
    #json_result = json.dumps(result, indent=2)
    #print(json_result)
#except InputError as e:
    #print(json.dumps({'error': str(e)}, indent=2))
#except Exception as e:
    #print(json.dumps({'error': 'An unexpected error occurred'}, indent=2))

def process_image_and_task(image_path, task):

    image = None

    if image_path is not None:
        image = Image.open(image_path)
        response = model.generate_content([image, prompt.format(task=task)])
    
    else:
        response = model.generate_content([prompt.format(task=task)])
    
    parsed_output = parser.parse(response.text)
    
    return parsed_output

@app.route('/detect_intent', methods=['POST'])
def detect_intent_route():
    if 'image' not in request.files or 'task' not in request.form:
        return jsonify({"error": "Missing image file or task"}), 400

    image_file = request.files['image']
    task = request.form['task']

    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    try:
        result = process_image_and_task(image_path=temp_image_path, task=task)
        
        result_dict = result.dict()

        return jsonify(result_dict), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7999)