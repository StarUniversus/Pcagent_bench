import math
import torch
import random
import numpy as np
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
from openai import OpenAI
from pydantic import BaseModel as PdBaseModel
env_token = os.environ.get('OPENAI_API_KEY', '')
class CorrectFlow(BaseModel):
    is_api: bool = True
    def __init__(self,
                 model:str = "gpt-4o-2024-08-06",
                 api_base:str = "https://api.openai.com/v1/chat/completions",
                 token:str = env_token,
                 max_tokens:int = 2048,
                 temperature:float = 0.0,
                 seed:int = 1234,
                 **kwargs):
        self.kwargs = kwargs
        self.API_url = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.model = model
        # Your gpt-4o-mini-2024-07-18 API Token
        #token = "sk-XETSDvacdRFYYgQTB0AcA3Fa145f48Ab8b0d1aC2A8BfD06d"
        self.token = token
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def inference_chat(self, chat, model, api_url, token):    
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        data = {
            "model": model,
            "messages": [],
            "max_tokens": self.max_tokens,
            'temperature': self.temperature,
            "seed": self.seed,
        }
        #"seed": 1234         'top_p':1.0,

        for role, content in chat:
            data["messages"].append({"role": role, "content": content})

        while True:
            try:
                res = requests.post(api_url, headers=headers, json=data)
                res_json = res.json()
                res_content = res_json['choices'][0]['message']['content']
            except Exception as e: 
                print(e)
                try:
                    print(res.json())
                except:
                    print("Request Failed")
                continue
        # else:
        #     break
        
            return res_content
    def COT_instruction(self, prompt):
        client = OpenAI(base_url="https://api.openai.com/v1/", api_key=self.token)
        class ActionReasoning(PdBaseModel):
            title: str
            content: str
            next_action: str

        completion = client.beta.chat.completions.parse(
            temperature=0.0,
            model="gpt-4o-2024-08-06",
            messages=prompt,
            response_format=ActionReasoning,
        )

        action_reason = completion.choices[0].message
        return action_reason
    
    def COT_action_squences(self, prompt):
        client = OpenAI(base_url="https://api.openai.com/v1/", api_key=self.token)
        class ActionReasoning(PdBaseModel):
            content: str
            title: str
            next_action: str
            
        completion = client.beta.chat.completions.parse(
            temperature=0.0,
            model="gpt-4o-2024-08-06",
            messages=prompt,
            response_format=ActionReasoning
        )

        action_reason = completion.choices[0].message
        return action_reason
    def generate_img_text(self, role, prompt, image):
        base64_image = self.encode_image(image)
        content = [
            {
                "type": "text", 
                "text": prompt
            },
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
        ]
        return [role, content]
    
    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        base64_image = self.encode_image(image_path)
        
        user_instruction = prompt

        query = (
            "### User's Question: ### "+user_instruction+'\n'
            "### Your task ###: Use the user's question to generate knowledge relevant to the objects in the question. \n"
            "Only output the knowledge. \n Your answer:"
        )
        feedback = [["system", [{"type": "text", "text": query}]]]
        # feedback = generate_img_text("user", query, image_path)
        text_analysis = self.inference_chat(feedback, 'gpt-4o-2024-08-06', self.API_url, self.token)+'\n'
    
        query = (
            "### User's Question: ### "+user_instruction+'\n'
            "### Your task ###: Provide a precise, objective description of the image content. Then, extract relevant knowledge related to the objects in the image. Do not provide any conclusions or assumptions. Only describe and relay knowledge objectively.\n"
            "\n Your answer:"
        )
        # feedback = [["system", [{"type": "text", "text": query}]]]
        feedback = self.generate_img_text("user", query, image_path)
        text_analysis += self.inference_chat([feedback], 'gpt-4o-2024-08-06', self.API_url, self.token)
        
        ########### role generation ############
        role_split_query = (
            "### User's Question: ### "+user_instruction+'\n'
            "### You are an expert in assigning roles depending on required skills and attributes. "+'\n'
            "### Your task ###: Based on the User's Question and the reference image, generate the most suitable role to solve the problem. Provide only the role name. \n"
            "Only output the role name. \n Your answer:"
            # "### output format ###: Each role name is split using '##########'. \n"
            # "Example 1: [meticulous observer ########## systems manipulation expert ########## ]. \n" There are some role sequences, and you must refer to the following output format:
            # "Example 2: [Google Chrome]. \n" 
        )
        # feedback = [["system", [{"type": "text", "text": role_split_query}]]]
        feedback = self.generate_img_text("user", role_split_query, image_path)
        role_name = self.inference_chat([feedback], 'gpt-4o-2024-08-06', self.API_url, self.token)
        role_list = ['Junior '+role_name, 'Mid-Level '+role_name, 'Senior '+ role_name]
        # print(role_name)
        
        ########### role level ############
        # query = (
        #     "Based on the required skills and attributes, assign three appropriate level for a" + role_name +  " role. \n"
        #     "**You are an expert in assigning roles based on experience, expertise, and responsibilities.** \n"
        #     "**Your Task:** Based on the user's requirements and context (skills, experience, responsibilities), assign the most suitable level of the " + role_name + " role. \n"
        #     "Provide only the role name."
        #     "### output format ###: Each role name is split using '##########'. \n"
        #     "**Your Answer:**"
        #     # "Example 1: [meticulous observer ########## systems manipulation expert ########## ]. \n" There are some role sequences, and you must refer to the following output format:
        #     # "Example 2: [Google Chrome]. \n" 
        # )
        # feedback =  [["system", [{"type": "text", "text": query}]]]
        # role_name = self.generate_img_text("system", query, image_path)
        # role_name = self.inference_chat(feedback, 'gpt-4o-2024-08-06', self.API_url, self.token)
        # print(role_name)
        # role_list = role_name.split('##########')
        
       
        selected_role_list = [role_list[0],role_list[1]]
        role_decision_list =  [[] for _ in range(len(selected_role_list))]
        rest_role = [role_list[2]]
        final_answer = []
        ############# update role ##############
        for i, role_i in enumerate(selected_role_list):
            step_i = 0
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert "+ role_i +  " in solving reasonning problem. "
                    "Your task is to identify key information from the user's problem, ensuring accurate recognition of complete visual objects without mistaking parts of an object as the full target of the problem, connect relevant details from both textual and visual content, and solve the problem based on the information provided. Clearly explain your reasoning in a step-by-step manner, providing detailed insights drawn from the content. "
                    "For comparisons, briefly summarize the differences between all options, focusing on the main distinctions without repeating detailed descriptions to keep it concise. Each step should address a unique aspect of the problem, ensuring that consecutive steps are distinct from one another.\n" 
                    "### For each step ###\n"
                    "1. Provide a clear, concise title describing the current reasoning phase.\n"
                    "2. Elaborate on your thought process in the content section.\n"
                    "3. Decide whether to continue reasoning or provide a final answer.\n"
                    "Response Format:\n"
                    "Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')\n"
                    "Example of a valid JSON response:\n{\n\"title\": \"Initial Problem Analysis / Initial Analysis of the Problem\",\n\"content\": \"To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.\",\n\"next_action\": \"continue\"\n}"
                    "Note: This example of the output format is for illustration only; each response should adapt to the specifics of the current problem. \n "
                    "### Key Instructions ###\n"
                    "- Employ at least 5 distinct reasoning steps.\n"
                    "- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.\n"
                    "- Actively explore and evaluate alternative answers or approaches.\n"
                    "- Critically assess your own reasoning; identify potential flaws or biases.\n"
                    "- Acknowledge potential errors. When re-examining, truly challenge your initial approach by using a fundamentally different method or perspective—don not just rephrase or modify. Engage in a genuine re-evaluation. "#YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. When re-examining, employ a fundamentally different approach or perspective.\n"
                    "- Consider the possibility that you may be wrong, and pinpoint where your reasoning could potentially fail.\n"
                    "- Utilize at least 5 diverse methods to critically challenge your answer.\n  "#derive or verify your answer.\n"
                    "- Incorporate relevant domain knowledge and best practices in your reasoning.\n"
                    "- Quantify certainty levels for each step and the final conclusion when applicable.\n"
                    "- Consider potential edge cases or exceptions to your reasoning.\n"
                    "- Provide clear justifications for eliminating alternative hypotheses.\n"
                    "- Accurately align evidence with corresponding options by employing intermediate conclusions, must be sure to match the evidence to conclusions or options that acknowledge all factors. \n"
                    "### ATTENTION ###\n"
                    "When you re-exam the reasoning path, you must output 're-exam' or 're-evaluate' to show the behavior. \n"
                    "- Each Problem only has a correct answer. "
                # "role": "system", "content": "You are an expert "+ role_i +  " in solving reasonning problem. "
                # "Your task is to recognize the key information from the user's problem and solve user's problem according to the image conetent. "
                # "You need clearly explain your reasoning step by step in a detail manner.  Briefly summarize the differences between all options. Focus on highlighting the main differences without repeating detailed descriptions, and keep it concise. " #And each step only solve one aspect of the problem. The previous step and next step must  be not same.
                # "For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', 'important information' and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.\n\nExample of a valid JSON response:\n{\n\"title\": \"Identifying Key Information\",\n\"content\": \"To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...\",\n\"next_action\": \"continue\"\n}"
                },
                {"role": "user", "content": [
                    {
                        "type": "text", 
                        "text": "#### Basic Background ###\n"+text_analysis+'\n\n'+user_instruction
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                ]},      
                {"role": "assistant", "content": "Thank you! I will generate the thinking step, following my instructions and starting from the beginning after breaking down the problem."}
            ]
            
            while True:
                while 1:
                    try:
                        step_data = self.COT_action_squences(messages) #self.inference_chat(messages, 'gpt-4o-2024-08-06', API_url, token)
                        break
                    except:
                        continue
                # current_step_data = f"Step {step_i}: title={step_data.parsed.title},  content={step_data.parsed.content}" #  action_squences={step_data.parsed.action_squences
                current_step_data = f"Please continue generating the coontents in Step {step_i+1} using the following information: \n" + f"Step {step_i}: title={step_data.parsed.title},  content={step_data.parsed.content} \n In Step {step_i+1}, build upon the previous step by introducing a new perspective, additional details, or different aspects of the topic. Focus on exploring variables, conditions, or scenarios that were not covered before. Make sure the new content brings fresh information or analysis to the process, avoiding any repetition of previous content."  #  action_squences={step_data.parsed.action_squences
                logger_step_data = f"Step {step_i}: title={step_data.parsed.title},  content={step_data.parsed.content}"
                
                if(len(rest_role)>0 and 'Re-examin' in current_step_data or 're-' in current_step_data or 'Re-' in current_step_data):
                    if(len(current_step_data)==0):
                        infer_path = current_step_data
                    else:
                        infer_path = '\n'.join(role_decision_list[i])
                    dis_query = (
                        "**User's Question:**  \"" + user_instruction + "\" \n" +
                        "**Role:** You are a highly skilled \"" + rest_role[0] + "\" with expertise in reasoning problems. \n" +
                        "**Task:** Evaluate a reasoning path "+ infer_path +" provided by an expert \"" + role_i + "\". Your job is to assess whether the re-examination step in the reasoning process is logical and valid. \n" +
                        "**Response Format:** Respond with only 'yes', 'no', or 'uncertain'. \n" +
                        "**Your Answer:**"
                    )
                    feedback = [["user", [{"type": "image_url",   "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]], ["user", [{"type": "text", "text": dis_query}]]]
                    dis_rs = self.inference_chat(feedback, 'gpt-4o-2024-08-06', self.API_url, self.token)
                    if(dis_rs=='yes'):
                        pass
                    elif(dis_rs=='no'):
                        update_query = (
                            "**User's Question:**  \"" + user_instruction + "\" \n" +
                            "**Role:** You are a highly skilled \"" + rest_role[0] + "\" with expertise in reasoning problems. \n" +
                            "**Task:** You need to use the previous steps: \n" '\n'.join(role_decision_list[i][0:len(role_decision_list[i])-1]) + " to generate a new thinking step. \n" +
                            "**Response Format:** only output the content using the format: title= , content= . \n" +
                            "**Your Answer:**"
                        )
                        feedback = [["user", [{"type": "image_url",   "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]], ["user", [{"type": "text", "text": update_query}]]]
                        update_step = self.inference_chat(feedback, 'gpt-4o-2024-08-06', self.API_url, self.token)
                        current_step_data =f"Step {step_i}:"  + update_step
                    elif(dis_rs=='uncertain'):
                        break
                role_decision_list[i].append(current_step_data)
                messages.append({"role": "assistant", "content": current_step_data})
                if step_data.parsed.next_action == 'final_answer' and len(role_decision_list[i])==1:
                    current_step_data+='\n '+ 'Please carefully check the reasoning step, step by step. \n'
                    continue
                if step_data.parsed.next_action == 'final_answer' or step_i > 25: # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
                    break
                step_i += 1
            #messages.append({"role": "user", "content": "Please only respond the final instruction2 based on your reasoning above. \n### [Output Format] ###: step(i) -description  \n -action "})
            messages.append({"role": "user", "content": "Please provide the final answer based solely on your reasoning above. Only output 'A','B','C' or 'D'' without any additional content, titles, or formatting."}) #Do not use JSON formatting. Only provide the text response without any titles or preambles. \n ** Output Format: ** yes or no, do not output other contents. "})  Only output 'A','B','C' or 'D'' without any additional content, titles, or formatting.
            final_data = self.COT_instruction(messages)
            result1 = json.loads(final_data.content)['content']
            result = ''.join(filter(str.isalpha, result1))
            if(result.lower() not in ['a','b','c','d']):
                query = (
                    "**User's Question:**  "+user_instruction+' \n'
                    "**Model Response:**" + result1 + '\n'
                    "**Your Task:** Carefully read the example provided, then select the most suitable answer from the model response based on the options given.  \n"
                    "Note: Do not be confused by any appearances of A, B, C, or D within the text itself. Only select the answer that best matches the instructions from options.\n"
                    "**Output Format:** You must respond with either A, B, C or D  only. Do not provide any additional content. \n"
                    "**Your Answer:**"
                )
                feedback = [["user", [{"type": "text", "text": query}]]] #[{"type": "image_url",   "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]],
                result = self.inference_chat(feedback, 'gpt-4o-2024-08-06', self.API_url, self.token)
                
            final_answer.append(result)

        all_final_answer = list(set(final_answer))
        if(len(all_final_answer)==1):
            final_answer = all_final_answer[0]
        else:
            if(len(final_answer)==3):
                final_answer = max(set(final_answer), key=final_answer.count)
            else:
                ######### method 1. use high-level verify #########
                re_exam_rs = []
                for i, path_route_i in enumerate(role_decision_list):
                    path_route =  '\n'.join(path_route_i) + '\n Conclusion:'+final_answer[i]
                    dis_query = (
                        # "**User's Question:**  \"" + user_instruction + "\" \n" +
                        # "**Role:** You are a highly skilled \"" + rest_role[0] + "\" with expertise in finding mistakes in the reasoning step. \n" +
                        # "**Task:** Evaluate provided reasoning pathes and its conclusion as following:"+ path_route + "\n Your job is to assess whether the reasoning path and its conclusion is logical and valid. \n" +
                        # "**Response Format:** Respond with only yes or no. \n" +
                        # "**Your Answer:**"
                        "#### Basic Background ###\n"+text_analysis+'\n\n'+
                        "**User's Question:** \"" + user_instruction + "\"\n" +
                        "**Role:** You are a highly skilled \"" + rest_role[0] + "\" with expertise in identifying errors in reasoning.\n" +
                        "**Task:** You need to observe the image from distinct aspects separately, rather than evaluating it as a whole. For each aspect, analyze specific details individually. Once you have observed each aspect, combine these observations to form a comprehensive understanding of the image, and then evaluate the provided reasoning paths and conclusions as follows: " + path_route + "\n"
                        "Your job is to assess whether the reasoning paths and conclusions are logical and valid.\n" +
                        "**Example for Reference:**\n" +
                        "- **Reasoning Path**: The magnets are positioned with opposite poles facing each other (N-S and S-N), which should result in attraction.\n" +
                        "- **Conclusion**: The magnets will repel each other.\n" +
                        "- **Your Evaluation**: Yes\n\n" +
                        "- **Reasoning Path**: The magnets are positioned with like poles facing each other (N-S or N-S), which should result in attraction due to magnetic interaction.\n" +
                        "- **Conclusion**: The magnets will attract each other.\n" +
                        "- **Your Evaluation**: No\n\n" +
                        "**Response Format:** Respond only with 'yes' or 'no'.\n" +
                        "**Your Answer:**"
                    )
                    feedback = [["user", [{"type": "image_url",   "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]], ["user", [{"type": "text", "text": dis_query}]]]
                    re_rs = self.inference_chat(feedback, 'gpt-4o-2024-08-06', self.API_url, self.token)
                    re_rs = ''.join(filter(str.isalpha, re_rs)).lower()
                    re_exam_rs.append(re_rs)
                # try:
                # final_answer = final_answer[]
                if(len([index for index, value in enumerate(re_exam_rs) if value == 'yes'])>1):
                    ###### temp#######
                    # final_answer = final_answer[-1]
                    step_i = 0
                    messages = [
                        {
                            "role": "system", 
                            "content": "You are an expert "+ rest_role[0] +  " in solving reasonning problem. "
                            "Your task is to identify key information from the user's problem, connect relevant details from both textual and visual content, and solve the problem based on the information provided. Clearly explain your reasoning in a step-by-step manner, providing detailed insights drawn from the content. "
                            "For comparisons, briefly summarize the differences between all options, focusing on the main distinctions without repeating detailed descriptions to keep it concise. Each step should address a unique aspect of the problem, ensuring that consecutive steps are distinct from one another.\n" 
                            "### For each step ###\n"
                            "1. Provide a clear, concise title describing the current reasoning phase.\n"
                            "2. Elaborate on your thought process in the content section.\n"
                            "3. Decide whether to continue reasoning or provide a final answer.\n"
                            "Response Format:\n"
                            "Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')\n"
                            "Example of a valid JSON response:\n{\n\"title\": \"Initial Problem Analysis / Initial Analysis of the Problem\",\n\"content\": \"To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.\",\n\"next_action\": \"continue\"\n}"
                            "Note: This example of the output format is for illustration only; each response should adapt to the specifics of the current problem. \n "
                            "### Key Instructions ###\n"
                            "- Employ at least 5 distinct reasoning steps.\n"
                            "- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.\n"
                            "- Actively explore and evaluate alternative answers or approaches.\n"
                            "- Critically assess your own reasoning; identify potential flaws or biases.\n"
                            "- When re-examining, employ a fundamentally different approach or perspective.\n"
                            "- Consider the possibility that you may be wrong, and pinpoint where your reasoning could potentially fail.\n"
                            "- Utilize at least 5 diverse methods to derive or verify your answer.\n"
                            "- Incorporate relevant domain knowledge and best practices in your reasoning.\n"
                            "- Quantify certainty levels for each step and the final conclusion when applicable.\n"
                            "- Consider potential edge cases or exceptions to your reasoning.\n"
                            "- Provide clear justifications for eliminating alternative hypotheses.\n"
                            "- Accurately align evidence with corresponding options by employing intermediate conclusions, must be sure to match the evidence to conclusions or options that acknowledge all factors. \n"
                            "### ATTETNTION ###\n"
                            "- Each Problem only has a correct answer. "
                        },
                        # "You need clearly explain your reasoning step by step in a detail manner.  Briefly summarize the differences between all options. Focus on highlighting the main differences without repeating detailed descriptions, and keep it concise. " #And each step only solve one aspect of the problem. The previous step and next step must  be not same.
                        # "For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', 'important information' and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.\n\nExample of a valid JSON response:\n{\n\"title\": \"Identifying Key Information\",\n\"content\": \"To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...\",\n\"next_action\": \"continue\"\n}"},
                        {"role": "user", "content": [
                            {
                                "type": "text", 
                                "text": "#### Basic Background ###\n"+text_analysis+'\n\n'+user_instruction
                            },
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ]},      
                        {"role": "assistant", "content": "Thank you! I will generate the thinking step, following my instructions and starting from the beginning after breaking down the problem."}
                    ]
                    
                    while True:
                        while 1:
                            try:
                                step_data = self.COT_action_squences(messages) #self.inference_chat(messages, 'gpt-4o-2024-08-06', API_url, token)
                                break
                            except:
                                continue
                        # current_step_data = f"Step {step_i}: title={step_data.parsed.title},  content={step_data.parsed.content}" #  action_squences={step_data.parsed.action_squences
                        current_step_data = f"Please continue generating the coontents in Step {step_i+1} using the following information: \n" + f"Step {step_i}: title={step_data.parsed.title},  content={step_data.parsed.content} \n In Step {step_i+1}, build upon the previous step by introducing a new perspective, additional details, or different aspects of the topic. Focus on exploring variables, conditions, or scenarios that were not covered before. Make sure the new content brings fresh information or analysis to the process, avoiding any repetition of previous content."  #  action_squences={step_data.parsed.action_squences
                        logger_step_data = f"Step {step_i}: title={step_data.parsed.title},  content={step_data.parsed.content}"
                        messages.append({"role": "assistant", "content": current_step_data})
                        if step_data.parsed.next_action == 'final_answer' or step_i > 25: # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
                            break
                        step_i += 1
                    #messages.append({"role": "user", "content": "Please only respond the final instruction2 based on your reasoning above. \n### [Output Format] ###: step(i) -description  \n -action "})
                    messages.append({"role": "user", "content": "Please provide the final answer  based solely on your reasoning above based on the options. Only output 'A','B','C' or 'D'' without any additional content, titles, or formatting. "}) #Do not use JSON formatting. Only provide the text response without any titles or preambles. \n ** Output Format: ** yes or no, do not output other contents. "}) Only output 'A','B','C' or 'D'' without any additional content, titles, or formatting. 
                    final_data = self.COT_instruction(messages)
                    result1 = json.loads(final_data.content)['content']
                    result = ''.join(filter(str.isalpha, result1))
                    if(result.lower() not in ['a','b','c','d']):
                        query = (
                            "**User's Question:**  "+user_instruction+' \n'
                            "**Model Response:**" + result1 + '\n'
                            "**Your Task:** Carefully read the example provided, then select the most suitable answer from the model response based on the options given.  \n"
                            "Note: Do not be confused by any appearances of A, B, C, or D within the text itself. Only select the answer that best matches the instructions from options.\n"
                            "**Output Format:** You must respond with either A, B, C or D  only. Do not provide any additional content. \n"
                            "**Your Answer:**"
                        )
                        feedback = [["user", [{"type": "text", "text": query}]]] #["user", [{"type": "image_url",   "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]], 
                        result = self.inference_chat(feedback, 'gpt-4o-2024-08-06', self.API_url, self.token)
                        
                    final_answer.append(result)
                else:
                    final_answer = final_answer[0]
                    #pdb.set_trace()
                ######### method 2. retest using high-level #########   
        final_answer = max(set(final_answer), key=final_answer.count) 
        save_rs =user_instruction + '\t' + '\t'+final_answer #+ gt_answer+ image_name +  '\t' + 
        print(save_rs)
        return final_answer 
        # #save_rs = image_name +  '\t' + user_instruction + '\t' + gt_answer+ '\t'+final_answer

        # f.write(save_rs+'\n')  