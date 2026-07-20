from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
_=load_dotenv(find_dotenv(usecwd=True))

Nutrition_details= """ Creatine, Whey protien isolate"""

def prompt_execution (Prompt,title,**kwargs):
    llm1= ChatOpenAI(model="gpt-4o")
    user_query=input("Do you want to Enter the supplements (yes/no): ")
    if user_query.lower()=="yes":
        kwargs= input("Please enter the supplements sperated by comma")
        user_query1=input("Do you have any other query (yes/no): ")
        if user_query1.lower()=="yes":
            query = input("Please enter the query:")
            kwargs= kwargs + " and answer this follwoing query " + query
    else:
        kwargs=Nutrition_details
        
    #prompt1=ChatPromptTemplate.from_messages(Prompt)
    prompt1=ChatPromptTemplate.from_template(Prompt)
    chain1=prompt1|llm1|StrOutputParser()
    print(f"Title: {title}\n")
    print(f"Prompt:{Prompt} \n\n Output:{chain1.invoke(kwargs)}")

# with persona and output formats
prompt_template="""Act as a nutritionist and provide 3 benefits and sideeffects of the following {supplements}. 
the output should be in bullet points """    
# with system message and human message
prompt_messages=[("system","""Dont answer the questions other than nutrition like medical tablets advice. 
If any asked, answer strictly as that you're not doctor to prescribe it."""),
("human","Explain the benefits and sideeffects of the following supplements {supplements}")]
#prompt_execution(prompt_messages,"Benefits and Sideeffects of Supplements",supplements=Nutrition_details)
#prompt_execution(prompt_template,"Benefits and Sideeffects of Supplements",supplements=Nutrition_details)
# N-shot promp prompt 
#### Zero examples
Zero_shot_prompt= """ Act as a nutritionist and provide 3 benefits and sideeffects of the following {supplements}.
along with classifying the supplements as protein rich, vitamin rich or collagen rich. The classification in the output should be in Bold"""
#prompt_execution(Zero_shot_prompt,"Benefits and Sideeffects of Supplements",supplements=Nutrition_details)
####Few shot prompt with examples
Few_shot_prompt= """Act as a nutritionist and provide 3 benefits and sideeffects of the following {supplements}.
The output should be in bullet points and classify the supplements as protein rich, vitamin rich or collagen rich. The classification in the output should be in Bold.
Example 1: ---------Classification : Fiber rich-------------
 Supplement name: Psyllium Husk
 
 Benefits:
1. **Digestive Health**: Excellent for promoting regular bowel movements and preventing constipation.
2. **Heart Health**: Can help to lower cholesterol levels, contributing to heart health.
3. **Blood Sugar Control**: May aid in stabilizing blood sugar levels in people with diabetes.

 Side Effects:
1. **Gastrointestinal Distress**: Can cause gas, bloating, and stomach cramps in some individuals.
2. **Choking Hazard**: Risk of choking if not taken with sufficient water.
3. **Allergic Reactions**: Possible allergic reactions in some people, especially with respiratory or skin issues. 

recommended for kids : no
recommended for adults: yes

"""
#prompt_execution(Few_shot_prompt,"Benefits and Sideeffects of Supplements",supplements=Nutrition_details)

### Few shot prompt with template
examples=[{
    'input':'Give the benefits and sideeffects of the following supplements: Psyllium Husk,Spirullina',
    'output':'''``` json{
    SUPPLEMENT 1:
    "Name":"Psyllium Husk"
    "Recommended for Kids":"No"
    "Recommended for adults":"Yes"
    "Benefits":"Digestive Health,Heart Health.Blood Sugar Control"
    "Sideeffects":"Gastrointestinal Distress,Choking Hazard,Allergic Reactions"
     SUPPLEMENT 2:
     "Name":"Spirullina"
    "Recommended for Kids":"No"
    "Recommended for adults":"Yes"
    "Benefits":"Digestive Health,Heart Health.Blood Sugar Control"
    "Sideeffects":"Gastrointestinal Distress,Choking Hazard,Allergic Reactions"

    } ``` '''
},
{

    'input':'Give the benefits and sideeffects of the following supplements: Spirullina,Psyllium Husk',
    'output':'''``` json{
    SUPPLEMENT 1:
    "Name":"Spirullina"
    "Recommended for Kids":"No"
    "Recommended for adults":"Yes"
    "Benefits":"Digestive Health,Heart Health.Blood Sugar Control"
    "Sideeffects":"Gastrointestinal Distress,Choking Hazard,Allergic Reactions"
     SUPPLEMENT 2:
     "Name":"Psyllium Husk"
    "Recommended for Kids":"No"
    "Recommended for adults":"Yes"
    "Benefits":"Digestive Health,Heart Health.Blood Sugar Control"
    "Sideeffects":"Gastrointestinal Distress,Choking Hazard,Allergic Reactions"
    } ``` '''

},]
example_prompt=ChatPromptTemplate.from_messages([('human','{input}'),('ai','{output}'),])


Few_shot_prompt_template=FewShotChatMessagePromptTemplate(example_prompt=example_prompt,examples=examples,)
final_prompt= ChatPromptTemplate.from_messages([
    ('system','Act as a nutritionist and provide 3 benefits and sideeffects of the following list. The output should be as in the mentioned template'),
    Few_shot_prompt_template,
    ('human','{input}')    

])
print(final_prompt.format(input='Give the benefits and sideeffects of the following supplements: almond milk,isagbol'))
llm = ChatOpenAI(model="gpt-4o")
chain = final_prompt | llm | StrOutputParser()

response = chain.invoke({
    'input': 'Give the benefits and sideeffects of the following supplements: almond milk,castor oil'
})
print(response)