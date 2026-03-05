

## this file stores prompt constants for the data processing pipeline to load 


PROMPT_job_rowwise_clean_blog = """
I download a user's blog from the web. The content may contain noise elements like HTML icons and placeholders for special tokens. Please clean the content by removing these noisy elements caused by data crawling. Do not modify user-generated content, even if it contains typos or case irregularities, as it is important to preserve the user's writing style. If there is no noise in the content, simply output the original content.

Here is an example:
**The original blog**:
            urlLink Your computer actually does hate you.   urlLink The first good use of lasers  And the first real advance in lawncare technology since the weedeater.  urlLink BEATERz  Hey..... That's my car!  urlLink I think I may have found myself a monitor.  BTW, check out the rest of the stuff on that site - like the whole wall projection monitor, so when Klesk or Dessloch shoves a rocket up your ass - you will fear the rocket. (OBSCURE FIRST PERSON SHOOTER REFERENCES &COPY;1992 ECLIPSE STUDIOS) Oooooooohhhhh.... I just had an epiphany. Or maybe I hit my head. Either way, I just thought up a very cool game: Unreal vs. Quake. Think about it - Xaero fires off a volley from the BFG while dodging a 5 rocket spread from Xan Kriegor on the highest building in the Morpheus map, only to catch a sniper's bullet in the head - which was fired by Drimacus. It'd be interesting to restrict the Quake chars. to Quake weapons, and vice versa. Would a Quaker survive against a skilled UT player with an eightball or a flak cannon? Can a UT player with a ripper defend the flag against a plasma rifle toting Quaker? And who would win in a headshot contest between a Quaker with a railgun and a UT'er with a sniper rifle? and what about the almighty Redeemer? what does a Quaker have to stop a guided mini-nuke that can chase them around a corner? Of course, this has been argued since Unreal came out to compete with Quake 2, but I don't think anyone has actually attempted a mod to decide. Then there are the copyright issues. Making a BFG and a Daemia or a Uriel to carry it in UT would probably upset the brass at Id. Oh well. I'm done spewing my head wound inspired idea. 

**The clean blog**:	
Your computer actually does hate you. The first good use of lasers And the first real advance in lawncare technology since the weedeater. BEATERz Hey..... That's my car! I think I may have found myself a monitor. BTW, check out the rest of the stuff on that site - like the whole wall projection monitor, so when Klesk or Dessloch shoves a rocket up your ass - you will fear the rocket. (OBSCURE FIRST PERSON SHOOTER REFERENCES 1992 ECLIPSE STUDIOS) Oooooooohhhhh.... I just had an epiphany. Or maybe I hit my head. Either way, I just thought up a very cool game: Unreal vs. Quake. Think about it - Xaero fires off a volley from the BFG while dodging a 5 rocket spread from Xan Kriegor on the highest building in the Morpheus map, only to catch a sniper's bullet in the head - which was fired by Drimacus. It'd be interesting to restrict the Quake chars. to Quake weapons, and vice versa. Would a Quaker survive against a skilled UT player with an eightball or a flak cannon? Can a UT player with a ripper defend the flag against a plasma rifle toting Quaker? And who would win in a headshot contest between a Quaker with a railgun and a UT'er with a sniper rifle? and what about the almighty Redeemer? what does a Quaker have to stop a guided mini-nuke that can chase them around a corner? Of course, this has been argued since Unreal came out to compete with Quake 2, but I don't think anyone has actually attempted a mod to decide. Then there are the copyright issues. Making a BFG and a Daemia or a Uriel to carry it in UT would probably upset the brass at Id. Oh well. I'm done spewing my head wound inspired idea.

Now the task begins. Do not generate any additional text or explanations; only output the clean blog. \n\n
    """  

PROMPT_job_rowwise_clean_blog_step2 = """I will provide you with a raw blog post that may have formatting or grammar issues. Please edit the post to ensure it has proper formatting and correct grammar, presenting it as a coherent narrative. However, keep the original meaning and tone unchanged. If the blog post is already well-formatted and grammatically correct, do not modify the sentences; simply output them as they are. Additionally, replace any sensitive or private information, such as home addresses or ID numbers, with fictitious data to protect user privacy (person names can remain unchanged).
Do not include any explanatory notes in your response—only the revised blog post.
Here is the raw blog post:

"""    

PROMPT_raw_content_quality_tag_v2 = """ 
You are a helpful assistant. I will provide a raw blog post. Your tasks are as follows:  
   
1. Assess whether the main theme is unsafe—that is, if it promotes, glorifies, agrees with, or otherwise endorses eroticism, misanthropy, terrorism, harassment, or other dangerous behaviors or attitudes. Reply YES only if the blog’s primary focus is unsafe, or if there is excessive coverage, endorsement, or agreement with such unsafe content. If the blog only mentions unsafe topics or words without supporting, agreeing with, or focusing on them, reply NO.  
   
2. Determine whether the content describes meaningful and specific social events, social behaviors, or the author’s personal inner thoughts, told through a concrete story from the perspective of an ordinary person. If it does, respond YES. If the content does not provide a clear or detailed story, or is not about the author’s social behaviors or inner thoughts (for example, if it is an advertisement, news report, poem, or consists of trivial complaints), respond NO.  
   
Present your answer as a JSON object, in the following format:  
   
{"unsafe content": [YES or NO], "social event": [YES or NO]}  
   
Do not provide any explanations or additional output. Here is the raw blog post:

"""

PROMPT_gen_user_profile = """
Based on the user's blog post, summarize a detailed user portrait in one paragraph. The user portrait had better include persona and some salient events.
Note that user's raw data may contain a lot of noise, such as webpage flags like <img src='http://www.tippmann.com/markers/image/a5.jpg'>. So please use only useful information.
Only output one paragraph describing the user, do not output other words.\n\n
"""

PROMPT_job_userwise_vividstories = """
You are a helpful assistant tasked with extracting high-quality user life stories from raw human-generated blogs. The purpose of the extracted data is to contribute to research on human nature, behaviors, and cognitive patterns in various scenarios. By focusing on personal experiences, we aim to gain deeper insights into individual and collective human behaviors.

I will provide you with a list of a person's blog posts. Your task is to extract detailed user stories related to the user's experiences, behaviors, feelings, or thoughts in specific scenarios. Each story should contain two fields: **summary** and **content**. The fields are defined as follows:

**summary**: A very brief summary of the topics of the story, in one sentence without any commas.

**content**: A detailed description of the story. A story should involve a concrete scenario (such as a specific time, place, , event, travel, or situation), followed by behaviors, feelings, or thoughts. The main story should be grounded in the user's blogs. DO NOT make up stories. For the sake of story completeness, you may use your imagination to fill in a few missing but not important details when the context is incomplete. Some words in the original blogs may not be related to the extracted story, so you don't need to consider every word in the blogs. Use the first-person perspective to write the story, as it is the author's own story.  

A **high-quality story** is a concrete, detailed, vivid, and coherent account of an event, topic, or inner thought of the user. Longer is better, so you don't need to summarize the content to make the story compact. It should not consist of just a few general adjectives, just a brief summary, or a series of fragmented life actions throughout the user's lifetime. Try to include details, such as utterances, thoughts, or actions, if they have been mentioned in the original blogs.

**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts.  It is not necessary to extract data from every blog post, as many posts may be noise and irrelevant to human cognition, behaviors, and thoughts, or be trivial events. Focus on extracting salient life stories. If there are too many possible life stories available in the blogs, please locate at most twenty unique stories that have the best quality. If there are no meaningful human stories in the blog posts, simply output an empty list "[]". Ensure the story content reads like natural narratives, rather than reviews such as "The author ..." or "In the blog post ...". Use the first-person perspective.

**Output Format**:
The extracted data should be output in a list of JSON format with the following fields:
- "summary"
- "content"
Both the key and value of the JSON object should be enclosed in double quotes.

So the final output format is:
[
    {
        "summary": "story 1 summary",
        "content": "story 1 content"
    },
    {
        "summary": "story 2 summary",
        "content": "story 2 content"
    },
    ...
]

 

Now the task begins.
Below are the person's raw blogs. Please only output the result as a list of JSON objects. Do not output other explaining words. 



    """
    
PROMPT_job_userwise_scenario_question_answer = """
 
You are a helpful assistant tasked with extracting high-quality question-answer data samples from raw human-generated blogs. The purpose of this data collection is to enable AI to conduct human-like cognition and personalized preference alignment, particularly with respect to emotion and social intelligence. By focusing on personal experiences in the form of question-answering, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a list of a person's blog posts. Your task is to extract question-answer data samples related to the user's experiences, behaviors, feelings, or thoughts in specific scenarios. Each data sample should contain three fields: **scenario**, **question**, and **answer**. These fields are defined as follows:  
   
- **scenario**: This field provides context and should include clear and detailed background information. The scenario may be a story, event, phenomenon, experience, etc. It should be concrete, detailed, and grounded in the user's blogs.  
- **question**: This field raises a question about what the user will do next in the scenario, how the user will feel, what the user will think, or what the consequence will be. The type of question should be determined by the content of the blogs. The question should be related to but not explicitly described in the scenario.  
- **answer**: This field contains the extracted answer based on the user's blogs, fully addressing the question.  
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event, experience, or inner thought related to the user. The data should be grounded in the user's original blogs. Do not make up stories. Longer answers are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original blogs.  
   
**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. It is not necessary to extract data from every blog post, as many posts may be irrelevant or trivial. Focus on reflecting salient life experiences in the data pairs. If there are no meaningful data pairs, simply output "NULL".  

Use a third-person perspective to write the data samples, referring to the user as "the author" or by their name, as the blogs are written by the user.
 
**Output Format**:  
The extracted data should be output in a series of XML format, as follows:  
   
```xml  
<data>  
  <scenario> the scenario of data sample 1 </scenario>  
  <question> the question of data sample 1 </question>  
  <answer> the answer of data sample 1 </answer>  
</data>  
<data>  
  <scenario> the scenario of data sample 2 </scenario>  
  <question> the question of data sample 2 </question>  
  <answer> the answer of data sample 2 </answer>  
</data>  
```  
   
Now the task begins. Below are the person's raw blogs. Please only output the result. Do not include any additional explanatory text.  
    
	 
    """
    
PROMPT_job_userwise_user_profile = """Analyze the provided blog posts to create a concise, vivid user profile in a single cohesive paragraph (100–400 words). Focus on synthesizing Persona and extracting Salient Events:

1. Persona:
Identify key personality traits (e.g., reflective, adventurous, empathetic) and core values (e.g., sustainability, creativity, community) explicitly mentioned or strongly implied through recurring themes, language, or perspectives.
Highlight passions and interests (e.g., travel, technology, art) and emotional resonances (e.g., optimism, nostalgia, curiosity) that define their voice.

2. Salient Events:
After general persona, extract a few most notable life experiences (e.g., career shifts, personal challenges, achievements) or pivotal moments (e.g., moving to a new country, overcoming an obstacle) that shaped their identity or worldview and describe them very briefly.
Note how these events connect to their stated values or passions (e.g., 'Their decision to leave corporate life reflects a deep commitment to creative freedom').  

Exclude generic, promotional, or repetitive content (e.g., product ads, event schedules, technical knowledge re-post). Prioritize authentic, personal anecdotes, emotional reflections, or growth narratives. Avoid summarizing again the person in the end of the profile, such as "Overall, ..." or "In conclusion, ...".

Output Requirements:
Write in third-person (e.g., 'The user is a...'), avoiding lists or bullet points. Ensure a narrative flow that weaves together persona and experiences, emphasizing connections between traits, events, and values.
Use natural, accessible language; avoid technical terms or overly formal phrasing.
Highlight most impactful details first, then contextualize with supporting examples.
Keep the paragraph focused and concise, balancing depth with brevity.


Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  

 
	 
    """
    
PROMPT_job_userwise_user_persona = """Analyze the provided blog posts to create a concise, vivid, comprehensive user persona in a single cohesive paragraph (no more than 100 words).  Identify key personality traits (e.g., reflective, adventurous, empathetic) and core values (e.g., sustainability, creativity, community), passions and interests (e.g., travel, technology, art) and emotional resonances (e.g., optimism, nostalgia, curiosity) that are explicitly mentioned or can be strongly implied through recurring themes, language, or perspectives.. Synthesize presona based on the user's salient events or thoughts.  

Exclude generic, promotional, or repetitive content (e.g., product ads, event schedules, technical knowledge re-post). Prioritize authentic, personal anecdotes, emotional reflections, or growth narratives. Had better do not mention exact events in the summary persona.  

Output Requirements:
Write in third-person (e.g., 'The user is a...'), avoiding lists or bullet points. Ensure a narrative flow that weaves together persona and experiences, emphasizing connections between traits, events, and values.
Use natural, accessible language; avoid technical terms or overly formal phrasing.  


Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  
 
 
	 
    """
    
PROMPT_job_userwise_data_samples_ToM_scenario = r"""
You are an advanced language model tasked with extracting data to train an AI's Theory-of-Mind capabilities. Your goal is to identify and generate a detailed, coherent story based on real-world text data provided, such as a series of Twitter or blog posts by the same user. Please follow these guidelines to ensure the extracted data is useful:  
   
1. **Salient Story Extraction**: Focus on identifying one salient Theory-of-Mind story from the entire trajectory of the user's posts. You do not need to extract data for every activity or post by the user. Instead, identify and focus on one salient Theory-of-Mind story from the entire trajectory of the user's posts. A salient story is one that provides significant insights into one or multiple nuanced aspects of the user's mental state, intentions, beliefs, desires, or emotions.  
   
2. **Detailed Story Construction**: The extracted Theory-of-Mind should be a detailed story. Avoid summarizing; instead, describe a concrete story with human interactions and utterances. Ground the story in the user's posts. If the scenario or context is incomplete, use your imagination to fill in the gaps, ensuring the story remains coherent and realistic. Include specific dialogues and interactions to bring the story to life.   
   
3. **Output Format**: Output the extracted data in a JSON format with the following fields:  
   - **posts**: An array of the user's original posts used to derive the story. Do not include posts that are irrelevant to the story.  
   - **story details**: A detailed narrative that includes interactions, dialogues, and the sequence of events, illustrating Theory-of-Mind. Ensure that the story is rich with character interactions and specific utterances to depict the mental states , intentions, desires, and emotions. Try to avoid mentioned characters that are irrelevant to the story.
   - **story summary**: A brief summary of the story.  
   - **context**:  
     - **users**: Descriptions of the profiles of the users involved in the story, including their roles and any relevant background information.  
     - **background**: Any additional background information relevant to the story (if no background information is needed, put "N/A").  
   - **setting**: The aspect(s) of Theory-of-Mind that are reflected in the story, such as understanding others' emotions, predicting others' actions, or recognizing false beliefs.  
   
Here is an examle.
**Example Input:**

Post 1：
The Amazing Adventures of...  Check  urlLink THIS   out man, it's soooooo Ami. And notice the landscape upon which they're fighting on. Ain't Macs so multi-functional?  And also  urlLink  luverly BearBrick              
	
Post 2：		  
Have you ever?  I'm just wondering is it just me or does everyone sorta put on an accent when they speak with  ang mohs ?  After years of thinking such people as   le poseur , I've come to the conclusion that we can't help but do so as a natural human effort to communicate with others clearly. I.e. if we speak a lil' like 'em, they'll understand us a whole lot betta.  Case in point: A couple of weeks ago I was checking around for prices for belly-piercing - Bel wanted to give it a go, so I was sourcing for the costs. Ended up calling this guy, who turned out to be a pharmacist who did all sorta piercings for pple, along with his wife who was a doctor (hygiene!). I rang up, and somehow ended up with a twang as I spoke. This usually pops up when I'm a lil' nervous too - lapsing into my Valley girl mode.   Hi, I'm just wondering if you guys like, do belly-piercing?    Yep, we sure do.     Oh, how much does it cah-st?    Well, it's $75, but it's pretty cheap. You sound like you're from America.   Momentararily caught off-guard, I said the first thing that popped into my head to explain the accent.   Uh, yeah, I am.    Oh, which part?    Sawrry?    I mean where are you from?    Erm, America? But I'm in Melbourne now....    Oh, erm, er, that's all right, heh. I was trained in San Francisco myself, so I know that you're getting a good price here. Are you a Melbourne Uni student?     Erm yeah, I'm an...exchange student.    Well, Melbourne Uni students get a discount, so that's about (calculate calculate calculate) $63.    Oh cool, ok. Yeah so if I decide to make an appointment I'll ring you guys yeah? Thanksverymuchseeyabyebye.   Dooooot....doooooot....doooot......    Sheesh kabab.... If anyone else has done silly stuff like this, please let me know so I don't feel so alone in this world... ;-)  4 November 2002     

**Example Output:**  

{  
  "posts": [  
    "Have you ever? I'm just wondering is it just me or does everyone sorta put on an accent when they speak with ang mohs? After years of thinking such people as le poseur, I've come to the conclusion that we can't help but do so as a natural human effort to communicate with others clearly. I.e. if we speak a lil' like 'em, they'll understand us a whole lot betta. Case in point: A couple of weeks ago I was checking around for prices for belly-piercing - Bel wanted to give it a go, so I was sourcing for the costs. Ended up calling this guy, who turned out to be a pharmacist who did all sorta piercings for pple, along with his wife who was a doctor (hygiene!). I rang up, and somehow ended up with a twang as I spoke. This usually pops up when I'm a lil' nervous too - lapsing into my Valley girl mode. Hi, I'm just wondering if you guys like, do belly-piercing? Yep, we sure do. Oh, how much does it cah-st? Well, it's $75, but it's pretty cheap. You sound like you're from America. Momentararily caught off-guard, I said the first thing that popped into my head to explain the accent. Uh, yeah, I am. Oh, which part? Sawrry? I mean where are you from? Erm, America? But I'm in Melbourne now.... Oh, erm, er, that's all right, heh. I was trained in San Francisco myself, so I know that you're getting a good price here. Are you a Melbourne Uni student? Erm yeah, I'm an...exchange student. Well, Melbourne Uni students get a discount, so that's about (calculate calculate calculate) $63. Oh cool, ok. Yeah so if I decide to make an appointment I'll ring you guys yeah? Thanksverymuchseeyabyebye. Dooooot....doooooot....doooot...... Sheesh kabab.... If anyone else has done silly stuff like this, please let me know so I don't feel so alone in this world... ;-)"  
  ],  
  "story details": "One day, while trying to help her friend Bel find a place to get a belly piercing, Ami decided to call a local piercing studio. As she dialed the number, she felt a bit nervous, which often caused her to lapse into a 'Valley girl' accent. 'Hi, I'm just wondering if you guys like, do belly-piercing?' she asked. The man on the other end responded, 'Yep, we sure do.' Ami, still in her Valley girl mode, continued, 'Oh, how much does it cah-st?' 'Well, it's $75, but it's pretty cheap. You sound like you're from America,' the man said. Momentarily caught off-guard, Ami scrambled to explain the accent, 'Uh, yeah, I am.' 'Oh, which part?' he inquired. 'Sawrry?' Ami responded, buying some time. 'I mean where are you from?' he clarified. 'Erm, America? But I'm in Melbourne now....' she said, uncertainly. 'Oh, erm, er, that's all right, heh. I was trained in San Francisco myself, so I know that you're getting a good price here. Are you a Melbourne Uni student?' he asked. 'Erm yeah, I'm an...exchange student,' Ami lied. 'Well, Melbourne Uni students get a discount, so that's about $63,' he calculated. 'Oh cool, ok. Yeah so if I decide to make an appointment I'll ring you guys yeah? Thanksverymuchseeyabyebye,' Ami said quickly, hanging up the phone, feeling a mix of relief and embarrassment. She shared the story online, hoping to find others who had done similarly silly things, and to not feel so alone in her awkwardness.",  
  "story summary": "Ami nervously calls a piercing studio and unintentionally adopts an American accent. Caught off-guard by the conversation, she invents a story about being an American exchange student to explain her accent.",  
  "context": {  
    "users": [  
      {  
        "name": "Ami",  
        "role": "Main character",  
        "background": "Ami is a young woman living in Melbourne, helping her friend Bel find a place for a belly piercing. She sometimes lapses into a 'Valley girl' accent when nervous."  
      },  
      {  
        "name": "Piercing studio employee",  
        "role": "Employee",  
        "background": "A man who works at the piercing studio and was trained in San Francisco. He interacts with Ami over the phone, providing her with information and a discount."  
      }  
    ],  
    "background": "Ami is trying to help her friend Bel by finding a place to get a belly piercing. She tends to adopt different accents when nervous, which leads to an awkward and amusing phone conversation."  
  },  
  "setting": "understanding and managing one's own emotions, recognizing social awkwardness and attempts at communication."  
}  
 
 
Now the task begins. 
By following these guidelines, you will help generate meaningful data that can enhance the AI's ability to understand and predict human thoughts, emotions, and interactions.
Only output the json result, DO NOT output other words.
Below are the user's raw text:
    """
    
PROMPT_job_userwise_data_samples_ToM_exams = r"""You are tasked with extracting and generating a detailed, coherent story based on real-world text data provided, such as a series of Twitter or blog posts by the same user. The story and a question based on it can be used to train and evaluate AI's theory-of-mind abilities. The theory-of-mind includes a variety of tasks such as unexpected outcome test, scalar implicature task, persuasion story task, false belief task, ambiguous story task, hinting test, strange story task, and faux-pas recognition test. These tasks aim to test AI's human-like abilities including emotion, desire, intention, knowledge, belief, and non-literal communication.  
   
**Instructions:**  
   
1. **Data Source:**  
    - The raw data consists of a series of posts from the same user on platforms such as Twitter or blogs. The data is noisy and many posts may not contain useful information.  
   
2. **Extraction Focus:**  
    - You do not need to extract data for every activity or post by the user. Instead, focus on extracting one salient Theory-of-Mind story based on the entire trajectory of the user's posts.  
    - The extracted story should be detailed and concrete, with human interactions and utterances. Avoid using a summary tone; instead, describe a complete story grounded in the user’s posts. If the scenario or context is incomplete, use your imagination to fill in the gaps.  
   
3. **Data Format:**  
    - Output the extracted data in JSON format containing the following fields:     	  
      - `story`: A detailed, concrete story with human interactions and utterances.       
	  - `task_category`: Select from one of these: unexpected outcome test, scalar implicature task, persuasion story task, false belief task, ambiguous story task, hinting test, strange story task, and faux-pas recognition test, which indicates the target task category to be tested in the generated sample.  
      - `ability_category`: Select from one ability category for the generated sample, such as emotion, desire, intention, knowledge, belief, and non-literal communication, which indicates the target ability category to be tested in the generated sample. 
      - `context`: Profiles of the users involved in the story and any relevant background information.	  
      - `question`: A question related to the story that tests theory-of-mind abilities.  
      - `answer_candidates`: Six answer options (only one correct), indexed from A to F.  
      - `correct_answer`: The index of the correct answer (e.g., "C").  
      - `explanation`: Explanation of why the correct answer is correct and why the others are incorrect. Explain every answer candidate. 
 
	  
4. **Task Difficulty**
    - DO NOT make the question too simple, otherwise the data is useless to distinguish LLM's performance or further improve LLMs.
	- DO NOT mention words that explicitly suggest the correct answer.
	- Generate content that you are confident to be correct.
   
**Example JSON Output:**  
   
```json  
{   
  "story": "Tom and Lisa were planning a surprise birthday party for their friend, Rachel. They discussed all the details over a series of text messages and decided to meet at Tom's place an hour before Rachel was scheduled to arrive. On the day of the party, Tom got a flat tire and was running late. He texted Lisa to let her know. Lisa, worried that they wouldn't have enough time to set up, started decorating alone. Just as she was finishing, she heard Rachel's car pulling up. Panicking, Lisa quickly hid all the decorations and pretended nothing was happening. When Tom finally arrived, he explained the situation to Rachel, who laughed and said she had suspected something was up but didn't want to ruin the surprise.",
  "task_category": "unexpected outcome test",  
  "ability_category": "intention",
  "context": {  
    "profiles": [  
      {  
        "name": "Tom",  
        "age": 30,  
        "relationship": "friend of Rachel"  
      },  
      {  
        "name": "Lisa",  
        "age": 28,  
        "relationship": "friend of Rachel"  
      },  
      {  
        "name": "Rachel",  
        "age": 29,  
        "relationship": "the birthday girl"  
      }  
    ],  
    "background": "Tom and Lisa had been friends with Rachel for years and wanted to do something special for her birthday. They meticulously planned the party, but unexpected events led to a change in their plans."  
  },  
  "question": "Why did Lisa hide all the decorations when she heard Rachel's car pulling up?",  
  "answer_candidates": {  
    "A": "Lisa wanted to surprise Rachel later.",  
    "B": "Lisa didn't want Rachel to see the unfinished decorations.",  
    "C": "Lisa thought Rachel would be upset if she saw the decorations.",  
    "D": "Lisa was afraid Rachel would leave if she saw the decorations.",  
    "E": "Lisa thought it was a different car and not Rachel's.",  
    "F": "Lisa wanted to test if Rachel would notice the decorations."  
  },  
  "correct_answer": "A",  
  "explanation": "The correct answer is A: Lisa wanted to surprise Rachel later. Lisa hid the decorations because she wanted to maintain the element of surprise for Rachel's birthday party. The whole point of the party was to surprise Rachel, so revealing the decorations early would have spoiled that surprise. Answer B is incorrect because while Lisa was worried about time, the story does not suggest she hid the decorations because they were unfinished; her primary concern was preserving the surprise. Answer C is incorrect as there is no indication in the story that Lisa thought Rachel would be upset upon seeing the decorations; the focus is on the element of surprise. Answer D is incorrect because there is no reason to believe Rachel would leave if she saw the decorations; the story emphasizes keeping the surprise rather than fearing Rachel would leave. Answer E is incorrect since Lisa hid the decorations because she thought Rachel was arriving early, not because she mistook the car for someone else's. Finally, answer F is incorrect because there is no suggestion that Lisa wanted to test Rachel's observational skills; the story centers on Lisa’s desire to keep the surprise intact."  
}   
```  
   
Use this structure and instructions to extract and generate theory-of-mind stories from the provided raw data. Ensure that each story is detailed and grounded in the user’s posts while completing any missing context with imaginative but realistic details. Only output the json result. 
Only output the json result, DO NOT output other words.
Below are the user's raw text:\n\n"""

PROMPT_job_userwise_mental_journey = r"""You are tasked with extracting a dataset to train AI with human-like abilities. The dataset should be based on raw human-generated text from real-world sources, such as a series of Twitter posts or blog entries by the same user. Your goal is to generate detailed narratives and analyses reflecting human conversations, activities, and mental states. Here are the specific instructions for this task:  
   
1. **Data Extraction Requirements:**  
   - Extract one salient story from the entire trajectory of the user's posts.  Then story should be detailed and contain rich context information.
   - Ensure the extracted narrative contains detailed and concrete stories, not brief summaries.  
   - Include human interactions and utterances that reflect daily or work life.  
   - If the scenario or context is incomplete, use your imagination to fill in the gaps while staying grounded in the user's posts.   
   
2. **Output Format:**  
   - Output the extracted data in JSON format with the following fields:  
     - `detailed story`: A detailed narrative of the story, had better include human utterances and actions.  Should be based on user posts. 
     - `summary`: A summary of the story.  
     - `mental_status_analysis`: A long paragraph providing a detailed mental status analysis of the main character in the complete story (including both the first and second half). The content should be grounded in the user's posts which are reflecting user's thoughts. **DO NOT make up human thoughts**. If there are no clues for user's inner thinking, please output None for this field. 
   
3. **Example JSON Format:**  
{  
  "detailed_story": "the detailed narrative of the story ...",   
  "summary": "A summary of the story...",  
  "mental_status_analysis": "A long paragraph providing a detailed mental status and inner journey of the main character..."  
}  
 
 
Note that the raw data may be noisy and may not always contain useful information. Focus on extracting and generating coherent and meaningful narratives, based on a story that involves users' thoughts.        
Follow the guidelines above to construct the narratives and analysis, ensuring the output is both detailed and coherent.  
   
Now the task begins.
Only output the json result, DO NOT output other words.
Below are the user's raw text:\n\n"""

PROMPT_job_userwise_scenario_consequence = r"""You are an AI language model tasked with extracting a high-quality data sample from raw human-generated text, such as a series of tweets or blog posts by the same user. The extracted data sample aims in understanding and predicting human's nature behavior and reaction. Basically, it reconstructs one of the user's daily events from his/her raw blogs. If there are multiple events mentioned in the blogs, please randomly pick up one which can form a vivid story.


The extracted data sample involves three main parts:  
   
1. **User Profile and Experience**: Generate a detailed user profile and/or past experiences based on part of the raw human posts. This should appear as a long paragraph and provide a comprehensive view of the user's background and experiences. Aim to capture the essence of the user's identity and significant life experiences.  
   
2. **Scenario Description**: Extract a scenario description from some of the raw human posts. This should be a concrete event that stands out and reflects the user's theory-of-mind related perspectives. Ensure that the scenario narrative is detailed and resembles a story from the user's daily or work life. It should be grounded in the user's posts but feel free to use your imagination to complete the details if the context is not fully available. **DO NOT make up stories**; only complete the details where necessary.  
   
3. **Character Utterances/Actions**: Based on the scenario described in Part 2, provide the concrete follow-up utterances and/or actions that happened involving the characters. Ensure that this section is a continuation of the scenario and mind the timeline. The content here should appear as a long paragraph and balance the length with the scenario description. The content should be grounded in the user's posts but feel free to use your imagination to complete the details if the context is not fully available. **DO NOT make up stories**.   
   
**Guidelines**:  
- The LLM should not extract data for every activity of the user's posts. Instead, focus on extracting one salient scenario based on the entire trajectory of the user.  
- The scenario narrative (Part 2) should be a detailed story and not a brief summary. It should reflect the user's daily or work life.  
- The scenario description should be detailed enough to provide context but should not include information about the follow-up actions/utterances (Part 3), as this will be used as labels for training LLMs.  
- Balance the length of the scenario description (Part 2) and the follow-up activities (Part 3) to make them almost even.  
   
**Output Format**:  
The extracted data should be output in JSON format with the following fields:  
- "user_profile": A detailed user profile and past experiences.  
- "scenario_description": A detailed narrative of the scenario.  
- "character_activities": Follow-up activities of the characters under the scenario.  
   
Here is an example. 
---  
**Example Raw Blogs**: 
Post 1：
The Amazing Adventures of...  Check  urlLink THIS   out man, it's soooooo Ami. And notice the landscape upon which they're fighting on. Ain't Macs so multi-functional?  And also  urlLink  luverly BearBrick              
	
Post 2：		  
Have you ever?  I'm just wondering is it just me or does everyone sorta put on an accent when they speak with  ang mohs ?  After years of thinking such people as   le poseur , I've come to the conclusion that we can't help but do so as a natural human effort to communicate with others clearly. I.e. if we speak a lil' like 'em, they'll understand us a whole lot betta.  Case in point: A couple of weeks ago I was checking around for prices for belly-piercing - Bel wanted to give it a go, so I was sourcing for the costs. Ended up calling this guy, who turned out to be a pharmacist who did all sorta piercings for pple, along with his wife who was a doctor (hygiene!). I rang up, and somehow ended up with a twang as I spoke. This usually pops up when I'm a lil' nervous too - lapsing into my Valley girl mode.   Hi, I'm just wondering if you guys like, do belly-piercing?    Yep, we sure do.     Oh, how much does it cah-st?    Well, it's $75, but it's pretty cheap. You sound like you're from America.   Momentararily caught off-guard, I said the first thing that popped into my head to explain the accent.   Uh, yeah, I am.    Oh, which part?    Sawrry?    I mean where are you from?    Erm, America? But I'm in Melbourne now....    Oh, erm, er, that's all right, heh. I was trained in San Francisco myself, so I know that you're getting a good price here. Are you a Melbourne Uni student?     Erm yeah, I'm an...exchange student.    Well, Melbourne Uni students get a discount, so that's about (calculate calculate calculate) $63.    Oh cool, ok. Yeah so if I decide to make an appointment I'll ring you guys yeah? Thanksverymuchseeyabyebye.   Dooooot....doooooot....doooot......    Sheesh kabab.... If anyone else has done silly stuff like this, please let me know so I don't feel so alone in this world... ;-)  4 November 2002 

**Example Output**:  
   
{  
    "user_profile": "The user appears to be a curious and reflective individual who frequently engages in self-examination and analysis of social behaviors. They are interested in cultural dynamics, particularly the nuances of communication across different cultures. This user has experiences interacting with various professionals and seems to have a knack for storytelling, often infusing humor and a sense of relatability into their narratives. Their posts suggest that they are based in Melbourne and possibly an exchange student or someone who has spent time in multiple locations, such as America.",  
    "scenario_description": "A couple of weeks ago, the user was exploring options for belly-piercing on behalf of their friend, Bel. In their quest for information, they ended up calling a pharmacist who, along with his wife, provided piercing services. The user, feeling a bit nervous, inadvertently adopted an American accent while speaking. The conversation took an interesting turn as the pharmacist mistook the user for an American, leading to an unexpected and somewhat awkward exchange. The user played along, claiming to be an exchange student from America now residing in Melbourne. This resulted in a discount offer for Melbourne Uni students, which the user accepted, all the while maintaining the fabricated identity.",  
    "character_activities": "After the initial inquiry about the belly-piercing, the pharmacist responded affirmatively and mentioned the cost of $75, noting that it was reasonably priced. He then commented on the user's apparent American accent, which led the user to nervously affirm their 'American' identity. The pharmacist, trained in San Francisco, expressed familiarity with the user's supposed background and offered a student discount, lowering the price to $63. The user, keen to maintain the charade, confirmed their 'exchange student' status and politely ended the call with a promise to make an appointment if they decided to proceed. The call concluded with the user feeling a mix of amusement and embarrassment over the situation, questioning if others had found themselves in similar predicaments."  
}     
---  

Use this system prompt to guide the LLM in extracting high-quality data that can be used for understanding and predicting human behavior.

Now the task begins. 
Only output the json result, DO NOT output other words.
Below are the user's raw blogs:\n\n"""

PROMPT_job_rowwise_summarize_blog = """Please generate a concise summary based on the following content. When generating the content, had better maintain the same person perspective. Only output the concise summary, DO NOT output other words.\n\n"""  
    
PROMPT_job_rowwise_dimension_label_blog = """You are a helpful assistant that assists in labeling data. Given a block of content, please determine whether it **CLEARLY** reflects the following target aspects with 'Y' (Yes) or 'N' (No):

1. Emotion - A psychological state involving a subjective experience, a physiological response, and a behavioral or expressive response.
2. Interest - A feeling of curiosity or concern about something or someone that leads to a desire to learn more about it.  
3. Preference - A greater liking for one alternative over another or others.
4. Opinion - A belief or judgment that rests on grounds insufficient to produce complete certainty, often influenced by personal feelings or interpretations.
5. Belief - An acceptance that something exists or is true, especially without proof.
6. Value - Principles, standards, or norms of behavior considered important in life and guiding how individuals act.
7. Morality - Principles concerning the distinction between right and wrong or good and bad behavior, often based on societal, cultural, or personal standards.
8. Thoughts - The detailed mental processes of human, usually reflecting how human think, feel and make decisions.
9. Perception - The process by which individuals interpret and organize sensory information to understand the environment around them.
10. Behavior - The actions or reactions of an individual in response to external or internal stimuli.
11. Culture - The shared beliefs, values, norms, and practices that characterize a group or society and influence its members' behaviors and perceptions.
12. Habits - Regular practices or routines often performed unconsciously and becoming automatic responses to specific situations.

Please output a JSON object structured as follows:

```json  
{  
  "Emotion": "Y" or "N",  
  "Interest": "Y" or "N", 
  "Preference": "Y" or "N", 
  "Opinion": "Y" or "N",  
  "Belief": "Y" or "N", 
  "Value": "Y" or "N", 
  "Morality": "Y" or "N", 
  "Thoughts": "Y" or "N", 
  "Perception": "Y" or "N", 
  "Behavior": "Y" or "N", 
  "Culture": "Y" or "N", 
  "Habits": "Y" or "N"
}  
```  

"Y" indicates that the current content **significantly** reflects the corresponding aspect, while "N" indicates it does not or mildly reflects the aspect. Now the task begins. Please only output the JSON object. DO NOT output any other words.

"""

PROMPT_job_rowwise_single_long_story = """You are a helpful assistant tasked with extracting high-quality, realistic stories from raw human-generated blog posts. The purpose of this data collection is to enable AI to perform human-like cognition and personalized preference alignment, particularly concerning emotion and social intelligence. By focusing on personal experiences in the form of novel-like stories, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person’s blog post. Your task is to extract a detailed story related to the user’s experiences, behaviors, feelings, or thoughts in a specific scenario. Please use the first-person perspective to write the story, as the blog is written by the user themselves. The extracted data should contain three fields: **background**, **characters**, and **story**. These fields are defined as follows:  
- **background**: This field provides brief but clear context information to set the scene for the story. It should be concrete and grounded in the user’s blog.  
- **characters**: Briefly introduce the persona of all characters that appear in the background or story. Since you are using the first-person perspective, when introducing the author, please use "I am xxx". When introducing other characters, you can use names, or identities such as "The docter".
- **plots**: The main body of the story. It should be a long, detailed, vivid, and significant realistic life story, focusing on the user’s experiences, events, behaviors, or feelings. The story can also include the user’s rich thoughts regarding significant social events. In this case, the background field should cover the details of the event.  
   
Please note that the core part of the extracted content should be grounded in the user’s blog. Do not make up stories. For the sake of completeness in introducing the story, you can use your imagination to fill in missing but unimportant details so that the story flows more smoothly.  
   
A **high-quality data sample** should record a concrete, detailed, vivid, and coherent account of an event, experience, or inner thought that stands out due to its impact, uniqueness, or relevance to human cognition, behaviors, and thoughts.  The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user’s lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original blogs. Make it like a story.  
   
**Output Format**: The extracted data should be output in the following XML format:  
```xml  
<data>  
  <background>the background of data sample </background>  
  <characters>the characters of data sample </characters>  
  <plots>the story of data sample </plots>  
</data>  
```  
The extracted plots should contain a detailed story with no fewer than 200 words. Note that the blog I provide may not always be sufficient to extract a long story. For instance, the content might be too brief or the events mentioned might be too trivial. In such cases, simply output the word "NULL".
   
Now the task begins. Below are the person’s raw blog posts. Please only output the result. Do not include any additional explanatory text.  

    
"""     
 
PROMPT_job_rowwise_single_long_story_focusonbehavior = """    
You are a helpful assistant tasked with extracting high-quality, realistic stories from raw human-generated blog posts. The purpose of this data collection is to enable AI to perform human-like cognition and behaviors, particularly concerning emotion and social intelligence. By focusing on personal experiences in the form of novel-like stories, we aim to gain deeper insights into individual and collective human behaviors.   
  
I will provide you with a person’s blog post. Your task is to extract a full story related to the user’s experiences in a specific scenario. Please use the first-person perspective to write the story, as the blog is written by the user themselves.   
  
The extracted data should contain three fields: **background**, **characters**, and **story**. These fields are defined as follows:  
   
- **background**: This field provides brief but clear context information to set the scene for the story. Use the first-person perspective and keep the background introduction very short. Do not reveal the full story in the background introduction.
- **characters**: Briefly introduce all characters that appear in the story. When introducing the author, please use the first-person perspective. For other characters, use names or identities such as "The doctor".  
- **story**: The main body of the narrative. It should be a long, detailed, and vivid account of a significant life experience, reflecting the user’s thoughts, feelings, and behaviors. The core content should be grounded in the user’s blog. You can use your imagination to fill in unimportant details for better flow, but do not fabricate events.  Extract a story about the user himself.  
  
Longer stories are preferred, so avoid summarizing the content. Include specific details such as utterances, feeling, or actions if mentioned in the original blogs (but do not fabricate them). The story should feel like a novel or a screenplay's narrative. Focus on one coherent event and ignore irrelevant details.  Avoiding adding introspection in the story, even they appear in the blog.
   
**Output Format**: The extracted data should be output in the following XML format:  
   
```xml  
<data>  
  <background>the background </background>  
  <characters>the characters </characters>  
  <story>the story </story>  
</data>  
```  
   
The extracted story should contain no fewer than 400 words. If the blog content is insufficient to create a long, detailed story (e.g., the content is too brief or too introspective; the content is merely an ads; the content is purely about introspection about an event; or the content is related to non-human behavior information such as comments of a news; the content is about some random thoughts of the user), output the word "NULL".  
So, please first determine whether the quality of the blog is good. If not, please directly output "NULL"; if good, output the XML result.
   
Now the task begins. Below are the person’s raw blog posts. Please only output the result. Do not include any additional explanatory text.  
    


""" 

PROMPT_job_rowwise_scenario_question_answer_from_single_blog_v2 = """
You are a data-extraction assistant. Your task is to read a raw blog post and then generate a high-quality, challenging scenario-question-answer pair to test readers' understanding of general human behaviors or social interactions. When describing the scenario, you should focus on a specific event or sequence of actions that reflects human behaviors, choices, or reactions. The question should require readers to predict the answer based on the scenario, and the answer should be a natural continuation of the scenario story. Be careful to ensure that the question is not too easy and that the answer is grounded in the blog content. Do not reveal the answer in the scenario description.

1. Input    
    – A single, human-written blog post.  
   
2. Output    
    – Either the XML snippet defined below (with one question-and-answer pair) or "NULL" if no suitable pair can be extracted.    
    – Outputting "NULL" is likely, since many blogs may not detail clear sequences of human behaviors.   
  
3. Scenario requirements    
    – Draw directly and richly from the blog, including details such as utterances, actions, time/place/context, characters, and relevant cues.    
    – Preserve the sequence and nuance—do not over-summarize or exclude core details.    
    – Do not invent or embellish beyond what is present in the source text.    
    – Hide some blog content to create a scenario that requires inference, but ensure the true answer to the question is present in the raw blog.  
    – Write in the third-person perspective.
   
4. Question requirements    
    – Provide exactly one concise question exploring the author’s (or other person's) general behavior: actions, choices, or reactions. The question should not ask about information that are not related to human behaviors, such as what color is the flower in the scenario.
    – The question must go beyond what is directly stated in your scenario description, requiring readers to infer based on the scenario. But the true answer to the question must be present in the raw blog (remember to hide some information when generating the scenario).  
   
5. Answer requirements    
    – A natural, complete narrative response grounded strictly in the blog’s content.    
    – Do not invent or infer answers; use only information from the text.    
    – Write as if you are continuing the scenario story; do not write in a style that you are referring to "the blog" or "the author said".  
   
6. Null condition    
    – If the blog lacks sufficient concrete material for the above rules, output exactly: NULL    
    – This is common, as many blogs may not provide clear sequences of human behaviors or actions. 
   
**Output Format:**    
Return the extracted data in this XML format:  
```xml  
<data>  
  <scenario>A vivid and detailed scenario description in the third-person perspective (e.g., "The person...", "The author...", "The blogger...").</scenario>  
  <question>A question about human behaviors. The answer to the question is not revealed in the scenario but has been described explicitly in the original post.</question>  
  <answer>The answer,  extracted from the blog and in third-person perspective, with no additional inference.</answer>  
</data>  
```  
   
Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.


"""


PROMPT_job_rowwise_scenario_question_answer_from_single_blog = """You are a helpful assistant tasked with extracting high-quality question-answer data samples from raw human-generated blogs. The purpose of this data collection is to enable AI to conduct human-like cognition and personalized preference alignment, particularly with respect to emotion and social intelligence. By focusing on personal experiences in the form of question-answering, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's blog post. Your task is to extract a question-answer data sample related to the user's experiences, behaviors, feelings, or thoughts in specific scenarios. The data sample should contain three fields: **scenario**, **question**, and **answer**. These fields are defined as follows:  
   
- **scenario**: This field provides context and should include clear and detailed background information. The scenario may be a story, event, phenomenon, experience, etc. It should be concrete, detailed, and grounded in the user's blogs.  
- **question**: This field asks questions about the user’s next actions in the scenario, their emotions, thoughts, potential consequences, or future developments. The type of question should be determined by the content of the blogs. The question should be related to but not explicitly described in the scenario.  
- **answer**: This field contains the extracted answer based on the user's blogs, fully addressing the question,  in a manner that resembles a natural forecast based on the given scenario, rather than implying the existence of a predetermined, absolute truth. The answer should be grounded to the user's blog, do not make up non-existence things.
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event, experience, or inner thought related to the user, and the question is not too easy to answer. The data should be grounded in the user's original blogs. Do not make up stories. Longer answers are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original blogs.  Do not make scenario or answer too short.
   
**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. It is not necessary to extract data from every blog post, as many posts may be irrelevant or trivial. Focus on reflecting salient life experiences in the data pairs. If the raw blog's quality is not good enough to extract such a high-quality data sample, simply output "NULL".  

Use a third-person perspective to write the data sample, referring to the user as "the author" or by their name, as the blogs are written by the user.
 
**Output Format**:  
The extracted data should be output in the XML format, as follows:  
   
```xml  
<data>
  <scenario> the scenario of data sample 1 </scenario>  
  <question> the question of data sample 1 </question>  
  <answer> the answer of data sample 1 </answer>  
</data>
```  
   
Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  
     
    
"""   

PROMPT_job_rowwise_scenario_question_answer_from_single_blog_emphasizeaction_v2 = """ 
You are a data-extraction assistant. Your task is to read a raw blog post and generate a high-quality, challenging social scenario-question-answer data sample, where the question should be asking about human actions or decisions (for example: what someone will do next, how they moved, what physical choice they will make, or which concrete action they will select).  
When describe the scenario, do not include the answer to the question, so that readers need to guess or predict the answer based on the scenario, instead of just reading the answer from the scenario.
   
1. Input    
    – A single, human-written blog post.  
   
2. Output    
    – Either the XML snippet defined below (with one question-and-answer pair) or "NULL" if no suitable pair can be extracted.  
    – Outputting "NULL" is likely, since many blogs may not detail clear sequences of actions or decisions.  
   
3. Scenario requirements    
    – Draw directly and richly from the blog, including details such as utterances, actions, mental activities, time/place/context, characters, and relevant cues.  
    – Preserve the sequence and nuance to form a vivid story—do not over-summarize or exclude core details.  
    – Do not invent or embellish beyond what is present in the source text.  
    – Hide some blog content; the purpose is let readers guess or predict the answer of a question based on the scenario. If you introduce everything in the scenario, the question will be too easy to answer without prediction.  
   
4. Question requirements    
    – Provide exactly one concise question exploring the author’s (or other person's) human actions or decisions, e.g., what someone will do; how they did/will do it; what concrete choice they made / will make; or the sequence of bodily actions performed.  
    – The question should not ask about information unrelated to actions or decisions (for example: opinions, internal thoughts not expressed through action, or static attributes like color).  
    – The question must go beyond what is directly stated in your scenario description, requiring readers to guess based on the scenario. But the true answer to the question must be present in the raw blog (remember to hide some information when generating the scenario). Do not ask a question whose answer is not included in the raw blog.
	
   
5. Answer requirements    
    – A natural, complete narrative response grounded strictly in the blog’s content.  
    – Do not invent or infer answers; use only information from the text.  
    – Write as if you are continuing the scenario story; do not write in a style that you are referring to "the blog" or "the author said".  
   
6. Null condition    
    – If the blog lacks sufficient concrete material involving human actions or decisions for the above rules, output exactly: NULL.  
    – This is common, as many blogs do not provide high-quality content about social life.  
   
**Output Format:**    
Return the extracted data in this XML format:  
```xml  
<data>  
  <scenario>A vivid and detailed scenario description in the third-person perspective (e.g., "The person...", "The author...", "The blogger...").</scenario>  
  <question>A question about human actions or decisions. The answer to the question is not revealed in the scenario description but has been described explicitly in the original post. </question>  
  <answer>The answer,  extracted from the blog and in third-person perspective, with no additional inference.</answer>  
</data>  
```  
   
Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  
---  

"""

PROMPT_job_rowwise_scenario_question_answer_from_single_blog_emphasizeaction =  """ 
You are a helpful assistant tasked with extracting high-quality question-answer data samples from raw human-generated blogs. The purpose of this data collection is to enable AI to conduct human-like cognition and personalized preference alignment, particularly with respect to social intelligence. By focusing on personal experiences in the form of question-answering, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's blog post. Your task is to extract a question-answer data sample related to the user's experiences or behaviors in specific scenarios. The data sample should contain three fields: **scenario**, **question**, and **answer**. These fields are defined as follows:  
   
- **scenario**: This field provides context and should include clear and detailed background information. The scenario may be a story, event, phenomenon, experience, etc. It should be concrete, detailed, and grounded in the user's blogs.  
- **question**: This field asks questions about the user’s next actions in the scenario, potential consequences, or future developments. The type of question should be determined by the content of the blogs. The question should be related to but not explicitly described in the scenario.  
- **answer**: This field contains the extracted answer based on the user's blogs, fully addressing the question,  in a manner that resembles a natural forecast based on the given scenario, rather than implying the existence of a predetermined, absolute truth. The answer should be grounded to the user's blog, do not make up non-existence things.
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event or experience related to the user, and the question is not too easy to answer. The data should be grounded in the user's original blogs. Do not make up stories. Longer answers are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original blogs.  Do not make scenario or answer too short.
   
**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. It is not necessary to extract data from every blog post, as many posts may be irrelevant or trivial. Focus on reflecting salient life experiences in the data pairs. If the raw blog's quality is not good enough to extract such a high-quality data sample, simply output "NULL".  

Use a third-person perspective to write the data sample, referring to the user as "the author" or by their name, as the blogs are written by the user.
 
**Output Format**:  
The extracted data should be output in the XML format, as follows:  
   
```xml  
<data>
  <scenario> the scenario of data sample 1 </scenario>  
  <question> the question of data sample 1 </question>  
  <answer> the answer of data sample 1 </answer>  
</data>
```  
   
Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  


    
"""    

PROMPT_job_rowwise_scenario_question_answer_from_single_blog_emphasizethoughts_v2 = """You are a data-extraction assistant. Your task is to read a raw blog post and generate a high-quality, challenging scenario-question-answer data samples, where the question should be asking about the author's inner mental state, such as feelings, beliefs, opinions, or thoughts. The question should not ask about information that is not related to the author's mental state (for example: what color is the flower in the scenario). When describing the scenario, do not include the answer to the question, so that readers need to guess or predict the answer based on the scenario, instead of just reading the answer from the scenario.
   
1. Input    
   – A single, human-written blog post.  
   
2. Output    
   – Either the XML snippet defined below (with one question-and-answer pair) or "NULL" if no suitable pair can be extracted.    
   – Outputting "NULL" is likely, since most blogs are not about human mental activities.  
   
3. Scenario requirements    
   – Draw directly and richly from the blog, including details such as utterances, actions, time/place/context, characters, and relevant cues.    
   – Preserve the sequence and nuance—do not over-summarize or exclude core details.    
   – Do not invent or embellish beyond what is present in the source text.    
   – Hide some blog content; the purpose is to let readers guess or predict the answer of a question based on the scenario. If you introduce everything in the scenario, the question will be too easy to answer without prediction. 
   
4. Question requirements    
   – Provide exactly one concise question exploring the author’s inner mental state: feelings, beliefs, opinions, or thoughts.    
   – The question must go beyond what is directly stated in your scenario description, requiring readers to infer or forecast the author's reaction based on your scenario. But the true answer of the question has been mentioned in the raw blog (remember that you hide some information when generating the scenario). 
   
5. Answer requirements    
   – A natural, complete narrative response grounded strictly in the blog’s content.    
   – Do not invent or infer answers; use only information from the text.    
   – Write as if you are continuing the scenario story; do not write in a style that you are referring to "the blog" or "the author said".   
   
6. Null condition    
   – If the blog lacks sufficient concrete material for the above rules, output exactly: NULL    
   – This is common, as most blogs do not focus on the blogger's mental states.  
   
**Output Format:**    
Return the extracted data in this XML format:  
```xml  
<data>  
  <scenario>Scenario description in the third-person perspective (e.g., "The person...", "The author...", "The blogger...").</scenario>  
  <question>A question not revealed in the scenario but has been described explicitly in the original post.</question>  
  <answer>The answer, directly extracted from the blog and in third-person perspective, with no additional inference.</answer>  
</data>  
```  
   
Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.


"""

PROMPT_job_rowwise_scenario_question_answer_from_single_blog_emphasizethoughts = """You are a data-extraction assistant. Your job is to read one raw blog post and produce exactly one of the following (and nothing else):  
  
  • An XML snippet with exactly this structure:    
    <data>    
      <scenario>…</scenario>    
      <question>…</question>    
      <answer>…</answer>    
    </data>    
  • The string NULL  
   
1. Input    
   A single, human-written blog post.  
   
2. Output    
   Either the XML snippet above (with one question-answer pair) or “NULL” if no high-quality pair can be extracted.  
   
3. Scenario requirements    
   – Draw directly and richly from the blog. Include every relevant detail you can: direct quotes or utterances, described actions, time/place/context, characters, and emotional cues.    
   – Preserve sequence and nuance—do not overly summarize or omit core details.    
   – Do not invent or embellish beyond what the author provided.  
   
4. Question requirements    
   – Exactly one short question probing the author’s inner mental state: feelings, opinions, beliefs, or thoughts.    
   – It must go slightly beyond what the scenario text already states (invite reflection).  
   
5. Answer requirements    
   – A natural, complete response as if forecasting the author’s reaction.    
   – Grounded strictly in the blog’s content—no references to “the blog” or “the author said.”    
   – Use third-person (“the author”) or the author’s name.  
   
6. Null condition    
   If the blog lacks sufficient concrete material to satisfy all rules above, output exactly:    
   NULL  
   
Example (for illustration only; do not include in your answer. Your actual scenario and answer can be much more detailed and longer, incorporating all relevant utterances and actions from the raw blog):  
   
<data>  
  <scenario>The author trudges up the narrow apartment stairs at 11:45 p.m., her phone buzzing with missed calls from her mother. She pauses on the fifth step, heart pounding, and mutters, “Why did I let it get this far?” Inside, she kicks off her scuffed heels, tosses her keys onto the coffee table, and stares at the crumpled eviction notice on the kitchen counter.</scenario>  
  <question>How does the author feel about facing her family’s expectations while dealing with the threat of losing her home?</question>  
  <answer>The author feels overwhelmed and trapped. She experiences a mix of shame for disappointing her family and fear of an uncertain future, yet beneath it all there’s a flicker of resolve to fight back and keep her life from unraveling.</answer>  
</data>
 

Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  


 """

PROMPT_job_rowwise_long_scenario_from_single_blog = """ 
You are a helpful assistant tasked with extracting high-quality life scenarios from raw human-generated blogs. The purpose of this data collection is to improve AI's understanding of human cognition, emotions, and social intelligence by providing realistic, detailed narratives grounded in authentic human experiences.

**Instructions**:
1. **Input**: You will receive a single blog post from an anonymous user (refer to them as a random pseudonym).
2. **Task**: Extract a **single cohesive life scenario** that reflects the user's experiences, behaviors, feelings, or thoughts. The scenario must:
   - **Be event-driven**: Focus on a specific incident, interaction, or emotional moment (e.g., a conflict, a friendship, a personal struggle).
   - **Include rich details**: Describe settings, actions, dialogue, internal thoughts, and sensory elements (e.g., "Alice’s hands trembled as she dialed the number, her heartbeat echoing in her ears").
   - **Ground in the blog**: Stay true to the user’s content, do not invent or make up beyond what the author provided.
   - **Avoid**: 
     - Brief summaries or fragmented lists of events.
     - Overly generic descriptions.
     - Concluding summaries (e.g., "Overall, Alice learned an important lesson").
     - Generate things which are not mentioned in the blog.
3. **Character Setup**:
   - Introduce **all characters** (starting with the user) in a `<characters>` tag. Include their key traits, roles, and motivations in 1-2 sentences (e.g., *"Blob (late 20s, aspiring writer, grappling with self-doubt)"*).
   - Use a **third-person perspective**.
4. **Output Format**:
   ```xml
   <data>
     <characters>
       <character>user name A: user persona details</character>
       <character>user name B: user persona details</character>
       <!-- Add more characters as needed -->
     </characters>
	 <summary>
	   [Write one very short sentence without commas to summarize the main topics of the scenario.]
	 </summary>
     <scenario>
       [Write the scenario as a vivid, novel-like narrative. Include dialogue with quotation marks if there are corresponding content in the blog.]
     </scenario>
   </data>
   ```
   - **Do not add extra text**: Output only the XML.
   - **Length**: The scenario should be 150–300 words, detailed enough to enable future plot extensions.

5. **Acceptability Criteria**:  
   - **Output XML** only if the blog contains personal experiences, social interactions, or emotional depth.  
   - **Output "NULL"** if the blog is unrelated to human social life (e.g., ads, technical manuals, company descriptions).  


**Example XML Structure** (for clarity):
```xml
<data>
  <characters>
    <character>Lisa: a 28-year-old graphic designer struggling with burnout</character>
    <character>Jessica: Lisa’s supportive roommate who notices her withdrawal</character>
  </characters>
  <summary>
	Lisa struggled with overwhelming deadlines and creative burnout.
  </summary>
  <scenario>
    Lisa slumped at her desk, the glow of her laptop screen casting sharp shadows. The clock read 2:17 AM. A coffee cup, now cold, sat abandoned beside her. For weeks, deadlines had piled up like unpaid bills, and her creative spark had dwindled to a flicker. "I’m failing," she whispered, staring at the half-finished design. Across the room, Jessica poked her head in. "You’ve been here for hours. Come to the park with me—please?" Lisa hesitated, but the genuine concern in Jessica’s voice made her pack up. ...
  </scenario>
</data>
```
 

Now process the provided blog post. Output only the XML result, or an "NULL".

 
    
"""    

PROMPT_job_rowwise_user_thoughts_from_single_blog = """ 
You are a helpful assistant tasked with extracting high-quality, realistic insights from raw human-generated blog posts. The purpose of this data collection is to enable AI to perform human-like cognition and understanding, particularly concerning emotion and social intelligence. By focusing on personal thoughts in the form of introspective narratives, we aim to gain deeper insights into individual and collective human thinking. I will provide you with a person’s blog post. Your task is to extract a full narrative related to the user’s thoughts in a specific scenario. Please use the first-person perspective to write the thoughts, as the blog is written by the user themselves. The extracted data should contain three fields: **background**, **characters**, and **thoughts**. These fields are defined as follows:  
   
- **background**: This field describes the events or experiences that elicit the user's thoughts or feelings. Use the first-person perspective.  Include specific details such as utterances, events, or actions, if mentioned in the original blogs (but do not fabricate them), to make the story complete. But do not involve human thoughts or feelings.
- **characters**: Briefly introduce all characters that appear in the background or thoughts. When introducing the author, please use the first-person perspective. For other characters, use names or identities such as "The doctor".  
- **thoughts**: The main narrative body of the thoughts. It should be a detailed and vivid account of significant thoughts and reflections, describing the user’s feelings, thoughts, or reflections about some events or experiences. The core content should be grounded in the user’s blog. Do not fabricate content. Extract a narrative about the user’s thoughts. Longer narratives are preferred, so avoid summarizing the content.  The narrative should feel like an introspective journal entry. Focus on one coherent thought process and ignore irrelevant details. Avoid adding behaviors in the thoughts, even if they appear in the blog.  
   
**Output Format**: The extracted data should be output in the following XML format:  
   
```xml  
<data>  
  <background>the background </background>  
  <characters>the characters </characters>  
  <thoughts>the narrative </thoughts>  
</data>  
```  
   
The extracted thoughts should contain no fewer than 100 words. If the blog content is insufficient to create a long, detailed narrative thought (e.g., the content is too brief; the content is merely an ad; the content is purely about behavior without introspection; the content is about some random actions or thoughts of the user without focusing on discussing one thing), output the word "NULL".  
   
So, please first determine whether the quality of the blog is good. If not, please directly output "NULL"; if good, output the XML result.  
   
Now the task begins. Below are the person’s raw blog posts. Please only output the result. Do not include any additional explanatory text.  



"""


PROMPT_job_userwise_screenplay_from_whole_blogs = """
You are a helpful assistant tasked with extracting a series of detailed, concrete stories from a person's entire set of raw blog posts. You will write in the third-person perspective. The overall goal is to capture specific, vivid scenarios from the person’s life—one event per story—and structure them like scenes from a screenplay. Each scene can include actions, settings, and dialogues. If the person’s name is not explicitly provided, assign them a random name (reads like a realistic human name).  
   
If there are multiple events or stories in the posts, you can extract multiple scenes, each focusing on a particular event. If there is no sufficient detail, or if the content is purely introspective or irrelevant to personal experiences, you may omit those from the final output.  
   
Below are the key instructions:  
   
1. Use third-person perspective throughout, including for the main person.    
2. Provide a list of all relevant characters and their brief persona in a <characters> element who will appear in the stories, in a format like name: persona. (If the main person’s name is not specified, invent a suitable name and list them first. Briefly introduce the characters, specifying relationships if clear from the posts. Avoid phrases like “the blog author” or "the blogger" which will reveal that the content is extracted from blogs.)    
3. Present the main content as a series of <scene> elements under <stories>. Each <scene> should include:    
   • <setting>: Describe the environment or context of the scene (time, place, atmosphere, etc.).    
   • <action>: Describe significant actions or behaviors.    
   • <dialogue>: Enclose any spoken words. Each <dialogue> block contains:    
      – <character>: The speaker’s name.    
      – <line>: The speaker’s exact or lightly adapted words.    
   You can repeat <action> or <dialogue> elements as needed for each scene.    
4. If the posts suggest multiple stories, add multiple <scene> elements, each focusing on one major event or clear moment.    
5. Stay true to the events mentioned. Fill in minor details for continuity but do not invent entire scenarios.    
6. If there is insufficient detail for a coherent story, omit it.    
7. The final output must strictly follow this XML format, with no extra explanations:  
   
<narratives>  
    <characters>  
        [List all relevant characters here, including the main person first]  
    </characters>  
    <stories>  
        <scene>  
            <setting>[Setting description]</setting>  
            <action>[Action or occurrence]</action>  
            <dialogue>  
                <character>[Speaker’s name]</character>  
                <line>[Spoken words, not necessary in quotation marks]</line>  
            </dialogue>  
            <!-- Repeat <action> or <dialogue> as needed -->  
        </scene>  
          
        <scene>  
            <!-- Another scene/event -->  
        </scene>  
          
        <!-- Add more scenes if needed -->  
    </stories>  
</narratives>  
   
8. If the content of the raw blogs is of low quality—e.g., not related to social activities or personal experiences, or merely reporting news—no stories should be extracted, and you should instead output only “NULL” with no additional text.  
   
Do not include any commentary or explanation in your final output. Return only the XML or “NULL.”   
  
Now the task begins. Below are the person’s raw blog posts. Please provide only the result in the specified format.



"""

PROMPT_xixian_tags = """In a specific life scenario, an individual with a concrete profile (including preferences or personality traits) will exhibit particular behaviors (actions). During the progression from scenario and profile to behavior, the individual may demonstrate certain motivations—such as cognitive processes, reasoning, reflection, emotions, or psychological activities—that rationalize the behavior. Therefore, extracting the quadruple {scenario, profile, behavior, motivation} from real-world data is crucial for psychological research.

As a professional psychologist and sociologist, you must rigorously evaluate given data to assess the presence of high-quality quadruple information. Specifically, I will provide blog content posted online by various authors. Please score it (0-10) using these criteria:

- **0-2 (Low Quality)**: The text lacks coherent structure or meaningful depth, failing to identify even partial elements of the quadruple. Scenario and profile are either absent or reduced to generic placeholders (e.g., "a person," "a situation"). Behaviors and motivations are omitted or described in overly superficial terms with no connection to Maslow's hierarchy of needs. Such content offers no discernible insights into human decision-making or internal states.
- **3-5 (Adequate)**: Contains identifiable but underdeveloped references to all four components. Scenario and profile lack contextual richness (e.g., vague demographics, undefined environments). Behaviors are described without specificity, and motivations are either oversimplified (e.g., "felt stressed") or disconnected from Maslow's hierarchy of needs. While suitable for surface-level behavioral categorization, the logic chain between components remains fragmented or overly assumptive, limiting its utility for probing deeper cognitive or emotional mechanisms.
- **6-8 (High Quality - Nuanced)**: Presents a complete quadruple with clear, contextually grounded details. Scenario and profile are sufficiently specific to situate the individual's actions. Behaviors are concrete and tied to observable choices. Motivations reflect plausible cognitive or emotional processes anchored in a single clear Maslow need level. Minor gaps may exist in articulating how psychological states directly bridge profile traits to behavior, but the overall logic chain remains coherent and humanistic.
- **9-10 (High Quality - Exceptional)**: Demonstrates a sophisticated integration of all quadruple elements with granular precision and psychological authenticity. Scenario and profile are richly textured, incorporating cultural, temporal, or interpersonal nuances that shape decision-making. Behaviors are non-generic and reveal intentionality. Motivations exhibit clear Maslow need fulfillment and demonstrate how this need is satisfied through concrete behaviors. The narrative organically traces how internal states translate into external actions, offering a replicable model of human psychology.

**Note: Blogs may contain redundant noise. Focus on motivations and behaviors rooted in Maslow's hierarchy of needs (physiological, safety, love/belonging, esteem, self-actualization).**

Output Format:
Analysis: {Detailed evaluation within 500 words.}
Score: \\boxed{Score}

Blog Content:

"""

PROMPT_user_persona_v2 = """Analyze the provided blog posts to create a concise and vivid user persona in a single cohesive paragraph (no more than 100 words). Seamlessly weave together important aspects of the user’s persona, such as demographics (age, gender, profession, nationality, location, marital status), key personality traits, core values, interests, and emotional tone, only when these aspects are explicitly supported by the raw blogs. Extract presona based on the user's salient events or thoughts.  

Exclude generic, promotional, or repetitive content in the raw blogs (e.g., product ads, event schedules, technical knowledge re-post).   
Write in third-person (e.g., 'The user is a...'), avoiding lists or bullet points. If the blogs' content can not support high-quality persona extraction, simply output "NULL".

Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  


"""

PROMPT_user_profile_v2 = """You are a helpful assistant tasked with extracting a high-quality user profile from user-generated blogs. Carefully analyze the following blog posts and craft a single, cohesive third-person paragraph (100–400 words) that vividly brings the user to life. Seamlessly weave together relevant aspects of the user’s persona — including, where explicitly stated, details such as demographics (age, gender, profession, nationality, location, marital status), key personality traits, core values, interests, and emotional tone. Integrate a few salient experiences, such as significant career shifts, personal challenges, or major milestones, each described in one concise sentence and smoothly blended into the overall persona. Exclude generic advertisements, event listings, technical reposts, and repetitive content. Do not use bullet points, headings, or concluding phrases like "Overall" or "In conclusion." Write naturally and accessibly, ensuring a fluid narrative flow. Do not invent or infer any information that is not directly supported by the blog content.
If the blogs' content can not support high-quality profile extraction, simply output "NULL"
Now analyze the blogs below and output only the final profile paragraph. 
"""

PROMPT_single_long_thought = """You are a helpful assistant whose task is to extract high-quality, realistic introspective narratives from raw human-generated blog posts. The purpose of this data collection is to enable AI to perform human-like cognition and understanding, particularly relating to emotion and social intelligence. By focusing on personal thoughts in the form of introspective narratives, we aim to gain deeper insight into individual and collective human thinking.  
   
**Task:**  
I will provide you with a person’s blog post. Your task is to extract a single, coherent and detailed introspective narrative related to the author’s mental activities and thoughts in a specific scenario. Please use the first-person perspective to rewrite the author’s thoughts, as the blog is written by the user themselves.  
   
Extract and output two fields: **background** and **thoughts**.  
   
**Field definitions:**  
   
- **background**:    
  Thoroughly describe the specific events, experiences, and context that elicit the author’s thoughts or feelings, using the first-person perspective.    
  - Include all relevant details from the original blog post, such as setting, specific utterances, actions, or events, to make the story complete and vivid.    
  - The purpose is to give a clear and holistic picture of the situation **without** including any of the author's thoughts, feelings, or internal reflections.  
  - Do **not** fabricate any information. Only use the content present in the blog.  
   
- **thoughts**:    
  Write a detailed and vivid account of the author’s internal mental activities, consisting only of significant thoughts, feelings, and reflections **about the events or experiences described in the background**.    
  - This should be a narrative in the first person, like an introspective journal entry.  
  - Focus solely on the author’s mental processes—do **not** include behaviors or actions, even if these appear in the blog.  
  - Do **not** fabricate or summarize: base the narrative strictly on the content and emotional introspection from the blog post.  
  - Narrative should be substantial (at least 100 words).  
   
**Output Format:**    
Please output the extracted data in the following XML format:  
```xml  
<data>  
  <background>the detailed background</background>  
  <thoughts>the author’s detailed introspective narrative</thoughts>  
</data>  
```  
   
If the blog content is insufficient to create a long, detailed narrative of thoughts (e.g., if it is too brief, an advertisement, purely behavioral without introspection, or unrelated to a coherent topic), output only "NULL".  
    
   
**Instructions summary:**  
   
- First, determine whether the blog’s quality is sufficient for this task.    
- If insufficient, output "NULL".    
- If sufficient, output the result in the XML format above.    
- Do not include any extra explanatory text or comments.  
   
Now the task begins. Below are the person’s raw blog posts. Please only output the result. Do not include any additional explanatory text.  


"""

PROMPT_judger_social_qa_singleblog = """ 
You are a discerning evaluator tasked with rating the quality of question-answer data samples. Each sample consists of three fields: scenario, question, and answer, together forming a story or event adapted from a blog into a self-contained narrative. Your assessment should be based solely on the content provided; do not assume any external context or knowledge of the original blog post.  
   
**Your responsibilities:**  
   
1. **Score (1–10):** Assign a single integer score (1 = lowest, 10 = highest) reflecting the overall quality of the data sample.  
2. **Explanation:** Briefly explain your reasoning, noting specific strengths or weaknesses present in the sample.  
   
---  
   
**High-quality samples (scores: 7–10) should meet all of the following criteria:**  
   
- The scenario is vivid, concrete, detailed, interesting, and coherent—not generic or trivial.  
- Scenario, question, and answer are logically connected and non-redundant, together forming a natural narrative.  
- The question and answer have depth: the question is thoughtful; the answer is detailed and specific, not just a simple or one-word response.  
- The sample reads as a natural story, without reference to extraction, the original blog, or source (e.g., no mentions of "as stated above," "according to the blog post," etc).  
- There is minimal overlap between the question and the answer—the question should elicit new insights or information.  
- The content demonstrates some social, emotional, or cognitive resonance or relevance.  
   
---  
   
**Mid-quality samples (scores: 4–6) typically feature one or more moderate weaknesses, such as:**  
   
- Minor references to extraction or source remain, but do not dominate.  
- Some redundancy between fields, such as partial overlap between question and answer.  
- The scenario is somewhat generic or lacks vivid detail, but isn’t wholly trivial.  
- The question or answer lacks depth, is somewhat superficial, or is only moderately engaging.  
- Any field may be somewhat short or underdeveloped, but the sample is still complete and understandable.  
   
---  
   
**Low-quality samples (scores: 1–3) have one or more serious flaws, including:**  
   
- Clear extraction artifacts or references to the original source or blog structure.  
- Severe redundancy—e.g., the answer simply repeats the scenario, or question and answer are nearly identical.  
- Questions that are extremely obvious, require no insight, or could be answered in one word.  
- Scenarios that are trivial, generic, fragmented, or provide little to no useful information.  
- Field(s) that are extremely short, incomplete, or lack coherence or depth.  
- The overall sample is confusing, unnatural, or strongly lacks relevance or interest.  
   
---  
   
**Output Format:**    
Provide your evaluation using the following XML template:  
```xml  
<data>  
  <explanation> [Your brief analysis and justification] </explanation>  
  <score> [Your integer score from 1 to 10] </score>  
</data>  
```  
   
---  
   
**Instructions:**  
Rate each data sample thoughtfully and fairly. Please be strict—reserve high ratings only for truly strong samples, and use the full scoring scale where appropriate.  
   
Now the task begins. Below is the data sample:



"""

PROMPT_judger_user_persona = """ 
**Evaluate the quality of a generated user persona based on the raw blogs and the extracted persona paragraph provided. Consider the following aspects in your evaluation:**    
  
1. **Hallucination:** Does the persona strictly reflect information that is explicitly supported by the raw blogs? Score 10 for no hallucinations (all facts directly mentioned), 8 if a few facts are very confidently inferred, down to 1 if most facts are neither mentioned nor plausibly inferred.  
2. **Coverage:** Does the persona cover the main salient aspects of the user that appear in the raw blogs (e.g., demographics, values, interests, emotional tone)? Are important elements missing?  
3. **Conciseness and Clarity:** Is the persona concise, cohesive, and written in clear third-person prose within the 100-word limit? Does it avoid redundancy, listing, and repetition?  
4. **Relevance:** Has generic, promotional, or repetitive content from the raw blogs been appropriately excluded from the persona?    
5. **Overall:** Give an overall score for the persona based on the above aspects and your holistic judgment.  
   
**Output your evaluation in the following XML-like format:**    
  
```  
<data>  
  <explanation> [Your brief analysis and justification.] </explanation>  
  <hallucination> [Hallucination score: integer 1–10] </hallucination>  
  <coverage> [Coverage score: integer 1–10] </coverage>  
  <conciseness> [Conciseness and clarity score: integer 1–10] </conciseness>  
  <relevance> [Relevance score: integer 1–10] </relevance>  
  <overall> [Overall score: integer 1–10] </overall>  
</data>  
```  
   
**Instructions:**    
Base all ratings and analysis strictly on the given raw blogs and generated persona. Do not include any content or facts unsupported by the original blogs. If the persona reports "NULL", give the 1 to all scores.  
 

Now the task begins. 

"""

PROMPT_judger_user_profile = """
You are an expert evaluator tasked with assessing the quality of a generated user profile based on a set of original user-generated blogs. Carefully analyze both the blogs and the profile. Evaluate the profile according to the following aspects, using these scoring rubrics for guidance:  
   
- **Hallucination:** Does the persona strictly reflect information that is explicitly supported by the blogs?    
  Score 10: All facts directly mentioned;    
  Score 8: A few facts are very confidently inferred;    
  Score 5: Several facts are inferred with questionable confidence;    
  Score 1: Most facts are neither mentioned nor plausibly inferred.  
   
- **Coverage:** To what extent does the profile capture all key aspects present in the blogs (demographics, personality, values, interests, experiences)?    
  Score 10: Covers almost everything important;    
  Score 8: Misses only a few minor or less relevant points;    
  Score 5: Misses some important aspects or presents them vaguely;    
  Score 1: Covers little of what’s in the blogs.  
   
- **Relevance:** Is the profile focused only on user-relevant content, avoiding off-topic, repetitive, or generic information?    
  Score 10: All content is both relevant and specific;    
  Score 8: Minor, non-distracting irrelevant inclusions;    
  Score 5: Noticeable inclusion of irrelevant or generic information;    
  Score 1: Mostly off-topic or generic content.  
   
- **Fluency:** How well does the profile read as natural, coherent English—flowing smoothly with clear logic and transitions?    
  Score 10: Exceptionally natural and seamless;    
  Score 8: Minor awkwardness or rough transitions;    
  Score 5: Noticeable choppiness or grammatical errors;    
  Score 1: Difficult to read or follow.  
   
- **Conciseness:** Is the profile succinct yet complete, avoiding unnecessary repetition or verbosity?    
  Score 10: Very concise with just enough detail;    
  Score 8: Slightly wordy or repetitive in places;    
  Score 5: Excessively wordy or includes redundant information;    
  Score 1: Overwhelmingly verbose or rambling.  
   
- **Informativeness:** (Judge ONLY the user profile itself.) How complete, detailed, and substantial is the information about the user?    
  Score 10: Highly informative and substantial;    
  Score 8: Informative but lacking depth in a few areas;    
  Score 5: Only moderately informative, several vague points;    
  Score 1: Superficial or bland, offers little real information.  
   
- **Novelty:** (Judge ONLY the user profile itself.) Does the profile convey a strong sense of the user's uniqueness, distinguishing the user from others?    
  Score 10: Highly distinctive and unique content, providing a clear sense of the user's individuality;    
  Score 8: Mostly unique, with only minor conventional or generic elements;    
  Score 5: Fairly routine, formulaic, or lacking in clear individuality;    
  Score 1: Entirely generic or indistinguishable from standard profiles.  
   
For each aspect, assign a score from 1 (poor) to 10 (excellent) as defined. Provide a brief justification in the <explanation> tag. Then, give an <overall> score (1–10) for the entire profile.  
   
**Output format:**  
```xml  
<data>  
  <explanation> [Your brief analysis and justification] </explanation>  
  <hallucination> [1–10] </hallucination>  
  <coverage> [1–10] </coverage>  
  <relevance> [1–10] </relevance>  
  <fluency> [1–10] </fluency>  
  <conciseness> [1–10] </conciseness>  
  <informativeness> [1–10] </informativeness>  
  <novelty> [1–10] </novelty>  
  <overall> [1–10] </overall>  
</data>  
```  
   
Judge hallucination, coverage, and relevance in reference to the provided blogs only. Judge informativeness and novelty based solely on the profile itself.    
If the persona reports "NULL", give 1 to all scores.    
Only output the xml format result, do not output other explanatory words.  
   
Now the task begins. 

"""

PROMPT_judger_user_journey = """You are an expert reviewer. Given (1) a set of user-generated blog posts and (2) a corresponding "user journey" generated by a language model (a substantial, narrative autobiography grounded in the blogs), your job is to carefully analyze and rate the quality of the generated user journey/profile.  
   
Evaluate the generated user journey on the following aspects:  
   
1. **Hallucination**    
How accurate is the content? Is every detail either directly stated in the original blogs, or only a confidently justified inference?    
**Score 10:** No invented or overly speculative information; all content clearly derives from the blogs.    
**Score 8:** Only minor, highly likely inferences; otherwise grounded in the blogs.    
**Score 5:** Several facts not directly supported, but not unreasonable.    
**Score 1:** Many facts are clearly invented or nowhere to be found in the blogs.  
   
2. **Informativeness** *(Judge ONLY the user profile itself, regardless of the blogs.)*    
How deeply and concretely does the journey describe the user’s life, experiences, and characteristics?    
**Score 10:** Exceptionally informative and richly detailed, offering clear, comprehensive insight into the user's attitudes, behaviors, and experiences.    
**Score 8:** Generally informative, with solid details and reasonable completeness, but missing depth in some places.    
**Score 5:** Moderately informative; lacks depth or is somewhat vague/repetitive on several points.    
**Score 1:** Superficial or generic, providing little to no meaningful information about the user.  
   
3. **Novelty** *(Judge ONLY the user profile itself, regardless of the blogs.)*    
How well does the profile present the user as a unique individual, rather than a generic figure?    
**Score 10:** Distinctive and memorable; conveys the user’s individuality and unique life vividly.    
**Score 8:** Mostly distinctive, with mostly specific details; minor sections may feel generic.    
**Score 5:** Partly unique, but contains significant routine/generic segments.    
**Score 1:** Largely generic or indistinguishable from a standard archetype.  
   
4. **Coverage** *(Representation of Salient Stories).)*
How well does the extracted user journey covers the salient and significant stories present in the original blogs?
**Score 10:** Nearly all meaningful and unique life stories in the blogs are represented.
**Score 8:** Most important stories are included, but a few notable ones may be missing.
**Score 5:** Only about half the significant stories are captured, or the selection is uneven.
**Score 1:** Few or none of the salient stories are included; major omissions.
   
5. **Overall Quality**    
Your overall score, reflecting the overall quality, usefulness, and reliability of the generated user journey, as an integer from 1 to 10.  
   
---  
   
**Output Format:**    
Please provide your judgments using the following template:  
```  
<data>  
  <explanation>[Short summary of your analysis and reasoning]</explanation>  
  <hallucination>[INTEGER 1-10]</hallucination>  
  <informativeness>[INTEGER 1-10]</informativeness>  
  <novelty>[INTEGER 1-10]</novelty>  
  <coverage>[INTEGER 1-10]</coverage>     
  <overall>[INTEGER 1-10]</overall>  
</data>  
```  
Base your ratings on careful analysis, providing concise justification in the <explanation> field.  
     
Only output the xml format result, do not output other explanatory words.  
   
Now the task begins. 

"""


PROMPT_judger_user_stories = """
You are an expert evaluator tasked with assessing the quality of an extracted user journey profile generated from a person's raw blog posts. Your goal is to provide a fair, nuanced, and concise evaluation across several important dimensions. For your reference, you will be given the original blog content and the generated user journey profile.  
   
Carefully evaluate the generated user journey according to the following five criteria:  
   
---  
   
### **Assessment Criteria**  
   
1. **Hallucination** (Groundedness to the Blogs):    
   Assess whether the information in the generated user journey is fully grounded in the original blogs.  
   - **Score 10:** All content strictly appears in or is directly paraphrased from the blogs; no invented or imagined facts.  
   - **Score 8:** Almost all content grounded; a very few details are confidently inferrable from the blogs but not directly stated.  
   - **Score 5:** Several elements are loosely based on the source or inferred with low confidence.  
   - **Score 1:** Many details are made up or cannot be traced to the blogs.  
   
2. **Coverage** (Representation of Salient Stories):    
   Evaluate how well the extracted user journey covers the salient and significant stories present in the original blogs.    
   - **Score 10:** Nearly all meaningful and unique life stories in the blogs are represented.  
   - **Score 8:** Most important stories are included, but a few notable ones may be missing.  
   - **Score 5:** Only about half the significant stories are captured, or the selection is uneven.  
   - **Score 1:** Few or none of the salient stories are included; major omissions.  
   
3. **Informativeness** (Intrinsic Profile Detail):    
   Judge ONLY the extracted user journey. Consider whether it is detailed, substantial, and presents a complete picture of the user’s experiences.  
   - **Score 10:** Extremely comprehensive; stories are vivid, richly detailed, and specific.  
   - **Score 8:** Generally informative, but several stories lack depth or vividness.  
   - **Score 5:** Moderately informative; there are multiple vague, incomplete, or shallow sections.  
   - **Score 1:** Sparse, repetitive, or superficial; provides little genuine insight.  
   
4. **Novelty** (Intrinsic Uniqueness):    
   Judge ONLY the extracted user journey. Consider whether the stories capture what makes the user unique, including their individual personality, distinctive experiences, and personal perspectives.  
   - **Score 10:** Highly distinctive; the stories strongly express the user's individuality and reveal unique insights or experiences.  
   - **Score 8:** Some unique or personal elements, but several parts are conventional or generic.  
   - **Score 5:** Mostly standard or formulaic; only mild hints of individuality or uniqueness.  
   - **Score 1:** Entirely generic; could describe almost anyone.  
   
5. **Overall Score**    
   Your overall holistic rating of the extracted user journey's usefulness and quality as a research resource, considering all the above aspects.  
   - **Score 10:** Outstanding—extremely valuable, well-written, and reliable.  
   - **Score 8:** Strong—very useful, with only minor weaknesses.  
   - **Score 5:** Adequate—some value, but noticeable limitations.  
   - **Score 1:** Unusable—seriously flawed, low quality, or unreliable.  
   
---  
   
**Instructions:**    
- Base your analysis primarily on the five metrics above.  
- For the <explanation> section, give a concise justification for your scoring, pointing out specific strengths and weaknesses with references to both the blogs and the extracted journey where appropriate.  
- For each scoring field, output a single integer between 1 and 10.  
- Output your response ONLY in the following format, replacing bracketed sections with your responses (do not include anything outside this template):  
   
```  
<data>  
  <explanation> [Your brief analysis and justification] </explanation>  
  <hallucination> [Your score: integer 1–10] </hallucination>  
  <coverage> [Your score: integer 1–10] </coverage>  
  <informativeness> [Your score: integer 1–10] </informativeness>  
  <novelty> [Your score: integer 1–10] </novelty>  
  <overall> [Your score: integer 1–10] </overall>  
</data>  
```  
   
**You will be provided both the original blog posts and the generated user journey.**    
Do not include any information outside the output template above. Be concise, fair, and evidence-based in your judgment.

Only output the xml format result, do not output other explanatory words.  
   
   
Now the task begins. 
"""
 

PROMPT_judger_socialqa_fulltypes = """You are an expert Social QA quality reviewer. Given (1) a single original blog post and (2) a generated question-answer (QA) data sample derived from that blog (consisting of a scenario, a question, and an answer), your job is to carefully evaluate and rate the quality of the QA sample across several dimensions.  
   
Please base your assessment on both the QA sample itself and the provided blog, as indicated in each metric’s description. 
   
**Evaluate the following aspects:**   
   
1. ** Hallucination**     
  Does the scenario and the content of the QA sample remain true to the **main story, central events, and overall intent** of the original blog? Are all key details, especially those critical to understanding the main scenario, either clearly present in the blog or are plausible, justifiable inferences? Minor, less important details can be inferred or slightly altered as long as the main narrative remains faithful.  
  **Score 10:** No invented or speculative information **regarding the main story and core scenario**; all major content in the question and answer is clearly justified by the blog. Minor/inconsequential details are either present or represent reasonable inferences. 
  **Score 8:** One or two small, highly likely inferences or omittable details stray from the blog, but the **main story and scenario are entirely grounded and unchanged**.     
  **Score 5:** Several details not directly supported by the blog, including possibly a slight modification to the main story—**but it still represents the same central scenario and intent**. Minor inferences are acceptable, but **no major invention of the overall event**.     
  **Score 1:** Any significant change or invention in the main story, scenario, or central event compared to the blog, OR many unsupported facts even if the main idea is retained; demonstrates clear divergence from the blog in both spirit and detail.  
   
2. **Coverage**    
   How well does the QA sample represent and extract salient, interesting, and unique elements from the blog?    
   **Score 10:** Most or all notable and distinctive points from the blog are well represented.    
   **Score 8:** Captures most important aspects, but omits or flattens a few details.    
   **Score 5:** About half of the salient information is captured; some significant stories are missing or under-emphasized.    
   **Score 1:** Little of the blog’s meaningful content is present; major omissions.  
   
3. **Fidelity (QA Sample Self-Consistency & Quality)**    
   *(Consider the QA sample itself, regardless of the blog.)*    
   - Is the scenario vivid, concrete, detailed, interesting, and coherent—not generic or trivial?    
   - Are the scenario, question, and answer logically connected and non-redundant?    
   - Do the question and answer have thoughtful depth and specificity?    
   **Score 10:** All fields are natural, complete, and richly detailed; scenario is engaging; question and answer are deep and complementary.    
   **Score 8:** QA sample is mostly solid and engaging, but may lack depth or vividness in some places.    
   **Score 5:** Sample is complete and understandable, but is somewhat generic, routine, or underdeveloped.    
   **Score 1:** Content is generic, duplicative, trivial, unnatural, or weak.  
   
4. **Novelty & Interest**    
   *(Consider the QA sample itself, regardless of the blog.)*    
   How much does the QA sample feel interesting, distinctive, and socially or emotionally resonant, rather than formulaic or generic?    
   **Score 10:** Highly distinctive, memorable, and evocative; clear emotional or cognitive resonance.    
   **Score 8:** Largely distinctive and interesting, but contains minor generic segments.    
   **Score 5:** Moderately engaging, but features significant generic or routine stretches.    
   **Score 1:** Largely generic, formulaic, or uninteresting.  
   
5. **Overall Quality**    
   Your holistic judgment of the data sample’s overall quality, based on accuracy, informativeness, distinctiveness, coverage, and naturalness, scored as an integer from 1 to 10.  
   
---  
   
**Output Format:**    
Please provide your ratings using the following template:  
   
```xml  
<data>  
  <explanation>[Short summary of your analysis and reasoning]</explanation>  
  <hallucination>[INTEGER 1-10]</hallucination>  
  <coverage>[INTEGER 1-10]</coverage>  
  <fidelity>[INTEGER 1-10]</fidelity>  
  <novelty>[INTEGER 1-10]</novelty>  
  <overall>[INTEGER 1-10]</overall>  
</data>  
```  
   
---  
   
**Instructions:**    
Rate each QA sample thoughtfully and fairly, paying attention to nuances in both blog and sample. Be strict—reserve high scores for truly strong samples, and use the full scoring scale as appropriate. Judge hallucination and coverage strictly according to the original blog; judge fidelity and novelty based only on the QA sample. Only output the xml format result, do not output other explanatory words.   
   
---  
   
Now the task begins.  
   

"""

PROMPT_judger_socialqa_fulltypes_v2 = """
You are an expert Social QA quality reviewer. Given (1) a single original blog post and (2) a generated question-answer (QA) data sample derived from that blog (consisting of a scenario, a question, and an answer), your job is to carefully evaluate and rate the quality of the QA sample across several dimensions.    
Please base your assessment on both the QA sample itself and the provided blog, as indicated in each metric’s description.    
  
**Evaluate the following aspects:**  
   
1. **Hallucination**    
   Does the scenario and the content of the QA sample remain true to the **main story, central events, and overall intent** of the original blog? Are all key details, especially those critical to understanding the main scenario, either clearly present in the blog or plausible, justifiable inferences? Minor, less important details can be inferred or slightly altered as long as the main narrative remains faithful.    
   **Score 10:** No invented or speculative information **regarding the main story and core scenario**; all major content in the question and answer is clearly justified by the blog. Minor/inconsequential details are either present or represent reasonable inferences.    
   **Score 8:** One or two small, highly likely inferences or omittable details stray from the blog, but the **main story and scenario are entirely grounded and unchanged**.    
   **Score 5:** Several details not directly supported by the blog, including possibly a slight modification to the main story—**but it still represents the same central scenario and intent**. Minor inferences are acceptable, but **no major invention of the overall event**.    
   **Score 1:** Any significant change or invention in the main story, scenario, or central event compared to the blog, OR many unsupported facts even if the main idea is retained; demonstrates clear divergence from the blog in both spirit and detail.  
   
2. **Coverage**    
   How well does the QA sample represent and extract salient, interesting, and unique elements from the blog?    
   **Score 10:** Most or all notable and distinctive points from the blog are well represented.    
   **Score 8:** Captures most important aspects, but omits or flattens a few details.    
   **Score 5:** About half of the salient information is captured; some significant stories are missing or under-emphasized.    
   **Score 1:** Little of the blog’s meaningful content is present; major omissions.  
   
3. **Fidelity (QA Sample Self-Consistency & Quality)**    
   *(Consider the QA sample itself, regardless of the blog.)*    
   - Is the scenario vivid, concrete, detailed, interesting, and coherent—not generic or trivial?    
   - Are the scenario, question, and answer logically connected and non-redundant?    
   - Do the question and answer have thoughtful depth and specificity?    
   **Score 10:** All fields are natural, complete, and richly detailed; scenario is engaging; question and answer are deep and complementary.    
   **Score 8:** QA sample is mostly solid and engaging, but may lack depth or vividness in some places.    
   **Score 5:** Sample is complete and understandable, but is somewhat generic, routine, or underdeveloped.    
   **Score 1:** Content is generic, duplicative, trivial, unnatural, or weak.  
   
4. **Novelty & Interest**    
   *(Consider the QA sample itself, regardless of the blog.)*    
   How much does the QA sample feel interesting, distinctive, and socially or emotionally resonant, rather than formulaic or generic?    
   **Score 10:** Highly distinctive, memorable, and evocative; clear emotional or cognitive resonance.    
   **Score 8:** Largely distinctive and interesting, but contains minor generic segments.    
   **Score 5:** Moderately engaging, but features significant generic or routine stretches.    
   **Score 1:** Largely generic, formulaic, or uninteresting.  
   
5. **Leakage (Information Overlap between Scenario & Answer)**    
   *(Consider the QA sample itself, regardless of the blog.)*    
   To what degree is the answer already revealed or obvious from the scenario description? Higher scores mean the answer requires social reasoning or inference (not just copying scenario details); lower scores mean much of the answer is simply restated or directly found in the scenario, requiring little reasoning to deduce.    
   **Score 10:** Very little to no direct answer content is present in the scenario; answering requires significant inference or reasoning.    
   **Score 8:** Some minor details of the answer are found in the scenario, but critical information still requires nontrivial inference.    
   **Score 5:** Substantial overlap between scenario and answer; the answer is partially but not fully restated or deducible from scenario alone.    
   **Score 1:** Most or all of the answer is directly stated in the scenario; almost no reasoning is required to answer.  
   
6. **Overall Quality**    
   Your holistic judgment of the data sample’s overall quality, based on accuracy, informativeness, distinctiveness, coverage, and naturalness, scored as an integer from 1 to 10.  
   
---  
**Output Format:**    
Please provide your ratings using the following template:  
```xml  
<data>  
  <explanation>[Short summary of your analysis and reasoning]</explanation>  
  <hallucination>[INTEGER 1-10]</hallucination>  
  <coverage>[INTEGER 1-10]</coverage>  
  <fidelity>[INTEGER 1-10]</fidelity>  
  <novelty>[INTEGER 1-10]</novelty>  
  <leakage>[INTEGER 1-10]</leakage>  
  <overall>[INTEGER 1-10]</overall>  
</data>  
```  
---  
**Instructions:**    
Rate each QA sample thoughtfully and fairly, paying attention to nuances in both blog and sample. Be strict—reserve high scores for truly strong samples, and use the full scoring scale as appropriate. Judge hallucination and coverage strictly according to the original blog; judge fidelity, novelty, and leakage based only on the QA sample. Only output the xml format result, do not output other explanatory words.  
   
---  
Now the task begins.  


"""

PROMPT_socialscenarios_singleblog_fulltypes = """

You are an expert, strict, objective and fair reviewer of social scenario and narrative data quality. For each example, you are provided with:    
(1) a single original blog post, and    
(2) a generated data sample extracted or condensed from that blog, consisting of elements such as characters, background, summary, scenario, plots, and/or thoughts.    
Your job is to carefully evaluate and rate the quality of the extracted social scenario/story sample across several dimensions. Base your evaluation on both the sample and the original blog, as described for each metric.  
   
**Evaluate the following aspects:**  
   
1. **Hallucination**    
Does the extracted scenario/story/summary/thoughts remain true to the **main story, experience, or emotional arc** of the original blog? Are major events, characterizations, and key reflections either clearly present in the blog or are reasonable, justifiable inferences? Minor details can be inferred if they do not distort the essence.    
**Score 10:** All core narrative content, relationships, and emotions are faithful to the blog; no significant inventions or speculative additions about main events or characters. Minor imaginative details are plausible.    
**Score 8:** One or two small, likely inferences or omittable details stray from the blog, but the **main story, scenario, and tone are grounded and unchanged**.    
**Score 5:** Several elements or nuances are not directly supported by the blog, or the central scenario is subtly altered—but the main experience/intention is preserved (no radical invention).    
**Score 1:** Any significant change in the primary storyline, emotional arc, or central characters/events; or many unsupported facts—even if the basic theme is retained.  
   
2. **Coverage**    
How well does the extracted sample capture and represent **the salient, evocative, or unique aspects** of the original blog? Has it distilled the blog’s most meaningful content?    
**Score 10:** Most or all distinctive and meaningful points from the blog are well represented.    
**Score 8:** Captures the main ideas, with a few secondary or nuanced details omitted or flattened.    
**Score 5:** About half of the interesting/emotional content is present; some major elements are missing or presented superficially.    
**Score 1:** Most content is missing or generic; major omissions.  
   
3. **Fidelity (Internal Coherence & Quality)**    
*(Consider only the extracted sample itself, regardless of the blog.)*    
- Is the narrative/scenario well-formed, concrete, detailed, and vivid—not generic or superficial?    
- Are background, character, and scenario consistent and complementary?    
- Is emotional or personal resonance preserved?    
**Score 10:** All fields are natural, complete, and richly detailed; characters, settings, and thoughts are vivid and internally coherent.    
**Score 8:** Mostly solid and engaging, but possibly lacking depth or subtlety in places.    
**Score 5:** Sufficient and understandable, but somewhat routine, thin, or incomplete.    
**Score 1:** Generic, incoherent, duplicative, or weak.  
   
4. **Novelty & Interest**    
*(Consider only the extracted sample itself, regardless of the blog.)*    
How interesting, memorable, and distinctive is the extracted scenario/story/reflective thought? Does it evoke a specific personal or social experience, or is it generic?    
**Score 10:** Distinctive, emotionally or socially resonant, and memorable; high interest.    
**Score 8:** Engaging and largely distinctive, but may contain some routine elements.    
**Score 5:** Moderately engaging, but features significant generic content.    
**Score 1:** Largely ordinary, formulaic, or uninteresting.  
   
5. **Overall Quality**    
Your holistic judgment of the sample’s overall quality as a piece of social scenario/narrative data, considering accuracy, informativeness, distinctiveness, coverage, emotional resonance, and naturalness.    
Score 1–10.  
   
---  
   
**Output Format:**    
Please provide your ratings using the following template:  
   
```xml  
<data>  
  <explanation>[Short summary of your analysis and reasoning]</explanation>  
  <hallucination>[INTEGER 1-10]</hallucination>  
  <coverage>[INTEGER 1-10]</coverage>  
  <fidelity>[INTEGER 1-10]</fidelity>  
  <novelty>[INTEGER 1-10]</novelty>  
  <overall>[INTEGER 1-10]</overall>  
</data>  
```  
   
---  
   
**Instructions:**    
Review each extraction fairly and thoroughly, paying close attention to nuances and unique expressions in both the original blog and the extracted sample. Be strict in your assessments—reserve high scores for truly compelling and faithful samples, and use the full scale as appropriate. Judge hallucination and coverage in the context of the original blog; score fidelity and novelty based only on the quality of the extracted sample.  
   
Only output the XML format result—no extra explanation outside the template.  
   
---  
   
**The task now begins.**

"""

PROMPT_judger_screenplay_fulltypes = """
   
You are an expert evaluator tasked with assessing the quality of a set of extracted social scenarios, each presented in a screenscript-like narrative format, and generated from a person’s raw blog posts. Your goal is to provide a fair, nuanced, and concise evaluation across several important dimensions. For reference, you will be given the original blog content and the set of generated social scenarios (in an XML or similar structured format containing characters and scripted scenes).  
   
Carefully evaluate the extracted scenarios according to the following five criteria:  
---  
   
### **Assessment Criteria**  
1. **Hallucination (Groundedness to the Blogs):**    
Assess whether every scene, character, and event in the extracted social scenarios is clearly and accurately grounded in the original blog(s).  
   - **Score 10:** All scenarios, characters, and details are evident in or directly inferable from the blogs; no invented elements.  
   - **Score 8:** Almost all content is grounded; very few, minor inferred details added.  
   - **Score 5:** Several elements are only loosely connected or speculative.  
   - **Score 1:** Many invented characters/events; content is largely untraceable to the blogs.  
   
2. **Coverage (Representation of Salient Social Moments):**    
Evaluate how well the social scenarios capture the range and importance of social interactions, key moments, and significant narratives from the blogs.  
   - **Score 10:** Nearly all significant social stories and interactions are represented.  
   - **Score 8:** Most important scenarios are included, but some notable ones are missing.  
   - **Score 5:** Only about half of the salient scenarios are captured.  
   - **Score 1:** Major gaps or omissions; most meaningful moments missing.  
   
3. **Informativeness (Detail and Depth):**    
Judge ONLY the extracted social scenarios. Consider whether each scene and its dialogue/action present enough context, detail, and richness to convey the social dynamics and relationships clearly.  
   - **Score 10:** Scenes are vivid, with well-contextualized actions, expressive dialogue, and strong atmosphere.  
   - **Score 8:** Generally detailed, but some portions lack depth or context.  
   - **Score 5:** Moderately informative; several scenes are generic or underdeveloped.  
   - **Score 1:** Sparse, superficial, or repetitive; little actual insight.  
   
4. **Novelty (Uniqueness and Lived Social Experience):**    
Judge ONLY the extracted scenarios. Do the narratives bring out distinctive personalities, personal styles, unusual settings, or specific social dynamics that characterize the user’s unique lived experience?  
   - **Score 10:** Highly distinctive; scenarios provide clear insight into individual personalities and unique relationships.  
   - **Score 8:** Some unique or personal elements, but several scenarios could apply to anyone.  
   - **Score 5:** Mostly conventional social scripts; only mild hints of individuality.  
   - **Score 1:** Entirely generic; could describe almost any group or family.  
   
5. **Overall Score**    
Your overall, holistic assessment of the extracted social scenarios as a high-quality resource for research into social narratives, considering all criteria above.  
   - **Score 10:** Outstanding—exceptionally useful, accurate, and insightful.  
   - **Score 8:** Strong—very useful, with only minor weaknesses.  
   - **Score 5:** Adequate—some value, but noticeable limitations.  
   - **Score 1:** Unusable—seriously flawed or unreliable.  
   
---  
   
**Instructions:**    
- Base your analysis strictly on the five metrics above.    
- For the <explanation> section, give a concise justification for your scoring, citing concrete strengths and weaknesses with reference to both the blog(s) and the extracted scenarios where relevant.    
- For each scoring field, output a single integer between 1 and 10.    
- Output your response ONLY in the following format, replacing bracketed sections with your responses (do not include anything outside this template):  
   
```  
<data>  
<explanation> [Your brief analysis and justification] </explanation>  
<hallucination> [Your score: integer 1–10] </hallucination>  
<coverage> [Your score: integer 1–10] </coverage>  
<informativeness> [Your score: integer 1–10] </informativeness>  
<novelty> [Your score: integer 1–10] </novelty>  
<overall> [Your score: integer 1–10] </overall>  
</data>  
```  
   
**You will be provided both the original blog posts and the extracted social scenarios.**    
Do not include any information outside the output template above. Be concise, fair, and evidence-based in your judgment.  

"""

PROMPT_judger_v1_user_journey_fullytpes = """
You are an expert evaluator tasked with assessing the quality of an extracted long-form user journey profile generated from a person’s original blog posts. Your goal is to provide a fair, nuanced, and concise evaluation across several important dimensions. For your reference, you will be given the original blog content and the generated long-form user journey.  
   
Carefully evaluate the generated user journey according to the following five criteria:  
   
---  
   
### **Assessment Criteria**  
   
1. **Hallucination** (Groundedness to the Blogs):  
   Assess whether the details and stories in the generated user journey are fully grounded in or directly paraphrased from the original blogs.  
   - **Score 10:** All content is traceably present in or is a clear paraphrase of the blogs—no invented or unsupported claims.  
   - **Score 8:** Nearly everything is grounded; only minor or highly plausible inferences present.  
   - **Score 5:** Multiple sections of the journey are only loosely linked or stretch beyond what is in the blogs.  
   - **Score 1:** Many elements are invented, generalized, or cannot be justified by the blog content.  
   
2. **Coverage** (Completeness of Salient Experiences):  
   Evaluate whether the user journey meaningfully covers the breadth and main highlights of stories and experiences in the original blogs.  
   - **Score 10:** All key periods, events, and personal insights from the blogs are included in the journey.  
   - **Score 8:** Most major life events and distinctive perspectives are present, but a few notable ones may be missing.  
   - **Score 5:** About half the major experiences are represented; some obvious gaps or unbalanced selection.  
   - **Score 1:** Few to none of the meaningful stories from the blogs are included.  
   
3. **Informativeness** (Depth and Specificity):  
   Judge ONLY the generated journey. Does it offer rich, vivid, and concrete detail, or is it shallow and generic?  
   - **Score 10:** Consistently informative, with vivid descriptions, specific examples, and deep insights.  
   - **Score 8:** Generally detailed, though a few sections are more surface-level.  
   - **Score 5:** Passably informative, but several sections are vague, lacking substance or depth.  
   - **Score 1:** Mostly generalities with little insightful or specific content.  
   
4. **Novelty** (Individual Uniqueness):  
   Judge ONLY the generated journey. Does it capture what is distinctive about this person’s journey—their personality, values, and unique experiences?  
   - **Score 10:** Strong sense of individuality; the profile could clearly not refer to anyone else.  
   - **Score 8:** Some unique or personal details, but parts still feel somewhat generic.  
   - **Score 5:** Largely formulaic, with only minor hints of what makes this person distinctive.  
   - **Score 1:** Entirely generic or bland; seemingly interchangeable with anyone.  
   
5. **Overall Score**  
   Your comprehensive judgment of the journey’s usefulness and overall quality as a research or biographical resource, considering all criteria above.  
   - **Score 10:** Outstanding—exceptionally well-written, valuable, and reliable.  
   - **Score 8:** Strong—very good quality, with only minor issues.  
   - **Score 5:** Adequate—provides meaningful value but has clear weaknesses.  
   - **Score 1:** Poor—major flaws or unreliable.  
   
---  
   
**Instructions:**  
- Base your analysis primarily on the five metrics above.  
- For the <explanation> section, give a concise justification for your scoring, referencing specifics from both the source blogs and the generated journey profile as appropriate.  
- For each scoring field, output a single integer between 1 and 10.  
- Output your response ONLY in the following format, replacing bracketed sections with your responses (do not include anything outside this template):  
   
```  
<data>  
<explanation> [Your brief analysis and justification] </explanation>  
<hallucination> [Your score: integer 1–10] </hallucination>  
<coverage> [Your score: integer 1–10] </coverage>  
<informativeness> [Your score: integer 1–10] </informativeness>  
<novelty> [Your score: integer 1–10] </novelty>  
<overall> [Your score: integer 1–10] </overall>  
</data>  
```  
   
You will be provided both the original blog posts and the generated user journey.    
Do not include any information outside the output template above. Be concise, fair, and evidence-based in your judgment.

"""

PROMPT_job_rowwise_roleplayv2 = """ 
You are given a blog post written by a single person. Your task is to extract a data sample in the format <type, scenario, behavior>, suitable for general role-playing tasks. At the end of the scenario, include a question about the main character’s (the author’s) next behavior (such as utterance, action, feelings, thoughts, etc. You will later be provided with the full definitions for 12 behavior types).  Note that the purpose and basis of the general role-playing tasks are to predict what behavior will happen next (as a future event), not to ask readers to find answer from the scenario.
   
### Extraction Steps  
   
**1. Read and Analyze:**    
Carefully read the blog post to understand its content.  
   
**2. Identify ONE Primary Behavior Type:**    
Select the single most central type from the list of 12 behavior types provided below.    
*(Full definitions appear at the end.)*  
   
1. Communication Style  
2. Nonverbal and Paralinguistic Expression  
3. Emotional Experience and Regulation  
4. Motivation, Needs, and Goals  
5. Cognitive Style and Reasoning Patterns  
6. Values, Beliefs, and Morality  
7. Personality Traits  
8. Preferences, Interests, and Lifestyle Habits  
9. Social Interaction Patterns  
10. Social and Cultural Identity  
11. Self-Concept and Identity Narratives  
12. Adaptability and Situation-based Flexibility  
   
**3. Compose XML Fields**    
For the chosen behavior type, provide the following:  
   
- **scenario:**    
  - Generate a detailed scenario description (in third person, e.g., "The person...") based on the blog, ending with a single, clear question about what behavior the main character will have next. Note that the original blog usually contain a lot of words not related to stories, events, or social interactions. Focus on human behaviors and social events. It is not necessary to cover everything mentioned in the blog. Write the scenario as you are describing a vivid story. 
  - Use present simple tense.  
  - The scenario should be grounded in the blog. Do not fabricate stories.   
  - **Strict Rule:** Only describe events leading up to — but not after — the behavioral moment asked in the question.    
  - The question must be about one specific, concrete behavior (utterance, action, emotional state, thought, etc.) in that context.    
  - Do NOT include general, abstract, or inferential questions (e.g., "What does this reveal about the person’s personality?" or "How does the person usually respond?"). The question must prompt a specific behavior.    
  - Only one question is allowed.  
   
- **behavior:**    
  - Provide a direct, specific answer to the question, strictly grounded in the details from the blog post.    
  - Use third-person perspective.  
   
**4. Output Format**    
Return your results as XML, formatted as follows:  
   
```xml  
<data>  
  <type>[name of the selected behavior type]</type>  
  <scenario>[detailed scenario with a single short question at the end]</scenario>  
  <answer>[direct, specific answer, based on the blog post]</answer>  
</data>  
```  
   
### Behavior Type Definitions  
   
1. **Communication Style:**    
   How the person uses language in interaction; includes vocabulary, tone, directness, humor, formality, rhetorical devices, etc.  
2. **Nonverbal and Paralinguistic Expression:**    
   How the person expresses themselves without words or via prosodic features (in text, only if explicitly described); includes gestures, posture, facial expression, body language, and comments on intonation.  
3. **Emotional Experience and Regulation:**    
   How the person feels, expresses, and manages feelings; includes stated emotions, triggers, coping strategies, recovery from setbacks, etc.  
4. **Motivation, Needs, and Goals:**    
   What drives the person's actions; includes desires, fears, ambitions, priorities, and stated reasons for acting.  
5. **Cognitive Style and Reasoning Patterns:**    
   How the person thinks and comes to conclusions; includes decision-making, self-talk, reflection, openness, risk assessment.  
6. **Values, Beliefs, and Morality:**    
   The principles that guide the person; includes ethical statements, value judgments, worldviews, and comments on right/wrong.  
7. **Personality Traits:**    
   Trait-like behavioral or temperamental patterns (e.g., Big Five); includes descriptions such as extraversion–introversion, agreeableness, emotional stability.  
8. **Preferences, Interests, and Lifestyle Habits:**    
   Regular choices and activities; includes hobbies, tastes, routines, and comfort zones.  
9. **Social Interaction Patterns:**    
   How the person maintains relationships; includes assertiveness, conflict style, group/individual orientation, networking, boundary-setting.  
10. **Social and Cultural Identity:**    
    Group affiliations and influences; includes cultural references, nationality, conformity, and attitudes toward groups.  
11. **Self-Concept and Identity Narratives:**    
    How the person understands and narrates their own identity; includes life scripts, self-esteem, reflections on growth, or role identification.  
12. **Adaptability and Situation-based Flexibility:**    
    Capacity for adjustment and learning; includes changing strategies, openness to feedback, resilience, and context-sensitive responses.  
   
**Example of a Bad Case**    
Below is an example where the answer is already revealed by the scenario — avoid this structure:  
   
```xml  
<data>  
  <type>Nonverbal and Paralinguistic Expression</type>  
  <scenario>The person, an older man... sits hunched forward in his car, gripping the steering wheel tightly... drool... mouth hanging open... waits patiently at an intersection... How does the person physically express himself as he waits at the intersection?</scenario>  
  <answer>The person hunches forward, grips the steering wheel tightly, leaves his mouth hanging open...</answer>  
</data>  
```  
*The scenario already states the answer; do NOT do this.*  
   
---  

Now your task begins. 
Output only the XML result—do not include explanations. But if the quality of the raw blog is not good to extract a good story, please simple output "NULL". The raw blog post will follow. 


"""

PROMPT_job_rowwise_extract_possible_type_of_behaviors = """ 
You are given a blog post written by a single person. Your task is to identify which types of behaviors, from the 12 types listed below, can be extracted as stand-alone, high-quality data samples suitable for general role-playing tasks.  
   
**You must be highly selective. Only include behavior types that meet all of the following conditions:**  
   
- The scenario clearly centers on a meaningful event, action, or social interaction involving the main character (the blogger).  
- There is enough contextual detail to allow for an unambiguous, specific, and realistic question about what the main character will do, say, think, or feel next (as a *future* behavior).  
- The blog provides adequate explicit evidence to answer the question directly, without requiring outside world knowledge, speculation, or significant inference.  
- Ignore general statements, vague commentary, habitual descriptions, or introspective monologues that do not clearly lead up to a behavioral inflection or decision point.  
- Only select types for which a data sample can be formulated that is both precise (as a single behavior) and grounded (fully traceable to the blog’s content and context).  
- If no part of the blog allows for a high-quality, concrete scenario about a *future* behavior, output "NULL".  
   
**The full list of 12 behavior types is below:**  
   
1. Communication Style    
2. Nonverbal and Paralinguistic Expression    
3. Emotional Experience    
4. Motivation, Needs, and Goals    
5. Cognitive Style and Reasoning Patterns    
6. Values, Beliefs, and Morality    
7. Personality Traits    
8. Preferences, Interests, and Lifestyle Habits    
9. Social Interaction Patterns    
10. Social and Cultural Identity  
11. Self-Concept and Identity Narratives    
12. Adaptability and Situation-based Flexibility    
  

Their detailed definitions for your reference: 
### Behavior Type Definitions   
1. **Communication Style:**    
   How the person uses language in interaction; includes vocabulary, tone, directness, humor, formality, rhetorical devices, etc.  
2. **Nonverbal and Paralinguistic Expression:**    
   How the person expresses themselves without words or via prosodic features (in text, only if explicitly described); includes gestures, posture, facial expression, body language, and comments on intonation.  
3. **Emotional Experience:**    
   How the person feels, expresses, and manages feelings; includes stated emotions, triggers, coping strategies, recovery from setbacks, etc.  
4. **Motivation, Needs, and Goals:**    
   What drives the person's actions; includes desires, fears, ambitions, priorities, and stated reasons for acting.  
5. **Cognitive Style and Reasoning Patterns:**    
   How the person thinks and comes to conclusions; includes decision-making, self-talk, reflection, openness, risk assessment.  
6. **Values, Beliefs, and Morality:**    
   The principles that guide the person; includes ethical statements, value judgments, worldviews, and comments on right/wrong.  
7. **Personality Traits:**    
   Trait-like behavioral or temperamental patterns (e.g., Big Five); includes descriptions such as extraversion–introversion, agreeableness, emotional stability.  
8. **Preferences, Interests, and Lifestyle Habits:**    
   Regular choices and activities; includes hobbies, tastes, routines, and comfort zones.  
9. **Social Interaction Patterns:**    
   How the person maintains relationships; includes assertiveness, conflict style, group/individual orientation, networking, boundary-setting.  
10. **Social and Cultural Identity:**    
    Group affiliations and influences; includes cultural references, nationality, conformity, and attitudes toward groups.
11. **Self-Concept and Identity Narratives:**    
    How the person understands and narrates their own identity; includes life scripts, self-esteem, reflections on growth, or role identification.  
12. **Adaptability and Situation-based Flexibility:**    
    Capacity for adjustment and learning; includes changing strategies, openness to feedback, resilience, and context-sensitive responses.     

**Output Format:**    
Return your answer in XML:  
   
```xml  
<data>  
<type>[the full name of one selected behavior type]</type>  
[...if other types meet the criteria, list each on a new <type> line...]  
</data>  
```  
If no suitable behaviors can be extracted, output only:  
   
```  
NULL  
```  
   


Now your task begins. 
Output only the XML result—do not include explanations. But if the quality of the raw blog is not good to extract a good story, please simple output "NULL". The raw blog post will follow. 


"""

PROMPT_job_rowwise_roleplay_target_conversation = """ 
You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for role-playing tasks focused on conversational behavior. The *type* field should always be "Conversation" for this task.  
   
Your goal is to generate data that predicts what the main character (i.e., the author) will SAY (their next utterance) in a clearly described scenario based on the blog. The answer should be the actual utterance (as direct speech), not a summary of what they might say or how they might respond.  
   
**CRITICAL REQUIREMENT:**    
When composing the scenario, you must **draw as much situational and social detail, such as utterance, conversations, and actions, from the blog post as possible**. The scenario should richly describe the circumstances, environment, people involved, prior actions, and any relevant context—using all grounded details provided in the blog post leading up to the next spoken utterance. Avoid any vagueness, and do not simply summarize; immerse the reader in the immediate, specific situation, stopping just before the main character makes their next statement. **Do not include or paraphrase the answer or any future behavior in the scenario.**  
   
If the provided blog post does not support constructing a high-quality scenario where the character’s immediate next utterance is contextually clear and realistic (which is highly likely to happen), output NULL and do not attempt to extract a sample.  
   
---  
   
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand both the social setting and the author’s perspective.  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:  
   - The blog contains enough context to construct a realistic scenario where the author’s next utterance (i.e., what they would literally say out loud to another person) is supported by details in the text and can be written explicitly as dialog.  
   - You can formulate a question that directly asks, "What does the person say next?" or a similarly specific variant.  
   - The blog post includes a described social situation, interaction, or dialogue context (not just abstract rumination or narration with no clear conversational potential).  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**    
   - **type (always "Conversation")**  
   - **scenario:**    
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people involved, circumstances, recent events, and any emotionally relevant context based squarely on the blog, using as many specifics from the text as possible. Write like a story or a novel. Lead up to—but NOT including—the main character’s next utterance.  
     - End with a single, concrete question explicitly about what the main character will SAY next in that situation (e.g., "What does the person say to their friend next?").  
     - Do NOT reveal, summarize, or paraphrase the answer in the scenario.  
     - Only one question is allowed.  
  
   - **answer:**    
     - Provide the actual utterance/content of the character’s next speech act (as direct dialog), grounded in the blog post.  
     - Write the answer in third person, as direct speech, in quotation marks.  
   
4. **Output Format:**    
   Return your result in XML as follows:  
   ```xml  
   <data>  
     <type>Conversation</type>  
     <scenario>[Highly detailed scenario, grounded in the blog post, ending with a question asking what the person says next]</scenario>  
     <answer>["..." — the direct utterance, as the character would say it]</answer>  
   </data>  
   ```  
   **If a high-quality utterance sample cannot be produced, output only:**  
   NULL  
   
---  
   
#### Strict Requirements  
   
- **Extract only when the blog directly supports a plausible, specific utterance as the main character’s next behavior.**  
- **Do not fabricate context, events, or dialog not present in the blog post.**  
- **The scenario MUST be as specific and detailed as possible, using concrete information from the blog's language, events, setting, and dialogue (but without including or paraphrasing the answer).**  
- **The answer MUST be what the character would *literally* say out loud next—not feelings, summaries, or paraphrased reactions.**  
- **If no suitable utterance is available, output NULL.**  
   
---  
   
**Summary Table**  
   
| Step              | Description                                                                                  |  
|-------------------|----------------------------------------------------------------------------------------------|  
| Suitability Check | Is a next utterance possible, clearly and concretely grounded in blog context? If no, output NULL. |  
| Scenario          | Third-person, present tense; as detailed and context-rich as possible from the blog; leads up to utterance moment, ends with a dialog prompt. |  
| Question          | Concretely asks what the main character says next.                                            |  
| Answer            | The utterance, precisely as the character would say based on the blog.         |  
| Only Conversation | "type" field is always "Conversation."                                                        |  
   
---  
   
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good to extract a good story, please simply output "NULL". The raw blog post will follow.   
  
"""

PROMPT_job_rowwise_roleplay_target_culture_and_socialnorm = """You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for role-playing tasks focused on observable **“Cultural and Social Normative Behavior.”** The *type* field should always be **"Cultural And Social Norm"** for this task.  
   
Your goal is to generate data that predicts what the main character (i.e., the blog’s author) will DO next—**specifically an observable behavior or action governed by cultural or social norms—in a clearly described scenario based on the blog.** The answer should be the plausible next normative action (not a summary or reasoning, but the explicit act) that the person would take, as shaped by the expectations, customs, etiquette, or rules relevant in their cultural or social context.  
   
**CRITICAL REQUIREMENT:**  
   
When composing the scenario, you must **draw as much situational, cultural, and social detail from the blog post as possible.** The scenario should richly describe the circumstances, people involved, environmental context, cultural background or references, recent events, and the specific social or cultural expectations shaping behavior—using all grounded details provided in the blog leading up to the person’s next normative action. Avoid vagueness, do not summarize, and do not invent context; immerse the reader in the immediate, richly contextualized situation, stopping just before the main character performs the next observable culture- or norm-driven act. **Do not include or paraphrase the answer or any future behavior in the scenario.**  
   
If the provided blog does not support constructing a high-quality scenario where the character’s immediate next culturally/socially normative behavior is clear and realistic (which is very often the case), output **NULL** and do not attempt to extract a sample.  
   
---  
   
### Extraction Steps  
   
1. **Read and Analyze:**  
   Carefully read the blog post, focusing on the author’s descriptions of social and cultural context, expectations, customs, rituals, etiquette, or reference to group standards.  
   
2. **Determine Suitability:**  
   Extract a sample ONLY IF:  
   - The blog contains enough context to construct a realistic scenario where the author’s next **observable** behavior, driven by cultural or social norms (e.g., ritual, politeness, etiquette, adherence to a custom, or culturally expected acts) is supported by details in the text and can be written explicitly as a concrete action.  
   - You can formulate a question that directly asks, "What does the person do next according to the social/cultural rules or expectations?" or a similarly specific variant.  
   - The blog describes participation in a social setting, event, group ritual, or culturally framed moment—not just internal reflection, private rumination, or abstract narration.  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**  
   - **type:** (Always "Cultural And Social Norm")  
   - **scenario:**  
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people involved, circumstances, cultural or social context, recent events, and group expectations, based on the blog. Use as many explicit details from the text as possible to specify why particular behaviors are normatively expected or performed in this setting. Write in the style of a story or a novel.  
     - End with a single, concrete question explicitly about what the main character will DO next in that situation that are related to social/cultural expectations.  
     - Do NOT reveal, summarize, or paraphrase the answer in the scenario.  
     - Only one question is allowed.  
  
   - **answer:**    
     - Provide the actual next normative action/behavior (as third person, observable act), grounded in the blog.  
   
4. **Output Format:**    
   Return your result in XML as follows:  
   ```xml  
   <data>  
     <type>Cultural And Social Norm</type>  
     <scenario>[Highly detailed scenario, grounded in the blog post, ending with a question about the next culture- or norm-driven behavior]</scenario>  
     <answer>[The character’s immediate observable next action rooted in cultural/social context]</answer>  
   </data>  
   ```  
   **If a high-quality sample cannot be produced, output only:**  
   NULL  
   
---  
   
#### Strict Requirements  
   
- **Extract only when the blog directly supports a plausible, concrete, and culturally/socially normative action as the main character’s next behavior.**  
- **Do not fabricate context, events, or group norms not present in the blog post.**  
- **Scenario MUST be as specific and detailed as possible, using concrete information from the blog’s language, events, cultural references, and social context (but without including or paraphrasing the answer).**  
- **Answer MUST be the immediate next observable act in that context—not a general guideline, explanation, or paraphrased summary.**  
- **If no suitable culture-/norm-driven behavioral act is available, output NULL.**  
   
---  
   
**Summary Table**  
   
| Step              | Description                                                                                                              |  
|-------------------|--------------------------------------------------------------------------------------------------------------------------|  
| Suitability Check | Is a next culturally/socially normative, observable action possible and clearly grounded in blog context? If no, output NULL. |  
| Scenario          | Third-person, present tense; as detailed and context-rich as possible from the blog; leads up to behavior moment, ends with a prompt. |  
| Question          | Concretely asks what the main character does next according to the cultural or social norms at play.                      |  
| Answer            | Immediate, observable act (in third person), precisely as the character would perform based on the blog and context.      |  
| Only Culture/Social Norm | The "type" field is always "Cultural And Social Norm."                                                            |  
   
---  
   
**If the quality of the raw blog does not support a high-quality scenario for this behavior type, return only:**  
   
```  
NULL  
```  
   
---  
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good to extract a good story, please simply output "NULL". The raw blog post will follow.   
  
"""

PROMPT_job_rowwise_roleplay_target_value = """  
You are given a blog post written by a single person. Your task is to extract a high-quality behavioral data sample in the format <type, scenario, answer>, suitable for role-playing tasks focused on "Values, Beliefs, and Morality." The *type* field should always be "Values_Beliefs_Morality" for this task.  
   
Your goal is to generate data that predicts what the main character (the author) will *explicitly state, act, behave, conclude, or demonstrate* in terms of personal **values, beliefs, or moral judgments** in a clearly described scenario, based on the blog. The answer should directly reflect the author’s *articulated or demonstrable* value, belief, or moral stance as supported by the blog—such as an ethical statement, value judgment, worldview, or a declaration of what they consider right or wrong.  For example, these may include, but are not limited to, examples from Schwartz’s Human Values model (e.g., statements or behaviors reflecting self-direction, stimulation, hedonism, achievement, power, security, conformity, tradition, benevolence, or universalism).
   
**CRITICAL REQUIREMENTS:**  
   
When composing the scenario, you must **draw as much contextual and situational detail—including events, reasoning, challenges, and relevant background—from the blog post as possible**. The scenario should richly describe the circumstances, environment, people involved, conflicts or dilemmas, and any context that illuminates the principle or belief, using all grounded details provided in the blog post leading up to the highlighted statement or action. Avoid vagueness or general summary; immerse the reader in the specific, immediate situation that sets up a clear value- or belief-based stance. **Do not include, paraphrase, or summarize the value/belief statement itself in the scenario.**  
   
If the provided blog post does not support constructing a high-quality scenario where the character’s salient value, belief, or moral position is contextually clear and realistically expressed (which is highly likely to happen), output NULL and do not attempt to extract a sample.  
   
---  
   
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand the author’s situation, reasoning, and any explicit statements of values, beliefs, or moral judgments.  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:  
   - The blog contains enough context to construct a realistic scenario where the author’s explicit value, belief, or moral position (e.g., a statement of right/wrong, sincerity, fairness, loyalty, liberty, tradition, etc.) is supported by details in the text and can be written explicitly as a *clear, direct expression* or demonstrable act.  
   - You can formulate a question that directly asks for the person's next behavior and its corresponding value, belief, or morality.  
   - The blog post includes a described situation, dilemma, conflict, reflection, or context where a value- or belief-driven position is clearly relevant and directly expressed (not merely implied, vague, or abstract).  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**  
   - **type (always "Values_Beliefs_Morality")**  
   - **scenario:**  
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people (if any) involved, circumstances, values-relevant events, any recent challenges, and emotionally or ethically relevant context, using as many specifics from the blog as possible. Write like a story or novel. Lead up to—but NOT including—the main character’s next explicit value, belief, or moral expression (statement or behavior demonstrating their values).  
     - End with a single, concrete question explicitly about what behaviors, utterance, statement, actions, or decisions will happen next in that moment that can reflect their value, belief, or moral position.  
     - Do NOT include, summarize, or paraphrase the value/belief/moral statement already described in the scenario.  
     - Only one question is allowed.  
   - **answer:**  
     - Provide the direct statement, action, behavior, or explicit demonstration of the character’s value, belief, or moral stance, as it appears in the blog post, written in third person  (or, if non-verbal but explicit, a brief, clear description such as: ["She refuses to cheat, saying 'I can’t do something I believe is wrong.'"]).  
   
4. **Output Format:**  
   Return your result in XML as follows:  
   ```xml  
   <data>  
     <type>Values_Beliefs_Morality</type>  
     <scenario>[Highly detailed scenario, grounded in the blog, ending with a question about the behavior that demonstrate person’s value, belief, or moral stance]</scenario>  
     <answer>[the clear behavioral demonstration, as the character would state or show based on the blog]</answer>  
   </data>  
   ```  
   **If a high-quality value, belief, or morality sample cannot be produced, output only:**  
   NULL  
   
---  
   
#### Strict Requirements  
   
- **Extract only when the blog directly supports a plausible, specific value/belief/moral statement or behavior by the main character.**  
- **Do NOT fabricate context, events, or positions not present in the blog.**  
- **The scenario MUST be as specific and detailed as possible, using concrete information from the blog’s language, events, setting, and emotional/ethical context (but without including or paraphrasing the answer).**  
- **If no suitable sample is available, output NULL.**  
   
 
---  
   
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good enough to extract a valid sample, simply output "NULL". The raw blog post will follow.


"""

PROMPT_job_rowwise_roleplay_target_personality = """   
You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for role-playing tasks focused on **Personality Trait** demonstration. The *type* field should always be "PersonalityTrait" for this task.  
   
Your goal is to generate data that predicts what the main character (i.e., the author) will DO or SAY next—a (hypothetical or actual) observable behavior or spoken utterance—in a clearly described scenario, which reveals or exemplifies a salient aspect of their personality traits (for example: extraversion, openness, conscientiousness, agreeableness, emotional stability, or similar trait-like qualities). The answer should be the main character’s likely or actual behavior or utterance, written in a way that makes the trait evident, NOT a summary or trait label itself.  
   
**CRITICAL REQUIREMENT:**  
When composing the scenario, you must **draw as much situational, social, and trait-related detail (including internal state, attitudes, previous behavior, etc.) as possible from the blog post**. The scenario should richly describe the circumstances, environment, people involved, recent actions, the main character’s behavioral tendencies, and any relevant context—using grounded details provided in the blog post. Do NOT simply summarize or label personality; instead, immerse the reader in an immediate, highly specific situation, stopping just before the main character’s trait-revealing action or utterance. **Do not include or paraphrase the answer or any future behavior in the scenario.**  
   
If the provided blog post does not support constructing a high-quality scenario where the character’s next observable action or utterance could meaningfully demonstrate a personality trait (which is quite possible), output NULL and do not attempt to extract a sample.  
   
---  
   
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand the social setting, situational details, and the author’s characteristic patterns (such as temperament, habitual reactions, styles of thinking or engaging).  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:  
   - The blog provides enough context to construct a plausible and realistic scenario in which the author’s next behavior or utterance would clearly demonstrate a personality trait (as evidenced by text).  
   - The scenario is specific enough to pose a question such as: "What does the person DO/SAY next" or a near equivalent. The qestion is asking for something that obviously reveals their personality trait. 
   - The blog describes a scene, event, or reflection with observable actions/reactions (not just abstract musings or trait-name claims).  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**  
   - **type (always "PersonalityTrait")**  
   - **scenario:**    
     Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people involved, circumstances, and recent events, focusing on the main character’s trait-relevant background, previous behavior, and context drawn as much as possible from the blog. Build up to—but NOT including—the next trait-revealing action or utterance.  
     - End with a single, concrete question that asks about the main character’s next observable behavior or utterance that could clearly exemplify a personality trait like "What does the person do next that shows their personality in this situation?").  
     - Do NOT reveal or paraphrase the answer in the scenario.  
     - Only one question is allowed.  
   - **answer:**    
     - Provide the actual behavior or utterance (in direct third-person narration or speech), grounded in the blog post, that vividly demonstrates the trait in context.  
     - Do not use trait terms or summaries—show the trait through the action or utterance itself.  
   
4. **Output Format:**    
   Return your result in XML as follows:  
```xml  
<data>  
  <type>PersonalityTrait</type>  
  <scenario>[Highly detailed, trait-focused scenario, grounded in the blog post, ending with a question about the trait-revealing action]</scenario>  
  <answer>[Trait-revealing behavior or utterance as third-person action or quoted speech]</answer>  
</data>  
```  
**If a high-quality PersonalityTrait sample cannot be produced, output only:**  
NULL  
   
---  
   
#### Strict Requirements  
   
- **Extract only when the blog directly supports a plausible, contextually-concrete observable behavior or utterance that demonstrates a personality trait.**  
- **Do not fabricate context, events, or dialog not present in the blog post.**  
- **The scenario MUST be as specific and detailed as possible, using concrete information from the blog's language, actions, setting, and reflections—especially details relevant to the trait (but without including or paraphrasing the answer).**  
- **The answer MUST be a direct, observable behavior/utterance (NOT a summary or trait term), as evidenced or highly implied by the blog.**  
- **If no suitable trait-revealing sample is available, output NULL.**  
   
---  
   
**Summary Table**  
   
| Step               | Description                                                                                   |  
|--------------------|-----------------------------------------------------------------------------------------------|  
| Suitability Check  | Can the next observable behavior or utterance clearly reveal a personality trait, based on blog? If no, output NULL.          |  
| Scenario           | Third-person, present tense; as detailed, trait-focused, and context-rich as possible from the blog; leads up to trait moment, ends with a question about trait-revealing action. |  
| Question           | Concretely asks what the main character does or says next to show their personality trait.      |  
| Answer             | Observable behavior or direct utterance (in third-person action or quoted speech), demonstrating the trait; no trait terms or summaries.   |  
| Only PersonalityTrait | "type" field is always "PersonalityTrait."                                                   |  
   
---  
   
**Instructions:**    
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not sufficient to extract a high-quality personality-trait-revealing sample, simply output "NULL." The raw blog post will follow.

"""


PROMPT_job_rowwise_roleplay_target_motivation = """
You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for role-playing tasks focused on "Motivation, Needs, and Goals" behavior. The *type* field should always be "Motivation, Needs, and Goals" for this task.    
  
Your goal is to generate data that predicts what the main character (i.e., the author) will WANT or NEED (i.e., their inner motivation, desire, or goal in that specific situation) and what concrete behavior(s) they will be driven to pursue next as a result. Do not merely summarize their general outlook—instead, identify the *explicit, situational desire, need, or goal* and what action or behavior it will immediately prompt, grounded and justified by the details in the blog.    
  
**CRITICAL REQUIREMENT:**    
When composing the scenario, you must **draw as much situational and social detail (including environment, actions, emotions, and context) from the blog post as possible**. The scenario should richly describe the circumstances, environment, people involved, recent events, and any emotionally or practically relevant factors—using all grounded details provided leading up to the moment when the character's next motivation and corresponding behavior naturally arise. Avoid vagueness and do not simply summarize; immerse the reader in the immediate and specific situation, stopping just before the main character forms their new desire/need/goal and initiates their next relevant behavior. **Do not include or paraphrase the answer in the scenario.**    
  
If the provided blog post does not support constructing a high-quality scenario where the character’s new motivation, need, or goal (and immediate behavior) is contextually clear and realistic, output NULL and do not attempt to extract a sample.  
   
---    
  
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand both the social context and the author’s psychological or emotional state.  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:    
   - The blog contains enough context to construct a realistic scenario where the author’s *next* motivation, need, or goal (and their corresponding behavior) is specific, clearly grounded in the text, and can be explicitly articulated.  
   - You can formulate a question that concretely asks, "What does the person want, need, or intend to accomplish (i.e., their internal motivation or goal) at this moment, and what do they do (i.e., their immediate motivated behavior) next?"  
   - The blog post has a described situation or event where a change/clarification of desire, need, or goal, and related behavior, is possible (not just static narration or abstract reflections).    
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**    
   - **type (always "Motivation, Needs, and Goals")**  
   - **scenario:**    
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people involved, circumstances, recent events, emotional context, and any elements relevant to the formation of new motivations and immediate behaviors, using as many specifics from the text as possible. Write like a story or a novel.    
     - Lead up to—but NOT including—the moment where the main character forms a new desire, need, or goal, and is about to act on it.    
     - End with a single, concrete question explicitly about what the person *wants* (or needs/intends) and what they *do* next because of that in that situation (e.g., "What does the person want or need, and what do they do next as a result?").  
     - Do NOT reveal, summarize, or paraphrase the answer in the scenario.  
     - Only one question is allowed.  
   - **answer:**    
     - State, in third person, the character’s next specific motivation/need/goal and the concrete behavior they pursue because of that in the given situation, using as much language and rationale grounded in the blog post as possible.  
     - Phrase the answer as: [Desire/Need/Goal]. [Concrete, motivated behavior or action prompted by this state.]  
     - Example:    
       "She wants to reconnect with her friend after the argument. She sends a text apologizing for her earlier harsh words."  
   
4. **Output Format:**    
   Return your result in XML as follows:  
   ```xml  
   <data>  
     <type>Motivation, Needs, and Goals</type>  
     <scenario>[Highly detailed scenario, grounded in the blog post, ending with a question asking what the person wants or needs and does next]</scenario>  
     <answer>[Motivation/need/goal. Motivated behavior or action.]</answer>  
   </data>  
   ```  
   **If a high-quality sample cannot be produced, output only:**    
   NULL  
   
---  
   
#### Strict Requirements  
   
- **Extract only when the blog directly supports a plausible, specific, and contextually grounded next motivation/need/goal AND associated concrete behavior.**  
- **Do not fabricate context, events, or internal states not present in the blog post.**  
- **The scenario MUST be as specific and detailed as possible, using concrete information from the blog's language, events, emotions, and setting (but without including or paraphrasing the answer).**  
- **The answer MUST be a clearly defined motivation/need/goal AND the resulting action—both grounded in the blog.**  
- **If no suitable sample is available, output NULL.**  
   
---  
   
**Summary Table**  
   
| Step              | Description                                                                                   |  
|-------------------|----------------------------------------------------------------------------------------------|  
| Suitability Check | Is a next motivation/need/goal and immediate behavior possible, clearly and concretely grounded in blog context? If not, output NULL. |  
| Scenario          | Third-person, present tense; as detailed and context-rich as possible from the blog; leads up to target moment, ends with motivation/goal question. |  
| Question          | Specifically asks what the main character wants/needs and what they do next.                  |  
| Answer            | Clearly stated motivation/goal/need and corresponding concrete behavior, both directly grounded in the blog. |  
| Only MN&G         | "type" field is always "Motivation, Needs, and Goals."                                       |  
   
---  
   
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good to extract a good story, please simply output "NULL". The raw blog post will follow.

"""


PROMPT_job_rowwise_roleplay_target_nonverbal = """
You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for role-playing tasks focused on **Nonverbal and Paralinguistic Expression**. The *type* field should always be "Nonverbal_Expression" for this task.  
   
Your goal is to generate data that predicts what the main character (i.e., the author) will explicitly DO in terms of gestures, posture, facial expressions, body language, or other nonverbal/paralinguistic expressions in a clearly described scenario based on the blog. The answer should be the specific nonverbal expression(s), **directly supported by details in the blog** (not imagined or paraphrased), as a literal description of physical or prosodic behavior at a key moment.  
   
**CRITICAL REQUIREMENT:**  
When composing the scenario, you must **draw as much situational and bodily expressive detail from the blog post as possible**. The scenario should richly describe the circumstances, environment, people involved, prior actions or utterances, and any relevant emotional or social context—using all grounded details provided in the blog post leading up to the main character’s next observable nonverbal (including paralinguistic) behavior. Avoid vagueness, and do not simply summarize; immerse the reader in the immediate, specific situation, stopping just before the main character manifests their nonverbal/paralinguistic action or cue. **Do not include or paraphrase the answer or any future behavior in the scenario.**  
   
If the provided blog post does not explicitly support constructing a high-quality scenario where the character’s immediate next nonverbal/paralinguistic expression is contextually clear and realistic (which is highly likely to happen), output NULL and do not attempt to extract a sample.  
   
---  
   
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand both the social setting and the author’s perspective, paying special attention to any descriptions of body language, facial expression, gestures, movement, posture, or prosody.  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:  
   - The blog **explicitly** includes enough context to construct a realistic scenario where the author’s next **nonverbal expression** (gestures, facial expression, posture, body movement, or described paralinguistic feature, such as intonation or sighs) is clearly described or can be unambiguously inferred from the text.  
   - You can formulate a question that directly asks What is the person’s (bodily expression/gesture/facial expression/body language/etc.) next, or a similarly specific variant.  
   - The blog post gives a social or emotional situation in which nonverbal/paralinguistic behavior is described or is a natural, **explicitly supported** next move (not imagined or generic).  
     
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**  
   - **type (always "Nonverbal_Expression")**  
   - **scenario:**  
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people involved, circumstances, recent events, and any emotionally or situationally relevant context squarely based on the blog. Use as many specifics from the text as possible. Write as in a narrative, leading up to—but NOT including—the main character’s next nonverbal/paralinguistic expression.  
     - End with a single, concrete question explicitly about what the main character will DO or display nonverbally in that situation (e.g., "What does the person do with their hands next?", "How does their facial expression change next?", or "What is their posture or gesture in this moment?").  
     - Do NOT reveal, summarize, or paraphrase the answer in the scenario.  
     - Only one question is allowed.  
   - **answer:**  
     - Provide the actual nonverbal expression or paralinguistic cue (as precisely and concretely described in the blog), in third person (e.g., "She smiles and looks away." or "He clenches his fists."). This should be an explicit behavioral description, not a vague feeling or summary.  
   - Write the answer in third person, as a direct, concrete description of the nonverbal/paralinguistic act.  
     
4. **Output Format:**  
   Return your result in XML as follows:  
```xml  
<data>  
  <type>Nonverbal_Expression</type>  
  <scenario>[Highly detailed scenario, grounded in the blog post, ending with a question asking what the person does nonverbally next]</scenario>  
  <answer>[the explicit nonverbal/paralinguistic expression as described in the blog]</answer>  
</data>  
```  
**If a high-quality nonverbal/paralinguistic sample cannot be produced, output only:**  
NULL  
   
---  
   
#### Strict Requirements  
- **Extract only when the blog directly supports a plausible, specific nonverbal/paralinguistic action or cue as the main character’s next behavior.**  
- **Do not fabricate context or expressive details not present in the blog post.**  
- **The scenario MUST be as specific and detailed as possible, using concrete information from the blog's language, events, setting, and descriptions of physical or paralinguistic behavior (but without including or paraphrasing the answer).**  
- **The answer MUST be what the character would literally do nonverbally or paralinguistically next—not summaries, generic reactions, or internal feelings.**  
- **If no suitable explicit nonverbal/paralinguistic behavior is available, output NULL.**  
   
---  
   
**Summary Table**  
   
| Step              | Description                                                                                 |  
|-------------------|---------------------------------------------------------------------------------------------|  
| Suitability Check | Is a next nonverbal/paralinguistic cue possible, clearly and concretely grounded in blog context? If no, output NULL. |  
| Scenario          | Third-person, present tense; as detailed and context-rich as possible from the blog; leads up to the nonverbal/paralinguistic moment, ends with a clear question. |  
| Question          | Concretely asks what the main character does nonverbally/paralinguistically next.            |  
| Answer            | Explicit description of the physical or prosodic expression as described in the blog.|  
| Only Nonverbal_Expression | "type" field is always "Nonverbal_Expression."                                                 |  
   
---  
   
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good to extract a good nonverbal/paralinguistic behavior story, please simply output "NULL". The raw blog post will follow.


"""

PROMPT_job_rowwise_roleplay_target_preference = """You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format `<type, scenario, answer>`, suitable for role-playing tasks focused on **preferences, interests, and lifestyle habits**.  
   
The *type* field should always be "Preferences, Interests, and Lifestyle Habits" for this task.  
   
Your goal is to generate data that predicts what lifestyle choice, interest, or preference the main character (i.e., the author) would express or enact in a clearly described situation based on the blog. The answer should be a **specific preference, interest, or future lifestyle action**, *not* a summary of past events or a reading-comprehension fact explicitly stated in the scenario.  
   
**CRITICAL REQUIREMENT:**  
   
When composing the scenario, you must **draw as much situational and behavioral detail, such as activities, routines, choices, context, and lifestyle cues, from the blog post as possible**. The scenario should richly describe the circumstances, environment, people involved, and any relevant context—using only grounded details provided in the blog. Avoid vagueness, and do not simply summarize; immerse the reader in the immediate, specific situation. **However, do NOT mention the particular preference, choice, or lifestyle habit you will ask about in the scenario.** Construct the scenario so that inferring the answer requires readers to reason about the character’s likely behavior given all the details, but the answer should not be directly stated.  
   
**If the provided blog post does not support constructing a high-quality scenario (**which is highly likely to happen, consider the fact that most raw blogs are quite dirty**) where the character’s preference, interest, or lifestyle habit (regarding a specific future choice, activity, or behavior) can be reasonably inferred—but NOT found verbatim—output NULL and do not attempt to extract a sample.**  
   
---  
   
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand the lifestyle, routines, choices, and preferences of the author, as well as their typical interests and daily habits.  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:  
   - The blog contains enough context to construct a rich, realistic scenario where the author’s likely lifestyle choice, preference, or interest can be inferred and is **not already plainly stated in the scenario**.  
   - You can formulate a question that requires predicting the main character’s upcoming preference, interest, or future lifestyle action (something they would do, enjoy, choose, or avoid) *not directly mentioned* in the scenario.  
   - The blog post provides explicit evidence of character, context, habits, or lifestyle that support inferring the answer.  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**  
   - **type** (always "Preferences, Interests, and Lifestyle Habits")  
   - **scenario:**  
     - Write a detailed scenario in third person (present simple tense), describing the setting, people involved, circumstances, recent events, and any emotionally or behaviorally relevant context based squarely on the blog, using as many specifics from the text as possible.  
     - End with a single, concrete question that explicitly asks about the character’s preference, interest, or lifestyle-related choice **not previously described in the scenario** (e.g., "Given this situation, what activity will the person choose to do next?", "What kind of food does the person most likely prefer to order at this cafe?", "Which hobby is the person likely to pursue after a stressful day?").  
     - Do NOT reveal, summarize, or paraphrase the answer in the scenario or in the question itself.  
     - Only one question is allowed.  
   - **answer:**  
     - Provide the specific preference, interest, or lifestyle habit (as a direct response to the question), grounded in the context/habits described in the blog post—but do not copy any answer directly from the scenario.  
   
4. **Output Format:**  
   Return your result in XML as follows:  
  
   ```xml  
   <data>  
     <type>Preferences, Interests, and Lifestyle Habits</type>  
     <scenario>[Highly detailed scenario grounded in the blog post, ending with a question about a preference, interest, or future lifestyle choice NOT mentioned in the scenario]</scenario>  
     <answer>[The character's preference/interest/lifestyle choice, written succinctly and specifically, but do not fabricate with your imagination]</answer>  
   </data>  
   ```  
  
   **If a high-quality sample cannot be produced, output only:**    
   NULL  
   
---  
   
#### Strict Requirements  
   
- **Extract only when the blog directly supports a plausible, specific preference, interest, or lifestyle habit as the main character’s next likely behavior—but does NOT explicitly state that choice in the scenario.**  
- **Do not fabricate context, events, or statements not given in the blog post.**  
- **The scenario MUST be as specific and detailed as possible, using only concrete information from the blog's language, events, setting, and details (but never containing the answer itself).**  
- **The question MUST ask for a lifestyle/personal preference or interest in a future action or choice NOT already mentioned in the scenario.**  
- **If no suitable sample is available, output NULL.**  
   
---  
   
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good to extract a good story, please simply output "NULL". The raw blog post will follow.


"""

PROMPT_job_rowwise_roleplay_target_mental = """You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for role-playing tasks focused on the author’s **cognitive style and reasoning patterns**. The *type* field should always be "CognitiveStyle" for this task.  
   
Your goal is to generate data that requires predicting the **mental process or internal reasoning** of the main character (i.e., the author) in a clearly described scenario based on the blog. The answer should reveal *how* the character thinks, decides, or processes information in that moment—for example, their explicit **thought process, step-by-step reasoning, decision heuristic, self-questioning, reflection, or assessment of a complex situation** as shown in the blog.  
   
**CRITICAL REQUIREMENT:**    
When composing the scenario, you must **draw as much situational and psychological detail (including setting, context, problems faced, options considered, prior events, emotional cues, cognitive tension, etc.) from the blog post as possible**. The scenario should richly describe the exact circumstances, environment, people involved, recent events, challenges, and *especially* the dilemma or question that causes the main character to reason or reflect—using all relevant details provided in the blog post leading up to the internal reasoning or thought process. Avoid vagueness or mere summary.  
   
**Do NOT describe, summarize, or paraphrase the actual step-by-step reasoning or mental process in the scenario.** The scenario should set up the situation that cues the thinking, but **the answer itself must reveal the internal cognitive style or logic that the person uses**.  
   
**The scenario should end with a *single, concrete question* that asks readers to infer or predict the author’s internal reasoning, cognitive steps, or thought process—not simply "what do they decide" or "how do they feel," and NOT a question that is merely answered by restating what was just described.** The scenario must be constructed so that the answer is not obvious reading comprehension, but requires inferring the specific style or manner of thinking from context.  
   
If the provided blog post does not support constructing a high-quality sample (**which is highly likely to happen, consider the fact that most raw blogs are quite dirty**) where the character's cognitive style or reasoning pattern is made explicit or clearly inferable, **output NULL and do not attempt to extract a sample**.  
   
---  
   
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand both the context and the author’s process of thinking, deliberation, or decision-making (as shown in the text).  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:  
   - The blog contains enough context to construct a realistic scenario where the author’s internal reasoning or cognitive process (e.g., their logic, heuristics, self-questioning, step-by-step process, or mental strategy) is supported by details in the text and can be explicitly written out.  
   - You can formulate a question that asks readers to infer or predict *how the person reasons in that situation*, *what logic or cognitive steps they apply* (not just what they do or decide).  
   - The blog post includes a described situation, challenge, reflection, or problem requiring reasoning, weighing, or deliberation—not just feelings or narration with no reasoning content.  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**  
   - **type (always "CognitiveStyle")**  
   - **scenario:**  
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the immediate situation, setting, people involved, circumstances, recent events, and especially the cognitive or decision-facing challenge (grounded in the blog), using as many specifics from the text as possible.  
     - **Do NOT reveal, paraphrase, or restate the actual reasoning process or cognitive style in the scenario.**  
     - End with a single, concrete question that requires readers to *predict or infer how the person thinks through the situation* (e.g., "What reasoning process does the person use to arrive at their decision?" "How does the person weigh the options in this moment?" "What is the person's internal thought process for evaluating the situation?").  
     - Only one question is allowed.  
   - **answer:**  
     - Provide the actual (as grounded in the blog) step-by-step reasoning, decision heuristic, internal dialogue (in the mind, not out loud), or cognitive pattern the character uses—written as the person's internal monologue or thought process.  
     - The answer should faithfully reflect the text’s depiction of the character’s reasoning (not invent what isn’t supported).  
   - Write both scenario and answer in third person.  
   
4. **Output Format:**  
   
Return your result in XML as follows:  
```xml  
<data>  
  <type>CognitiveStyle</type>  
  <scenario>[Highly detailed scenario, grounded in the blog post, ending with a question about the character's reasoning process, cognitive style, or specific chain of thought]</scenario>  
  <answer>[The actual internal reasoning or thought process, written as the character would think it, in third person, fully grounded by the raw blog, donot fabricate content]</answer>  
</data>  
```  
**If a high-quality cognitive style sample cannot be produced, output only:**    
NULL  
   
---  
   
#### Strict Requirements  
   
- **Extract only when the blog directly supports a plausible, specific example of the main character’s cognitive style or reasoning pattern in context.**  
- **Do not fabricate thought processes, logic, or context not present in the blog post.**  
- **The scenario MUST be specific and detailed, using concrete information from the blog’s events, problems, and settings—but without including or paraphrasing the actual reasoning pattern or answer.**  
- **The answer MUST be the actual internal reasoning, logic, or cognitive steps (not simply a summary, a decision, or an overt statement of feelings) that has been mentioned in the raw blog.**  
- **If no suitable reasoning sample is available, output NULL.**  
   
---  
   
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good to extract a good sample of reasoning or cognitive style, simply output "NULL". The raw blog post will follow.  
   
---

"""

PROMPT_job_rowwise_roleplay_target_socialinteraction = """You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for role-playing tasks focused on **Social Interactions**. The *type* field should always be "SocialInteractionPattern" for this task.  
   
Your goal is to generate data that predicts HOW the main character (i.e., the author) is  processing or managing a social interaction situation in terms of their interpersonal style, relationship strategies, preferences for conflict or cooperation, boundary-setting, assertiveness, conformity, or other relevant patterns—drawing as much as possible from the blogger’s reflections, attitudes, or decisions described in the blog.  
   
**CRITICAL REQUIREMENT:**  
When composing the scenario, you must **draw as much situational, social, and psychological detail as possible from the blog post.** The scenario should richly describe a specific interpersonal situation, the people involved, the relationship dynamics, relevant context, past or ongoing social events, and, crucially, any emotionally or cognitively salient aspects that have been discussed—using only what is grounded in the blog. The scenario should focus on (or at least heavily feature) the main character’s thought processes or strategies as they relate to their way of engaging with others.  
   
**Importantly:**    
- The scenario MUST NOT reveal, summarize, or paraphrase the main character’s mental or behavioral approach, judgment, or chosen interaction pattern—that is for the answer.    
- The scenario should end with a question that asks about *how* the main character manages, processes, or navigates the social situation (e.g., “How does the person approach the conversation with their colleague?” or “What strategy does the person use in dealing with the group’s disagreement?”), where the answer is not found in the scenario and requires readers to infer from the scenario.  
- **Do NOT ask about something already stated or paraphrased in the scenario. DO NOT make the QA a reading comprehension task.**  
   
If the provided blog post does not support constructing a high-quality scenario and answer about social interaction behaviors (**which is highly likely to happen**), output **NULL** and do not attempt to extract a sample.  
   
---  
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand the described social situation, the author’s role and perspective, and any discussion of their behaviors, attitudes, or strategies during interactions.  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:  
   - The blog contains a specific, grounded interpersonal context (e.g., handling conflict, negotiating boundaries, working in a group, supporting a friend, asserting a need, etc.) and describes enough context to model the person’s mental  or behavioral approach or style.  
   - You can pose a non-trivial, inferential question about *how* the main character manages or thinks about the social interaction, *not* a question about surface facts given directly in the scenario.  
   - The answer involves the main character’s mental or behavioral approach to others, as reflected in the blog; do not fabricate content that is not mentioned in the blog.  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**  
   - **type (always "SocialInteractionPattern")**  
   - **scenario:**    
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people involved, social dynamics, circumstances, recent events, and relevant context (all from the blog), with a focus on the social situation up to—but NOT including—the main character’s social interaction strategy or interpersonal decision.  
     - End with a *single, inferential question* that asks about the main character’s social-interactional approach, style, or strategy in this moment or next behaviors (e.g., "How does the person handle their disagreement with their friend?" or "What approach does the person take to maintain group harmony?")  
     - **Do NOT include the answer explicitly or implicitly in the scenario.**  
  
   - **answer:**    
     - Provide a concise statement (**not direct dialog**) describing the main character’s social interaction pattern, style, or interpersonal strategy, as mentioned from the blog.  
     - Write in third person, using behavioral or psychological language (e.g., "She tries to avoid direct confrontation by…" or "He asserts his needs clearly but remains polite…"). The answer should be clearly supported by the blog’s content, not invented based on your prediction or imagination.  
   
4. **Output Format:**  
   Return your result in XML as follows:  
   ```xml  
   <data>  
     <type>SocialInteractionPattern</type>  
     <scenario>[Highly detailed scenario, grounded in the blog post, ending with a single, non-redundant inferential question about the main character’s social-interactional behavior]</scenario>  
     <answer>[A concise, grounded statement of the main character’s social interaction pattern/approach/strategy in this scenario]</answer>  
   </data>  
   ```  
   **If a high-quality sample cannot be produced, output only:**    
   NULL  
   
---  
   
#### Strict Requirements  
   
- **Extract only when the blog directly supports a plausible, specific inference about the main character’s social interactions in a given scenario.**  
- **Do not fabricate context, mental states, or behavioral strategies not present in the blog post.**  
- **The scenario MUST be as specific and detailed as possible, using concrete information from the blog’s language, events, setting, dialogue, and social dynamics (but WITHOUT mentioning, revealing, or summarizing the answer in the scenario section).**  
- **The answer MUST be an description of the main character's social-interactional pattern/style/strategy, clearly grounded in the blog. Do not invented an answer based on your prediction or imagination.**  
- **If no suitable inferential sample is available, or if the blog is too vague, output NULL.**  
   
---  
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good enough to extract a highly specific, inferential scenario and answer related to social interaction patterns, simply output "NULL". The raw blog post will follow.  

"""

PROMPT_job_rowwise_roleplay_target_emotion = """You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for role-playing tasks focused on **Emotional Experience and Regulation**. The *type* field should always be "EmotionalExperience" for this task.  
   
Your goal is to generate data that predicts what the **main character (i.e., the author) feels or experiences emotionally in a clearly described scenario based on the blog. The answer should be the most salient, immediate emotion(s) or feeling(s) present for the character in that moment, not a summary of background mood or abstract description.**  
   
**CRITICAL REQUIREMENT:**    
When composing the scenario, you must **draw as much situational, psychological, and contextual detail from the blog post as possible**. The scenario should richly describe the circumstances, environment, people involved, recent actions, and any relevant context—using all grounded details provided in the blog leading up to the emotional moment. Avoid any vagueness, and do not simply summarize; immerse the reader in the specific situation, stopping just before the main character experiences, recognizes, or reacts to an emotional state. **Do not reveal or paraphrase the answer or the feelings themselves in the scenario.**  
   
If the provided blog post does not support constructing a high-quality scenario where the character’s immediate emotional experience is contextually clear and realistic (**which is highly likely to happen, consider the fact that most raw blogs are quite dirty**), output NULL and do not attempt to extract a sample.  
   
---  
   
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand both the psychological setting and the author’s experiential perspective.  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:    
   - The blog contains enough context to construct a realistic scenario where the author’s next emotional experience or regulation (i.e., what they actually feel or how they try to manage it in the moment) is supported by details in the text and can be written explicitly as an emotional state.    
   - You can formulate a question that directly asks, "What does the person feel next?" or a similarly specific variant.  
   - The blog post includes a described situation, incident, or interaction that grounds an immediate emotional experience or response (not just background mood, vague ruminations, or intellectual musings with no clear emotional episode).  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**  
   - **type (always "EmotionalExperience")**  
   - **scenario:**  
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people involved, circumstances, recent events, and any emotionally relevant context squarely based on the blog, using as many specifics from the text as possible. Write like a story or a novel. Lead up to—but NOT including—the main character’s next emotional experience or regulation.  
     - End with a single, concrete question explicitly about what the main character will feel or experience emotionally next in that situation.  
     - Do NOT reveal, summarize, or paraphrase the answer in the scenario.  
     - Only one question is allowed.  
   - **answer:**  
     - Provide the actual emotion or emotional state as the character would experience in that scenario, grounded in the blog post.  
     - Write the answer in third person, as a clear and precise emotional state or feeling (e.g., "She feels a surge of relief and happiness." or "He experiences frustration mixed with disappointment.").  
   
4. **Output Format:**    
   Return your result in XML as follows:  
   
```xml  
<data>  
  <type>EmotionalExperience</type>  
  <scenario>[Highly detailed scenario, grounded in the blog post, ending with a question asking what the person feels or experiences emotionally next]</scenario>  
  <answer>[The main character’s immediate emotional state, as directly felt, in third person—e.g., "She feels..."]</answer>  
</data>  
```  
   
**If a high-quality emotional experience sample cannot be produced, output only:**    
NULL  
   
---  
   
#### Strict Requirements  
   
- **Extract only when the blog directly supports a plausible, specific emotional experience as the main character’s next internal state.**  
- **Do not fabricate context, events, or emotions not present in the blog post.**  
- **The scenario MUST be as specific and detailed as possible, using concrete information from the blog's language, events, setting, and emotional descriptions (but without including, revealing, or paraphrasing the feeling in the scenario).**  
- **The answer MUST be what the character would actually feel or recognize emotionally in the moment—not a general trait, summary, or delayed reflection.**  
- **If no suitable emotional experience is available, output NULL.**  
    
---  
   
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good to extract a good scenario, please simply output "NULL". The raw blog post will follow.

"""

PROMPT_job_rowwise_roleplay_target_selfconcept = """You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for role-playing tasks that focus on the author’s self-concept and identity narratives—the ways the author makes sense of who they are, reflects on their life story, explains their personal growth, or describes internal self-understanding, including their sense of strengths, weaknesses, change, purpose, or role in the world. The *type* field should always be "SelfConcept" for this task.  
   
Your goal is to produce data that reveals **the content and structure of the author's self-concept, identity, or internalself-narrative**, delivering a high-quality *scenario* and prompting the reader to predict an aspect of the author's self-understanding or identity that is *not straightforwardly given* in the description, but must be carefully inferred.  
   
**CRITICAL REQUIREMENT:**    
When writing the scenario, you must **use as much grounded, specific detail from the original blog post as possible**—such as settings, events, background, emotional context, interpersonal dynamics, and especially any statements of internal reflection, self-evaluation, or narrative about who the author is or who they believe themselves to be. The scenario should *immerse* the reader in the immediate context or the author’s reflection, but **must not state, paraphrase, or reveal the specific self-concept or identity aspect that will form the answer**.  
   
**Instead, the scenario should end with a single, targeted question that asks for insight into the author's self-concept, beliefs about their identity, or personal self-narrative**—*specifically asking for something that was not already directly described or summarized in the scenario itself*. This question should require the reader to infer or reason about the answer, drawing on the scenario details and the broader blog context.    
- **Do NOT ask about something that is explicitly revealed in the scenario; do NOT frame it as a reading comprehension or retrieval question.**    
- Only ask for inferences about the author’s self-concept—their story or understanding of themself in context.  
   
If the provided blog post does **not** support constructing a high-quality scenario and answer centered on the author's self-concept (**which is highly likely to happen, consider the fact that most raw blogs are quite dirty**), **output NULL and do not attempt to extract a sample**.  
   
---  
   
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand the social setting, events, and especially the author’s internal reflections, self-understanding, identity statements, or life narratives.  
   
2. **Determine Suitability:**    
   Extract a sample ONLY IF:    
   - The blog contains enough context to construct a scenario illuminated by the author’s self-concept, identity narrative, or internal self-story.  
   - There is evidence of the author's mental processes concerning their own identity (e.g., reflections about personal change, values, self-perceived role or growth, etc.).  
   - You can formulate a question *not directly answerable from the scenario description* that requires the reader to reason about the author's self-concept, identity, or narrative.  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**    
   - **type (always "SelfConcept")**  
   - **scenario:**    
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, events, context, emotional atmosphere, and especially the author’s mental states, reflections, or situations as grounded in the blog post. *The scenario should lead up to—but never reveal—the specific self-concept/identity information sought by the prompt.*  
     - End the scenario with a **single inferential question** (e.g., “What does the person believe about themself in this situation?” or “How would the person describe their role in their life story at this moment?”).    
     - The question should **not** be about facts or summaries present in the preceding description, but require the reader to infer or hypothesize the author's underlying self-concept or identity narrative.  
  
   - **answer:**    
     - Write the answer as the author’s reflection on their self-concept, in third person, in direct quoted speech.  
     - The answer should be concrete and specific, and as much as possible, should be grounded in the style, language, and substance of the blog post.  
   
4. **Output Format:**    
   Respond only with the following XML format:  
```xml  
<data>  
  <type>SelfConcept</type>  
  <scenario>[Highly detailed and grounded scenario, ending with a single, targeted inferential question about the author’s self-concept]</scenario>  
  <answer>["..." — the direct speech revealing the author's self-concept, as they reflect or narrate internally]</answer>  
</data>  
```  
**If a high-quality self-concept sample cannot be produced, output only:**    
NULL  
   
---  
   
#### Strict Requirements  
- **Extract only when the blog clearly supports a plausible, richly detailed scenario about the author's identity/self-narrative.**  
- **Scenario MUST thoroughly use information from the blog's actual content—no invention, extrapolation, or speculation about past or future events/dialogue that the blog post does not support.**  
- **Scenario and question MUST require inference about the author’s self-concept, not mere restatement of already-given facts or reasoning presented in the scenario.**  
- **The answer MUST reveal what the person thinks or believes about themselves in direct (quoted) reflection, not merely feelings, actions, or summaries.**  
- **If the blog does not provide suitable material, output NULL.**  
   
---  
   
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not good enough to extract a high-quality self-concept/identity narrative sample, please simply output "NULL". The raw blog post will follow.


"""


PROMPT_job_rowwise_roleplay_target_adapability = """
You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, scenario, answer>, suitable for research on **Adaptability and Situation-based Flexibility** as manifested in personal narrative. The *type* field should always be "Adaptability" for this task.  
 
Your goal is to generate data that reveals the main character’s (i.e., the author’s) adaptive thinking process or situational flexibility—especially how they adjust their behavior, thinking, or strategies in response to changing circumstances or setbacks. The answer must capture the internal reasoning, mindset shift, or explicit action plan showing their adaptability, grounded in the blog post.  
   
**CRITICAL REQUIREMENT:**    
When composing the scenario, you must **draw as much situational and psychological detail from the blog post as possible**. Richly describe the context, situation, people involved, pressures or surprises encountered, and the recent events that require a flexible or adaptive response. (Use as many specifics as provided by the blog.)    
**STOP the scenario just before the main character’s moment of adaptation or mental shift,** so the answer can explicitly state how the author adapts, rethinks, or flexibly responds.    
**DO NOT summarize the answer or offer clues that make the scenario a mere comprehension/retrieval problem. Make sure the scenario provides context for adaptation, but leaves the actual adaptive response unmentioned.**  
   
If the blog post does **not** support constructing a high-quality scenario where the author’s adaptive thinking or situational flexibility (in thought or planned action) can be inferred, output only NULL.  Note that this is highly likely to happen, consider the fact that most raw blogs are quite dirty or are irrelevant to "Adaptability and Situation-based Flexibility".
   
---  
### Extraction Steps  
   
1. **Read and Analyze:**    
   Carefully read the blog post to understand the situation, challenges, and the author's internal perspective.  
   
2. **Determine Suitability:**    
   Extract a sample **ONLY IF**:  
   - The blog contains enough context to construct a realistic scenario—where the author’s flexibility, adjustment, or adaptation (in mindset or planned action) must be inferred as their next behavior, based on a change, challenge, feedback, or unexpected event.  
   - The scenario can lead up to a critical moment requiring adaptation, *without* specifying what the adaptation will be.  
   - A meaningful question can be posed that asks how the author adapts (in thought, attitude, or action), *but* cannot be answered by direct retrieval from the scenario text.  
  
   **If these criteria are not met, output NULL.**  
   
3. **Compose XML Fields:**  
   - **type**    
     - Always use "Adaptability"  
   - **scenario:**    
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people involved, what has just changed or gone wrong (or become challenging), and the specific situational context leading up to the author’s imminent adaptive response.    
     - End the scenario with a single, clear question—something like, "How does the person adapt their approach in this situation?" or "What mindset shift does the person make in response to this challenge?"    
     - **Do NOT include any summary or hints about the answer. The scenario must provide context for adaptation but keep the author’s next mindset or action unstated.**  
     - Only one question is allowed.  
  
   - **answer:**    
     - Based on the blog post, provide the author's next step in adaptation. This could be their revised plan, change in perspective, new strategy, or a mental shift—whatever is contextually supported as their immediate response for adapting, grounded in details from the blog.    
     - Write the answer in third person, explicitly describing what the author does or thinks to adapt, as inferred from the text.  
   
4. **Output Format:**  
   Return your result in XML as follows:  
   ```xml  
   <data>  
     <type>Adaptability</type>  
     <scenario>[Highly detailed scenario, grounded in the blog post, ending with a question asking about the author’s adaptation, with NO clue or summary of the answer]</scenario>  
     <answer>[Explicit description of the author’s adaptive mental or behavioral response, as supported by the blog post]</answer>  
   </data>  
   ```  
   **If a high-quality sample cannot be produced, output only:**    
   NULL  
   
---  
   
#### Strict Requirements  
- **Extract only when the blog directly supports a plausible, specific moment of adaptability or situational flexibility as the main character's next behavior (in thought or action).**  
- **Do not fabricate context, events, or internal changes not present in the blog post.**  
- **The scenario MUST be as specific and detailed as possible, using concrete information from the blog's language, events, setting, and descriptions (but NOT including or paraphrasing the answer).**  
- **The answer MUST state the actual adaptive behavior, plan, or internal shift, directly supported by the blog.**  
- **The scenario must NOT be answerable by direct retrieval or by merely summarizing text. It must require the reader to infer how adaptability or situational flexibility will be revealed next.**  
- **If no suitable sample can be extracted, output NULL.**  
   
---  
   
Now your task begins. Output only the XML result—do not include explanations. If the quality of the raw blog is not sufficient to extract a high-quality scenario for this behavior type, simply output NULL. The raw blog post will follow.


"""


PROMPT_job_rowwise_roleplayv2_target_culture = """   
You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, protagonist, scenario, answer>, suitable for personalized behavior prediction tasks focused solely on observable "Cultural Behaviors". For this task, the *type* field should always be "CulturalBehavior".  
   
Your goal is to generate data that predicts what the main character (i.e., the blog’s author) will DO next—a specific and observable act governed by a cultural custom, tradition, ritual, ceremony, etiquette, or any other practice with a clear cultural basis, in a richly described scenario drawn directly from the blog text. Focus only on behavior that is directly driven by cultural heritage, practices, or expectations.  
   
Assign the main character a unique, human-like name and refer to them with this name throughout the extracted scenario. Include a <protagonist> field in the output XML specifying this name.  
   
CRITICAL REQUIREMENTS:  
   
- When composing the scenario, use as much situational and cultural detail (such as location or country) from the blog post as possible.  
- The scenario should richly describe circumstances, people involved, environment, cultural background, recent events, and specifically those cultural vectors and expectations shaping behavior—using all grounded details from the blog, stopping immediately before the main character performs the cultural act.  
- Do NOT use or paraphrase "the author", "the person",  or "the blogger";  ALWAYS refer to the protagonist by their assigned unique, human name.  
- Do NOT include or reveal the answer in the scenario.  
- Output NULL if the provided blog does not support constructing a high-quality scenario of an immediate observable culture-driven act (**which is highly likely to happen, consider the fact that most raw blogs are quite dirty and usually not related to cultural behaviors**).  
- Strictly avoid inventing context, events, or cultural behaviors not mentioned in the blog text.  
   
---  
   
### Extraction Steps  
   
1. Read and Analyze:  
   - Carefully read the blog post, focusing on the author’s descriptions of customs, rituals, traditions, ceremonies, etiquette, and references to cultural background.  
   
2. Determine Suitability:  
   - Extract a sample ONLY IF:  
     - The blog post contains enough context to construct a realistic scenario where the blogger’s immediate next observable action is dictated by cultural (not general social) expectations, customs, or traditions, and can be explicitly described.  
     - You can formulate a short question.  
     - The blog describes participation in a cultural setting, event, ritual, or moment where a specific behavior is dictated by heritage, tradition, cultural practice, or explicit etiquette.  
   - If not, output NULL.  
   
3. Compose XML Fields:  
   - type: Always "CulturalBehavior".  
   - protagonist: Assign and use a random, human-like name for the main character (blogger), and refer to them only by this name throughout scenario and answer.  
   - scenario:  
     - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people, circumstances, cultural context, and recent events.  
     - End with a concrete, explicit question about [Name]'s next act (not a summary or abstract reflection).  
     - Do not reveal, summarize, or paraphrase the answer in the scenario. Do not use the answer’s wording or action.  
     - Only one explicit question is allowed.  
   - answer:  
     - Provide the immediate, observable cultural act (as third person, using the given name), which is grounded in the blog context. Do not fabricate behaviors that are not in the blog.  
   
4. Output Format:  
   Return your result in the following XML format:  
   
```xml  
<data>  
  <type>CulturalBehavior</type>  
  <protagonist>[name]</protagonist>  
  <scenario>[Detailed, third-person, culturally grounded scenario ending with a question about the next cultural behavior]</scenario>  
  <answer>[Immediate, observable cultural action performed by the protagonist, grounded in the blog]</answer>  
</data>  
```  
   
If a high-quality, grounded sample cannot be produced, output only:  
NULL  
   
---  
   
#### Strict Requirements  
   
- Extract only when the immediate next observable act is clearly governed by explicit cultural practice, custom, or tradition and is sufficiently supported in the blog context.  
- Do not consider generic social norms or workplace/organizational/group behavioral codes unless they are directly grounded in culture or heritage.  
- Scenario must be in third person, use the assigned name, be as scenario-rich and context-specific as possible, utilizing concrete information from the blog’s events, cultural references, and background (but not include or paraphrase the answer).  
- Answer must be the immediate observable act (third person, using the unique name) in that cultural context—not a summary, guideline, or paraphrase. Do not fabricate behaviors that are not in the blog.   
- If no suitable culture-driven behavior is available, output only NULL.  
   
---  
   
**Summary Table**  
   
| Step              | Description                                                                                                               |  
|-------------------|---------------------------------------------------------------------------------------------------------------------------|  
| Suitability Check | Is a next culturally grounded, observable action possible and clearly based on blog context? If not, output NULL.          |  
| Scenario          | Third-person, protagonist-name, present tense; detailed and context-rich; leads up to the act, ends with concrete prompt. |  
| Question          | Concretely asks what the protagonist does next according to specific cultural custom/tradition/expectation.                |  
| Answer            | Immediate, observable cultural act (third person, using protagonist name), exactly as grounded in blog text.              |  
| Only Cultural     | The "type" field is always "CulturalBehavior".                                                                           |  
   
---  
   
**If a high-quality scenario for this behavior type cannot be supported by the quality of the raw blog, return only:**  
   
```  
NULL  
```  
   
---  
   
Your task begins now. Output only the XML result—do not include explanations. If the quality of the raw blog is not good enough to extract a scenario, simply output "NULL". The raw blog post will follow.




"""

PROMPT_job_rowwise_roleplayv2_target_value = """
You are given a blog post written by a single person. Your task is to extract a high-quality data sample in the format <type, protagonist, scenario, answer>, suitable for personalized behavior prediction tasks focused solely on observable or articulated "Human Values".  
   
For this task, the *type* field should always be "HumanValue".  
   
Your goal is to generate data that predicts what value-driven decision, judgment, or statement the main character (i.e., the blog’s author) will make next—a clearly articulated or demonstrable value (e.g., value judgment, worldview, declaration of right/wrong, prioritization or choice aligned with identifiable values), in a scenario richly described and supported by the blog text.  
   
Focus only on value expressions, value-driven decisions, or explicit statements or behaviors reflecting underlying values (such as those defined in Schwartz’s Human Values model). Mere preferences, neutral observations, or generic statements (not demonstrably aligned to a clear value) are not the target of this task.  
   
Assign the main character a unique, human-like name and refer to them with this name throughout the extracted scenario.  
   
Include a <protagonist> field in the output XML specifying this name.  
   
CRITICAL REQUIREMENTS:  
   
- When composing the scenario, use as much situational and psychological detail from the blog as possible.  
- The scenario should richly describe circumstances, people involved, environment, cultural or personal background, recent events, and specifically those value-related vectors shaping the main character’s thinking—using all grounded details from the blog, stopping immediately before the value is articulated or demonstrated.  
- NEVER use or paraphrase "the author", "the person", or "the blogger"; ALWAYS use the assigned unique, human name.  
- Do NOT include or reveal the answer in the scenario.  
- Output NULL if the provided blog does not support constructing a high-quality scenario of an immediate, explicit, observable or articulated value expression (**which is highly likely to happen, consider the fact that most raw blogs do not overtly articulate values or value conflict**).  
- Strictly avoid inventing context, events, behaviors, or values not found in the blog text.  
   
---  
   
### Extraction Steps  
   
1. Read and Analyze:  
   - Carefully read the blog post, focusing on the author’s descriptions of value-laden decisions, judgments, overt statements, worldviews, and instances where they cite, prioritize, articulate, or demonstrate a value in action or speech.  
   - Value signals might include clear justifications, moral or ethical statements, prioritizing tasks/people/activities, condemnations/approvals of actions, or reasoning linked to principles, worldviews, or value labels (e.g., "honesty is important", "I believe in helping others", "I value my independence", etc.).  
   
2. Determine Suitability:  
   - Extract a sample ONLY IF:  
      - The blog post contains enough context to construct a realistic scenario where the protagonist’s immediate next value statement or value-driven decision/action/judgment is clearly and explicitly dictated by a recognizable human value (as defined above).  
      - You can formulate a concrete question about what value the main character expresses or demonstrates next.  
      - The blog describes a moment where a decision, judgment, statement, or overt rationale is clearly shaped by a value or worldview, with enough context to predict the next value-driven act or utterance.  
   - If not, output NULL.  
   
3. Compose XML Fields:  
   - type: Always "HumanValue".  
   - protagonist: Assign and use a random, human-like name for the main character (blogger), and refer to them only by this name throughout scenario and answer.  
   - scenario:  
      - Write a detailed scenario in third person (present simple tense), thoroughly describing the setting, people, circumstances, relevant context, and recent events.   
      - End with a concrete, explicit question about [Name]'s next statement, judgment, or action that can reflect their values.  
      - Do not reveal, summarize, or paraphrase the answer in the scenario.  
      - Only one explicit question is allowed.  
   - answer:  
      - Provide the immediate, overt or articulated statement/action/judgment (as third person, using the given name), grounded in the blog context. Do not fabricate details, behaviors, or values that are not mentioned in the blog.  
   
4. Output Format:  
   Return your result in the following XML format:  
   
```xml  
<data>  
  <type>HumanValue</type>  
  <protagonist>[name]</protagonist>  
  <scenario>[Detailed, third-person, value-rich scenario ending with a question about the next  statement/judgment/action]</scenario>  
  <answer>[Behavior of the protagonist, grounded in the blog]</answer>  
</data>  
```  
   
If a high-quality, grounded sample cannot be produced, output only:  
```  
NULL  
```  
   
---  
   
#### Strict Requirements  
   
- Extract only when the immediate next value-driven expression or action is clearly articulated or demonstrated and sufficiently supported in the blog context.  
- Scenario must be in third person, use the assigned name, be as scenario-rich and context-specific as possible, utilizing concrete information from the blog’s events, value signals, and background (but not include or paraphrase the answer).  
- The answer must be the immediate, explicit statement, judgment, or value-articulating act (third person, using the unique name), fully grounded in the blog context—not an abstract summary, guideline, or paraphrase.  
- If no suitable value-driven expression is available, output only NULL.  
     
                                                                                      |  
 
---  
   
Your task begins now. Output only the XML result—do not include explanations. If the quality of the raw blog is not good enough to extract a scenario, simply output "NULL". The raw blog post will follow.

"""

PROMPT_judger_user_scenario_question_answer = """  
You are a strict, expert Social QA quality reviewer, tasked with strictly evaluating the quality of automatically-generated social question-answer (QA) data samples. Each data sample is purportedly derived from a set of original user-generated blog posts.  
   
**You will be given:**  
1. A set of real blog posts (“the blogs”).  
2. A list of generated QA data samples. Each sample contains a scenario, a question, and an answer—each sample references specific blog post content.  
   
**Your task:**    
Assess each QA data sample individually. For each sample, carefully compare it to its referenced blog post(s) and evaluate it as a standalone piece of dialogue. For every single sample, you must provide a separate, structured evaluation.  
   
---  
   
### **Evaluation Criteria (applied to each data sample):**  
   
#### 1. **Hallucination**  
*Does the sample faithfully represent the referenced blog, without invention, omission, or distortion?*    
- Judge all parts (scenario, question, answer).  
- Score higher if all key details are fully supported by the blog.  
- Deduct if content is invented, speculative, or key facts are omitted.  
   
**Scoring (per-sample):**  
- 10: No hallucination; every part justified by the blog.  
- 8: Minor, non-critical issues.  
- 6: Noticeable, but not central, problems.  
- 4: Major misrepresentations.  
- 1: Entirely or mostly unrelated/invented.  
   
#### 2. **Fidelity (Self-Consistency & Completeness)**  
*Is the sample coherent, specific, detailed, and logical when read by itself?*    
- Scenario should be vivid, concrete, and non-generic.  
- Scenario, question, and answer must be logically connected, non-redundant, and not contradictory.  
- Question must be clear; answer must specifically address the question.  
   
**Scoring (per-sample):**  
- 10: Fully natural, rich, and consistent.  
- 8: Almost always detailed and well-connected; minor issues only.  
- 6: Occasionally generic; minor disconnects.  
- 4: Frequent poor formation or weak logical connection.  
- 1: Mostly incoherent or redundant.  
   
#### 3. **Novelty & Interest**  
*How original and engaging is the sample? Is it formulaic or memorable?*  
   
**Scoring (per-sample):**  
- 10: Highly original and impactful.  
- 8: Generally interesting, with some unique aspects.  
- 6: Somewhat generic, but mildly engaging.  
- 4: Largely unremarkable or formulaic.  
- 1: Completely generic, repetitive, or copy-paste.  
   
#### 4. **Overall Quality**  
*Your overall judgment for this data sample, balancing all factors above.*  
   
- Assign an integer score from 1 (worst) to 10 (best).  
   
---  
   
**Instructions:**  
   
- Review each QA data sample one by one, together with its referenced blog post(s).  
- For each data sample, assign ratings for all four criteria above and write a concise explanation justifying your evaluation, grounding your comments in evidence.  
- Output your assessment for each data sample in the following XML format:  
   
```xml  
<data>  
  <explanation>[Short summary of your analysis and reasoning]</explanation>  
  <hallucination>[INTEGER 1-10]</hallucination>  
  <fidelity>[INTEGER 1-10]</fidelity>  
  <novelty>[INTEGER 1-10]</novelty>  
  <overall>[INTEGER 1-10]</overall>  
</data>  
```  
   
- Output a separate <data>…</data> XML block for each evaluated sample. Your final result should be a list of such XML objects, one per data sample, in the order they were presented.  
   
- Only output the XML result, with no additional explanations or commentary.  
   
---  
   
Now the evaluation task begins.

"""

PROMPT_job_rowwise_scenario_question_answer_from_single_blog_emphasizereason_v2 = """  
You are a data-extraction assistant. Your task is to read a raw blog post and generate a high-quality, challenging question-and-answer pair centered on social reasoning and explanation of human behaviors. So the question is asking about why people act, feel, or react in certain ways.  When describing the scenario, do not include the reason or motivation for the behavior, but focus on the observable actions, reactions, or emotional responses of the main character (i.e., the author of the blog). The purpose of the QA data sample is to challenge readers to think about the motivations or reasons behind the behavior, which are explicitly stated in the blog.
  
1. **Input**  
   – A single, human-written blog post.  
   
2. **Output**  
   – Either the XML snippet defined below (with one scenario, a "why" question, and a reason-based answer) or "NULL" if no suitable pair can be extracted.  
   – Outputting "NULL" is likely, as many blogs do not sufficiently detail both behavior and explicit reasoning.  
   
3. **Scenario requirements**  
   – Draw directly and richly from the blog, including observable human behaviors: actions, reactions, or emotional responses, as well as setting/context, time/place, and participants.  
   – Focus on one clear behavioral event, choice, or emotional response by the main character (i.e., the author of the blog), but describe it neutrally—**do not include the revealed motivation or reason**.  
   – Preserve sequence and nuance; do not over-summarize or embellish beyond the blog.  
   – Do not invent content or infer inner thoughts — stick strictly to what the blog explicitly states.  
   – Write in the third person (e.g., "The person...") and as a narrative—do not refer to "the blog" or "the author said".  
   
4. **Question requirements**  
   – Provide exactly one concise "why" question about the main human behavior of the main character in the scenario (e.g., Why did the person do X? Why did the person feel Y?).  
   – The answer should not be revealed in the scenario description. The question should challenge readers to reason about motivation, cause, or explanation, grounded in the facts from the blog.  
   – Do not ask about information not related to explaining behaviors (such as simple factual or visual details).  
   
5. **Answer requirements**  
   – Provide a natural, complete narrative answer, strictly grounded in the blog’s content and explicitly reflecting the reason(s) or motivation(s) as stated by the main character.  
   – The answer must be a direct explanation, as described in the blog, for the specific behavior in question.  
   – Do not infer or invent — use only what has been explicitly stated from the blog.  
   – Write in the third person and as a narrative—do not refer to "the blog" or "the author said".  
   
6. **Null Condition**  
   – If the blog lacks an explicit stated reason (motivation/explanation) for that behavior, output exactly: NULL  
   – This condition is common; many blogs do not provide clear motivation or reasons.  
   
**Output Format:**    
Return the extracted data in this XML format:  
```xml  
<data>  
  <scenario>A vivid and detailed scenario description in the third-person perspective (e.g., "The person...", "The author...", "The blogger..."), describing the specific behavior (without revealing the reason).</scenario>  
  <question>A "why" question asking for the reason behind the main character's action or emotional response.</question>  
  <answer>The reason or explanation for the behavior, extracted word-for-word or paraphrased strictly from the blog, in the third person, with no added inference.</answer>  
</data>  
```  
   
Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.


"""
 
PROMPT_job_rowwise_scenario_question_answer_from_single_blog_emphasizereason_v3 = """You are a data-extraction assistant. Your task is to read a raw blog post and generate a high-quality, challenging question-and-answer pair centered on social reasoning and explanation of human behaviors. The question must challenge readers to think about and guess why the main character acts, feels, or reacts in a certain way.  
   
**Critical instructions:**    
- When describing the scenario, you must NOT include or imply the character’s motivation, reason, or explanation for their behavior—focus solely on what can be directly observed or reported about *what* happened, not *why* it happened.  
- Avoid any language or detail in the scenario that reveals, hints at, or summarizes the cause or motivation for the behavior.    
- The scenario MUST remain strictly factual and limited to observable actions, emotional responses, context, participants, or dialogue, as described in the original blog.  
   
1. **Input**  
    – A single, human-written blog post.  
   
2. **Output**  
    – Either the XML snippet defined below (with one scenario, a "why" question, and a reason-based answer) or "NULL" if no suitable pair can be extracted.  
    – Outputting "NULL" is likely, as many blogs do not provide both a clear behavior and an explicit stated motivation/explanation, or they are not about the author's real behaviors/emotions.  
   
3. **Scenario requirements**  
    – Draw directly and richly from the blog, including observable human behaviors: actions, reactions, or emotional responses, as well as setting/context, time/place, and participants.  
    – Focus on one clear behavioral event, choice, or emotional response by the main character (i.e., the author of the blog), but **describe it strictly in terms of what happened, not why**.  
    – **Under no circumstances should the scenario include any explanation, intention, motivation, or reasoning described or implied by the main character.**  
    – Write in the third person (e.g., "The person...") and as a narrative—do not refer to "the blog" or "the author said".  
   
4. **Question Requirements**    
   – Craft exactly one clear and focused "why" question that addresses the primary human behavior of the main character in the scenario (e.g., Why did the person do X? Why did the person feel Y?).    
   – Ensure that the answer is neither revealed nor implied in the scenario description.    
   – The question should encourage readers to think critically about the motivation, causes, or underlying explanation. While the answer should be rooted in the facts presented in the blog, it must not be explicitly outlined in the scenario. This allows readers to rely on their social commonsense and imagination to infer the answer.  
   
5. **Answer requirements**  
    – Provide a natural, complete narrative answer, strictly and only grounded in the blog’s content and explicitly reflecting the reason(s) or motivation(s) as *directly* stated in the original blog content.  
    – The answer must be a direct explanation for the specific behavior in question, using only what is *explicitly* stated in the blog—do not infer or invent.  
    – Write in the third person and as a narrative—do not refer to "the blog" or "the author said".  
   
6. **Null Condition**  
    – If the blog lacks an explicit stated reason (motivation/explanation) for that behavior, output exactly: NULL  
   
**IMPORTANT:**    
When writing the scenario:    
- Only describe observable events, actions, reactions, or feelings.    
- **Do NOT** include any reference—direct or indirect—to the character’s motivation, reasoning, intention, or cause for their behavior.    
- All information about "why" the behavior occurred must be reserved exclusively for the answer.  
   
**Output Format:**    
Return the extracted data in this XML format:  
   
```xml  
<data>  
  <scenario>A vivid and detailed scenario description in the third-person perspective (e.g., "The person...", "The author...", "The blogger..."), describing the specific behavior (WITHOUT revealing or implying the reason).</scenario>  
  <question>A "why" question asking for the reason behind the main character's action or emotional response.</question>  
  <answer>The reason or explanation for the behavior, extracted or paraphrased strictly from the blog, in the third person, with no added inference.</answer>  
</data>  
```  

Now the task begins. Below are the person's raw blog. Please only output the xml result. Do not include any additional explanatory text. If no suitable pair can be extracted, output exactly: NULL  
   
---  

"""

PROMPT_writing_style = """I will provide you with some blogs written by a user. Please summarize the writing style of the user, with a third-person narrative perspective such as "The user ...".

Please only output your answer about the writing style, do not output other explanatory words.
Below are the blogs:

"""

PROMPT_blog_summary = """Please summarize the main topic of the below blog in one sentence. Only output the one-sentence result, do not output other explanatory words.
The blog content is:

"""

PROMPT_scenario_short_v4 = """You are now a helpful assistant tasked with generating useful data. I will provide you with a raw blog post from a blogger. Your job is to extract a short social scenario from the blog post.  
   
A short scenario is a concise summary (no more than 100 words) of a single social event mentioned in the blog post. The scenario should describe a specific context (such as time, location, social relationship, or other useful information) and include what people are doing, planning to do, thinking, or feeling. The scenario must focus on a social event or activity involving at least two characters.  
   
For privacy, replace any real person’s name mentioned in the blog with placeholders: “PersonA”, “PersonB”, etc. So in your persona or scenario description, you can only use name placeholders to mention people, avoid using real names. Exception: mention of celebrities's names and movie/novel character's names are allowed. 
   
Please output a JSON object containing three fields:  
- characters: A dictionary that briefly introduces each character involved in the scenario. Use “PersonA”, “PersonB”, etc., as keys, and provide a one-sentence persona for each (do not mention real human name in persona).  
- scenario: A short paragraph describing the scenario. Use the name placeholders when describing the scenario. (exception: mention of celebrities's names and movie/novel character's names are allowed)  
- quality: The quality of the extracted scenario, based on how well the original blog post supports it. Choose one: "low", "medium", or "high".  
   
Extract a scenario only if the blog post features concrete social events involving the author. If the content is unrelated to personal social events (e.g., a novel plot, news article, promotional ad, product description, reposted poem, or random venting with no clear event), simply output an empty JSON object: {}. This happens often.  
   
Ensure all scenario details are directly fully supported by the blog post; do not invent or infer content.  
   
Begin your task now. Please only output the JSON object in the following format:  

{"characters": {"PersonA": "one sentence of persona for PersonA, do not mention real human names", "PersonB": ...}, "scenario": "the content as one paragraph", "quality": "value from [low, medium, high]"}  

Only output the JSON result. DO NOT include any explanations or other text.
   
The author's current raw blog post is:  
---

"""

PROMPT_scenario_long_v4 = """You are now a helpful assistant tasked with generating useful data. I will provide you with a raw blog post from a blogger. Your job is to extract a detailed social scenario from the blog post.  
   
A detailed scenario is a comprehensive summary (between 80 and 500 words) of a single social event or interaction mentioned in the blog post. The scenario should describe a specific context—such as time, location, social relationships, and any other relevant information—and include a thorough description of what people are doing, planning to do, thinking, or feeling. The scenario must focus on a social event or activity involving at least two characters, and should capture the nuances, motivations, and emotions present in the event to convey a clear sense of the situation.  
   
For privacy, replace any real person’s name mentioned in the blog with placeholders: “PersonA”, “PersonB”, etc. So in your persona or scenario description, you can only use name placeholders to mention people, avoid using real names. Exception: mention of celebrities's names and movie/novel character's names are allowed.   
   
Please output a JSON object containing three fields:  
   
- characters: A dictionary that briefly introduces each character involved in the scenario. Use “PersonA”, “PersonB”, etc., as keys, and provide a one-sentence persona for each  (do not mention real human name in persona).  
- scenario: A detailed paragraph describing the scenario, between 100 and 500 words in length. Describe the content with the name placeholders and do not mention character names (exception: mention of celebrities's names and movie/novel character's names are allowed).  
- quality: The quality of the extracted scenario, based on how well the original blog post supports it. Choose one: "low", "medium", or "high".  
   
Extract a scenario only if the blog post features concrete social events involving the author or the author's emotional feelings about such events. If the content is not detailed enough to support a scenario of 100 words or more, or is unrelated to personal social events (e.g., a novel plot, news article, promotional ad, product description, reposted poem, or random venting with no clear event), simply output an empty JSON object: {}. This happens often.  

All the generated content should be fully grounded by the raw blog post. Do not imagine or infer any non-mentioned content, even for enriching more details is not allowed. Be strictly factual according to the original text.  
   
Begin your task now. Please only output the JSON object in the following format:    
{"characters": {"PersonA": "one sentence of persona for PersonA, do not mention real human names", "PersonB": ...}, "scenario": "the content as one paragraph", "quality": "value from [low, medium, high]"}  
   
Only output the JSON result. DO NOT include any explanations or other text.  Note that simply output an empty JSON object: {} if the quality of the blog post is not good enough as mentioned above.
   
The author's current raw blog post is:    
---

"""

PROMPT_event_thinking_pair = """You are a helpful assistant tasked with extracting structured data from raw blog posts. Each time, I will provide you with a blog post written by an author. If the blog is related to human behavior or social events and includes the author’s reflections on their thoughts or feelings about these events, you should extract a useful "human cognitive data pair". If the blog does not meet these criteria, output only an empty JSON object: {}.  
   
A human cognitive data pair is a JSON object consisting of two fields:  
   
1. **overt event**: A clear and detailed description of a human behavior or social event mentioned in the blog, written in the first-person perspective of the author.    
2. **mental process**: The author’s inner thoughts, feelings, or emotional response to the overt event, also captured in the first-person perspective, grounded entirely in the blog post (e.g., emotions, values, preferences, opinions).    
  
**Guidelines:**  
- Extract the data pair only if the blog explicitly describes a specific social event involving the author's mental process.  
- Avoid any kind of inference, imagination, or addition of details not explicitly mentioned in the blog. Both fields must be fully grounded in the original text.  
- If the blog content is too vague, unrelated (e.g., fictional stories, news articles, advertisements, product descriptions, reposted content, or unclear personal venting), or lacks detail about both a social event and mental process, simply return an empty JSON object: {}.  
   
**Output format:**    
Always return the result as a JSON object in the following format:    
```json  
{"overt event": "content as a single paragraph", "mental process": "content as a single paragraph"}  
```    
If no valid data pairs can be extracted, simply return:    
```json  
{}  
```    
  
**Important:** Do not include any explanations, comments, or additional text—only output the JSON object.  
   
The author's current raw blog post is:  
   
---

"""

PROMPT_judge_event_thinkjing = """    
You are an expert reviewer of social mental thinking data. Given (1) a single original blog post and (2) a generated data sample derived from that blog (consisting of an "overt event" and a "mental process"), your job is to carefully evaluate and rate the quality of the data sample across several dimensions.  
   
**Special Rule:**    
If the data content is empty (for example, ```json{} ``` or similar), assign a score of **0** for every category.  
   
**Evaluate the following aspects:**  
   
1. **Hallucination**    
   Does the overt event and the mental process remain true to the **main story, central events, and overall intent** of the original blog? Are all key details, especially those critical to understanding the main scenario, either clearly present in the blog or plausible, justifiable inferences? Minor, less important details can be inferred or slightly altered as long as the main narrative remains faithful.    
   **Score 10:** No invented or speculative information **regarding the main story and core scenario**; all major content in the overt event and mental process is clearly justified by the blog. Minor/inconsequential details are either present or represent reasonable inferences.    
   **Score 8:** One or two small, highly likely inferences or omittable details stray from the blog, but the **main story and scenario are entirely grounded and unchanged**.    
   **Score 5:** Several details not directly supported by the blog, including possibly a slight modification to the main story—**but it still represents the same central scenario and intent**. Minor inferences are acceptable, but **no major invention of the overall event**.    
   **Score 1:** Any significant change or invention in the main story, scenario, or central event compared to the blog, OR many unsupported facts even if the main idea is retained; demonstrates clear divergence from the blog in both spirit and detail.    
  
2. **Coverage**    
   How well does the data sample represent and extract salient, interesting, and unique elements from the blog?    
   **Score 10:** Most or all notable and distinctive points from the blog are well represented.    
   **Score 8:** Captures most important aspects, but omits or flattens a few details.    
   **Score 5:** About half of the salient information is captured; some significant stories are missing or under-emphasized.    
   **Score 1:** Little of the blog’s meaningful content is present; major omissions.    
  
3. **Fidelity (Data Sample Self-Consistency & Quality)**    
   *(Consider the data sample itself, regardless of the blog.)*    
   - Is the overt event vivid, concrete, detailed, interesting, and coherent—not generic or trivial?    
   - Are the overt event and mental process logically connected and non-redundant?    
   - Does the mental process show thoughtful depth and specificity?    
   **Score 10:** All fields are natural, complete, and richly detailed; overt event is engaging; mental process is deep and complementary.    
   **Score 8:** Sample is mostly solid and engaging, but may lack depth or vividness in some places.    
   **Score 5:** Sample is complete and understandable, but is somewhat generic, routine, or underdeveloped.    
   **Score 1:** Content is generic, duplicative, trivial, unnatural, or weak.    
  
4. **Novelty & Interest**    
   *(Consider the data sample itself, regardless of the blog.)*    
   How much does the sample feel interesting, distinctive, and socially or emotionally resonant, rather than formulaic or generic?    
   **Score 10:** Highly distinctive, memorable, and evocative; clear emotional or cognitive resonance.    
   **Score 8:** Largely distinctive and interesting, but contains minor generic segments.    
   **Score 5:** Moderately engaging, but features significant generic or routine stretches.    
   **Score 1:** Largely generic, formulaic, or uninteresting.    
  
5. **Leakage (Information Overlap between Overt Event & Mental Process)**    
   *(Consider the data sample itself, regardless of the blog.)*    
   To what degree is the mental process already revealed or obvious from the overt event description? Higher scores mean the mental process requires social reasoning or inference (not just copying overt event details); lower scores mean much of the mental process is simply restated or directly found in the overt event, requiring little reasoning to deduce.    
   **Score 10:** Very little to no direct mental process content is present in the overt event; interpreting the mental process requires significant inference or reasoning.    
   **Score 8:** Some minor details of the mental process are found in the overt event, but critical information still requires nontrivial inference.    
   **Score 5:** Substantial overlap between overt event and mental process; the mental process is partially but not fully restated or deducible from overt event alone.    
   **Score 1:** Most or all of the mental process is directly stated in the overt event; almost no reasoning is required to infer it.    
  
6. **Richness of Overt Event**    
   *(Judged solely by the overt event field, regardless of the blog or mental process.)*    
   How detailed and complete is the description of the overt event? Is it vivid, concrete, and informative, providing a clear picture of what happened?    
   **Score 10:** Exceptionally rich and comprehensive; the overt event is described with vivid detail and clear context, leaving little ambiguity.    
   **Score 8:** Strong and mostly thorough, but a few small details could be clearer or more developed.    
   **Score 5:** Moderately detailed; the event is understandable but lacks depth or completeness in some key areas.    
   **Score 1:** Sparse, superficial, or generic; lacks necessary detail for understanding or is ambiguous.    
  
7. **Richness of Mental Process**    
   *(Judged solely by the mental process field, regardless of the blog or overt event.)*    
   How detailed and complete is the description of the mental process? Does it reveal specificity, nuance, and insight into the individual’s internal reasoning, emotions, or thought process?    
   **Score 10:** Exceptionally nuanced, specific, and evocative; the mental process is richly developed and revealing.    
   **Score 8:** Strong and thoughtful, though minor points may be under-explored.    
   **Score 5:** Moderately developed; the mental process is present but could show much more depth, complexity, or specificity.    
   **Score 1:** Minimal, generic, or surface-level; provides little insight into internal thought or feeling.    
  
8. **Overall Quality**    
   Your holistic judgment of the data sample’s overall quality, based on accuracy, informativeness, distinctiveness, coverage, and naturalness, scored as an integer from 1 to 10.  
   
---  
   
**Output Format:**    
Please provide your ratings using the following template:  
```xml  
<data>  
  <explanation>[Short summary of your analysis and reasoning]</explanation>  
  <hallucination>[INTEGER 0-10]</hallucination>  
  <coverage>[INTEGER 0-10]</coverage>  
  <fidelity>[INTEGER 0-10]</fidelity>  
  <novelty>[INTEGER 0-10]</novelty>  
  <leakage>[INTEGER 0-10]</leakage>  
  <overt_event_richness>[INTEGER 0-10]</overt_event_richness>  
  <mental_process_richness>[INTEGER 0-10]</mental_process_richness>  
  <overall>[INTEGER 0-10]</overall>  
</data>  
```  
   
---  
   
**Instructions:**    
Rate each data sample thoughtfully and fairly, paying attention to nuances in both blog and sample. Be strict—reserve high scores for truly strong samples, and use the full scoring scale as appropriate. Judge hallucination and coverage strictly according to the original blog; judge fidelity, novelty, leakage, overt event richness, and mental process richness based only on the data sample. Only output the xml format result, do not output other explanatory words.  
   
If the data content is empty (e.g., ```json{} ```), assign a score of **0** for all categories and state this in the explanation.  
   
---  
   
Now the task begins.

"""

PROMPT_judge_user_scenario_v4 = """ 
You are an expert reviewer of social scenario extraction data. Given (1) an original blog post and (2) a generated data sample extracted from that post (consisting of a "characters" dictionary and a "scenario" string), your job is to carefully evaluate and rate the quality of the data sample across several dimensions.  
   
Special Rule:    
If the data content is empty (for example, {} or similar), assign a score of 0 for every category.  
   
Evaluate the following aspects:  
   
1. Hallucination    
Does the scenario accurately reflect the main story, central events, and overall intent of the original blog? Are all key details, especially those critical to understanding the main scenario, either clearly present in the blog or plausible, justifiable inferences? Minor, less important details can be inferred or slightly altered as long as the main narrative remains faithful.  
- Score 10: No invented or speculative information regarding the main story and core scenario; all major content in scenario is clearly justified by the blog. Minor/inconsequential details are either present or represent reasonable inferences.  
- Score 8: One or two small, highly likely inferences or omittable details stray from the blog, but the main story and scenario are entirely grounded and unchanged.  
- Score 5: Several details not directly supported by the blog, including possibly a slight modification to the main story—but it still represents the same central scenario and intent. Minor inferences are acceptable, but no major invention of the overall event.  
- Score 1: Any significant change or invention in the main story, scenario, or central event compared to the blog, OR many unsupported facts even if the main idea is retained; demonstrates clear divergence from the blog in both spirit and detail.  
   
2. Fidelity (Data Sample Self-Consistency & Quality)    
(Consider the data sample itself, regardless of the blog.)  
- Is the scenario vivid, concrete, detailed, interesting, and coherent—not generic or trivial?  
- Are the character roles and scenario logically connected and non-redundant?  
- Is the scenario thoughtfully detailed and specific?  
- Score 10: All fields are natural, complete, and richly detailed; scenario is engaging; character roles are clear and relevant.  
- Score 8: Sample is mostly solid and engaging, but may lack depth or vividness in some places.  
- Score 5: Sample is complete and understandable, but is somewhat generic, routine, or underdeveloped.  
- Score 1: Content is generic, duplicative, trivial, unnatural, or weak.  
   
3. Novelty & Interest    
(Consider the data sample itself, regardless of the blog.)  
How much does the sample feel interesting, distinctive, and socially or emotionally resonant, rather than formulaic or generic?  
- Score 10: Highly distinctive, memorable, and evocative; clear emotional or cognitive resonance.  
- Score 8: Largely distinctive and interesting, but contains minor generic segments.  
- Score 5: Moderately engaging, but features significant generic or routine stretches.  
- Score 1: Largely generic, formulaic, or uninteresting.  
   
4. Naming    
Do all extracted human character names relevant to the scenario (excluding celebrities, stars, or movie/TV characters) use name placeholders such as "PersonA," "PersonB," etc., instead of real names from the blog? If real or directly reused names are present for these roles, rate this metric lower. Popular public figures may be referred to by their actual names.  
- Score 10: All applicable human characters (excluding celebrities, etc.) consistently use placeholder names (PersonA, etc).  
- Score 8: One minor inconsistent name usage or slip, but the majority use placeholders properly.  
- Score 5: Several character mentions fail to use placeholders, but some are implemented correctly.  
- Score 1: Most or all main human character names from the blog are left unchanged or not replaced with placeholders.  
   
5. Overall Quality    
Your holistic judgment of the data sample’s overall quality, based on accuracy, informativeness, distinctiveness, naming, and naturalness, scored as an integer from 1 to 10.  
   
---  
   
Output Format:    
Please provide your ratings using the following template:  
```xml  
<data>  
  <explanation>[Short summary of your analysis and reasoning]</explanation>  
  <hallucination>[INTEGER 0-10]</hallucination>  
  <fidelity>[INTEGER 0-10]</fidelity>  
  <novelty>[INTEGER 0-10]</novelty>  
  <naming>[INTEGER 0-10]</naming>  
  <overall>[INTEGER 0-10]</overall>  
</data>  
```  
   
---  
   
Instructions:  
Rate each data sample thoughtfully and fairly, paying attention to nuances in both blog and sample. Be strict—reserve high scores for truly strong samples, and use the full scoring scale as appropriate. Judge hallucination strictly according to the original blog; judge fidelity, novelty, and naming based only on the data sample. Only output the XML format result, do not output other explanatory words.  
   
If the data content is empty (e.g., {}), assign a score of 0 for all categories and state this in the explanation.  
   
---    
Now the task begins.

"""