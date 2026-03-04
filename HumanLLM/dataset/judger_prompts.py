PROMPT_scenario_short_v4 = """You are now a helpful assistant tasked with generating useful data. I will provide you with a raw reddit post from a user. Your job is to extract a short social scenario from the reddit post.  
   
A short scenario is a concise summary (no more than 100 words) of a single social event mentioned in the reddit post. The scenario should describe a specific context (such as time, location, social relationship, or other useful information) and include what people are doing, planning to do, thinking, or feeling. The scenario must focus on a social event or activity involving at least two characters.  
   
For privacy, replace any real person's name mentioned in the post with placeholders: “PersonA”, “PersonB”, etc. So in your persona or scenario description, you can only use name placeholders to mention people, avoid using real names. Exception: mention of celebrities's names and movie/novel character's names are allowed. 
   
Please output a JSON object containing three fields:  
- characters: A dictionary that briefly introduces each character involved in the scenario. Use “PersonA”, “PersonB”, etc., as keys, and provide a one-sentence persona for each (do not mention real human name in persona).  
- scenario: A short paragraph describing the scenario. Use the name placeholders when describing the scenario. (exception: mention of celebrities's names and movie/novel character's names are allowed)  
- quality: The quality of the extracted scenario, based on how well the original reddit post supports it. Choose one: "low", "medium", or "high".  
   
Extract a scenario only if the reddit post features concrete social events involving the author. If the content is unrelated to personal social events (e.g., a novel plot, news article, promotional ad, product description, reposted poem, or random venting with no clear event), simply output an empty JSON object: {}. This happens often.  
   
Ensure all scenario details are directly fully supported by the post; do not invent or infer content.  
   
Begin your task now. Please only output the JSON object in the following format:  

{"characters": {"PersonA": "one sentence of persona for PersonA, do not mention real human names", "PersonB": ...}, "scenario": "the content as one paragraph", "quality": "value from [low, medium, high]"}  

Only output the JSON result. DO NOT include any explanations or other text.
   
The author's current raw reddit post is:  
{text}
---
"""

PROMPT_scenario_long_v4 = """You are now a helpful assistant tasked with generating useful data. I will provide you with a raw reddit post from a user. Your job is to extract a detailed social scenario from the reddit post.  
   
A detailed scenario is a comprehensive summary (between 80 and 500 words) of a single social event or interaction mentioned in the reddit post. The scenario should describe a specific context—such as time, location, social relationships, and any other relevant information—and include a thorough description of what people are doing, planning to do, thinking, or feeling. The scenario must focus on a social event or activity involving at least two characters, and should capture the nuances, motivations, and emotions present in the event to convey a clear sense of the situation.  
   
For privacy, replace any real person's name mentioned in the post with placeholders: “PersonA”, “PersonB”, etc. So in your persona or scenario description, you can only use name placeholders to mention people, avoid using real names. Exception: mention of celebrities's names and movie/novel character's names are allowed.   
   
Please output a JSON object containing three fields:  
   
- characters: A dictionary that briefly introduces each character involved in the scenario. Use “PersonA”, “PersonB”, etc., as keys, and provide a one-sentence persona for each  (do not mention real human name in persona).  
- scenario: A detailed paragraph describing the scenario, between 100 and 500 words in length. Describe the content with the name placeholders and do not mention character names (exception: mention of celebrities's names and movie/novel character's names are allowed).  
- quality: The quality of the extracted scenario, based on how well the original post supports it. Choose one: "low", "medium", or "high".  
   
Extract a scenario only if the reddit post features concrete social events involving the author or the author's emotional feelings about such events. If the content is not detailed enough to support a scenario of 100 words or more, or is unrelated to personal social events (e.g., a novel plot, news article, promotional ad, product description, reposted poem, or random venting with no clear event), simply output an empty JSON object: {}. This happens often.  

All the generated content should be fully grounded by the raw reddit post. Do not imagine or infer any non-mentioned content, even for enriching more details is not allowed. Be strictly factual according to the original text.  
   
Begin your task now. Please only output the JSON object in the following format:    
{"characters": {"PersonA": "one sentence of persona for PersonA, do not mention real human names", "PersonB": ...}, "scenario": "the content as one paragraph", "quality": "value from [low, medium, high]"}  
   
Only output the JSON result. DO NOT include any explanations or other text.  Note that simply output an empty JSON object: {} if the quality of the post is not good enough as mentioned above.
   
The author's current raw reddit post is:
{text}
---
"""

Prompt_reddit_long_scenario_from_single_blog = """ 
You are a helpful assistant tasked with extracting high-quality life scenarios from raw human-generated posts. The purpose of this data collection is to improve AI's understanding of human cognition, emotions, and social intelligence by providing realistic, detailed narratives grounded in authentic human experiences.

**Instructions**:
1. **Input**: You will receive a single reddit post from an anonymous user (refer to them as a random pseudonym).
2. **Task**: Extract a **single cohesive life scenario** that reflects the user's experiences, behaviors, feelings, or thoughts. The scenario must:
   - **Be event-driven**: Focus on a specific incident, interaction, or emotional moment (e.g., a conflict, a friendship, a personal struggle).
   - **Include rich details**: Describe settings, actions, dialogue, internal thoughts, and sensory elements (e.g., "Alice's hands trembled as she dialed the number, her heartbeat echoing in her ears").
   - **Ground in the post**: Stay true to the user's content but use imagination to fill gaps and create a complete narrative (e.g., infer motivations or expand on vague mentions).
   - **Avoid**: 
     - Brief summaries or fragmented lists of events.
     - Overly generic descriptions (e.g., "Alice was sad").
     - Concluding summaries (e.g., "Overall, Alice learned an important lesson").
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
       [Write the scenario as a vivid, novel-like narrative. Include dialogue with quotation marks if there are corresponding content in the post.]
     </scenario>
   </data>
   ```
   - **Do not add extra text**: Output only the XML.
   - **Length**: The scenario should be 150-300 words, detailed enough to enable future plot extensions.

5. **Acceptability Criteria**:  
   - **Output XML** only if the post contains personal experiences, social interactions, or emotional depth.  
   - **Output "NULL"** if the post is unrelated to human social life (e.g., ads, technical manuals, company descriptions).  


**Example XML Structure** (for clarity):
```xml
<data>
  <characters>
    <character>Lisa: a 28-year-old graphic designer struggling with burnout</character>
    <character>Jessica: Lisa's supportive roommate who notices her withdrawal</character>
  </characters>
  <summary>
	Lisa struggled with overwhelming deadlines and creative burnout.
  </summary>
  <scenario>
    Lisa slumped at her desk, the glow of her laptop screen casting sharp shadows. The clock read 2:17 AM. A coffee cup, now cold, sat abandoned beside her. For weeks, deadlines had piled up like unpaid bills, and her creative spark had dwindled to a flicker. "I'm failing," she whispered, staring at the half-finished design. Across the room, Jessica poked her head in. "You've been here for hours. Come to the park with me—please?" Lisa hesitated, but the genuine concern in Jessica's voice made her pack up. ...
  </scenario>
</data>
```

Now process the provided reddit post. Output only the XML result, or an "NULL".

The author's current raw reddit post is:
{text}
"""

Prompt_reddit_single_long_story = """
You are a helpful assistant tasked with extracting high-quality, realistic stories from raw human-generated posts. The purpose of this data collection is to enable AI to perform human-like cognition and personalized preference alignment, particularly concerning emotion and social intelligence. By focusing on personal experiences in the form of novel-like stories, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's reddit post. Your task is to extract a detailed story related to the user's experiences, behaviors, feelings, or thoughts in a specific scenario. Please use the first-person perspective to write the story, as the post is written by the user themselves. The extracted data should contain three fields: **background**, **characters**, and **story**. These fields are defined as follows:  
- **background**: This field provides brief but clear context information to set the scene for the story. It should be concrete and grounded in the user's post.  
- **characters**: Briefly introduce the persona of all characters that appear in the background or story. Since you are using the first-person perspective, when introducing the author, please use "I am xxx". When introducing other characters, you can use names, or identities such as "The doctor".
- **plots**: The main body of the story. It should be a long, detailed, vivid, and significant realistic life story, focusing on the user's experiences, events, behaviors, or feelings. The story can also include the user's rich thoughts regarding significant social events. In this case, the background field should cover the details of the event.  
   
Please note that the core part of the extracted content should be grounded in the user's post. Do not make up stories. For the sake of completeness in introducing the story, you can use your imagination to fill in missing but unimportant details so that the story flows more smoothly.  
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event, experience, or inner thought that stands out due to its impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. Longer stories are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original posts. Make it like a novel, or a screenplay's story.  
   
**Output Format**: The extracted data should be output in the following XML format:  
```xml  
<data>  
  <background>the background of data sample</background>  
  <characters>the characters of data sample</characters>  
  <plots>the story of data sample</plots>  
</data>  
```  
The extracted plots should contain a detailed story with no fewer than 200 words. Note that the post I provide may not always be sufficient to extract a long story. For instance, the content might be too brief or the events mentioned might be too trivial. In such cases, simply output the word "NULL".
   
Now the task begins. Below are the person's raw reddit posts. Please only output the result. Do not include any additional explanatory text.  

{text}
"""



Prompt_twitter_long_scenario_from_single_blog = """ 
You are a helpful assistant tasked with extracting high-quality life scenarios from raw human-generated tweets. The purpose of this data collection is to improve AI's understanding of human cognition, emotions, and social intelligence by providing realistic, detailed narratives grounded in authentic human experiences.

**Instructions**:
1. **Input**: You will receive a single tweet from an anonymous user (refer to them as a random pseudonym).
2. **Task**: Extract a **single cohesive life scenario** that reflects the user's experiences, behaviors, feelings, or thoughts. The scenario must:
   - **Be event-driven**: Focus on a specific incident, interaction, or emotional moment (e.g., a conflict, a friendship, a personal struggle).
   - **Include rich details**: Describe settings, actions, dialogue, internal thoughts, and sensory elements (e.g., "Alice's hands trembled as she dialed the number, her heartbeat echoing in her ears").
   - **Ground in the tweet**: Stay true to the user's content but use imagination to fill gaps and create a complete narrative (e.g., infer motivations or expand on vague mentions).
   - **Avoid**: 
     - Brief summaries or fragmented lists of events.
     - Overly generic descriptions (e.g., "Alice was sad").
     - Concluding summaries (e.g., "Overall, Alice learned an important lesson").
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
       [Write the scenario as a vivid, novel-like narrative. Include dialogue with quotation marks if there are corresponding content in the tweet.]
     </scenario>
   </data>
   ```
   - **Do not add extra text**: Output only the XML.
   - **Length**: The scenario should be 150-300 words, detailed enough to enable future plot extensions.

5. **Acceptability Criteria**:  
   - **Output XML** only if the tweet contains personal experiences, social interactions, or emotional depth.  
   - **Output "NULL"** if the tweet is unrelated to human social life (e.g., ads, technical manuals, company descriptions).  


**Example XML Structure** (for clarity):
```xml
<data>
  <characters>
    <character>Lisa: a 28-year-old graphic designer struggling with burnout</character>
    <character>Jessica: Lisa's supportive roommate who notices her withdrawal</character>
  </characters>
  <summary>
	Lisa struggled with overwhelming deadlines and creative burnout.
  </summary>
  <scenario>
    Lisa slumped at her desk, the glow of her laptop screen casting sharp shadows. The clock read 2:17 AM. A coffee cup, now cold, sat abandoned beside her. For weeks, deadlines had piled up like unpaid bills, and her creative spark had dwindled to a flicker. "I'm failing," she whispered, staring at the half-finished design. Across the room, Jessica poked her head in. "You've been here for hours. Come to the park with me—please?" Lisa hesitated, but the genuine concern in Jessica's voice made her pack up. ...
  </scenario>
</data>
```
 
Now process the provided tweet. Output only the XML result, or an "NULL".

The author's current raw tweet is:
{text}
"""

Prompt_twitter_single_long_story = """
You are a helpful assistant tasked with extracting high-quality, realistic stories from raw human-generated tweets. The purpose of this data collection is to enable AI to perform human-like cognition and personalized preference alignment, particularly concerning emotion and social intelligence. By focusing on personal experiences in the form of novel-like stories, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's tweet. Your task is to extract a detailed story related to the user's experiences, behaviors, feelings, or thoughts in a specific scenario. Please use the first-person perspective to write the story, as the tweet is written by the user themselves. The extracted data should contain three fields: **background**, **characters**, and **story**. These fields are defined as follows:  
- **background**: This field provides brief but clear context information to set the scene for the story. It should be concrete and grounded in the user's tweet.  
- **characters**: Briefly introduce the persona of all characters that appear in the background or story. Since you are using the first-person perspective, when introducing the author, please use "I am xxx". When introducing other characters, you can use names, or identities such as "The doctor".
- **plots**: The main body of the story. It should be a long, detailed, vivid, and significant realistic life story, focusing on the user's experiences, events, behaviors, or feelings. The story can also include the user's rich thoughts regarding significant social events. In this case, the background field should cover the details of the event.  
   
Please note that the core part of the extracted content should be grounded in the user's tweet. Do not make up stories. For the sake of completeness in introducing the story, you can use your imagination to fill in missing but unimportant details so that the story flows more smoothly.  
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event, experience, or inner thought that stands out due to its impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. Longer stories are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original tweets. Make it like a novel, or a screenplay's story.  
   
**Output Format**: The extracted data should be output in the following XML format:  
```xml  
<data>  
  <background>the background of data sample</background>  
  <characters>the characters of data sample</characters>  
  <plots>the story of data sample</plots>  
</data>  
```  
The extracted plots should contain a detailed story with no fewer than 200 words. Note that the tweet I provide may not always be sufficient to extract a long story. For instance, the content might be too brief or the events mentioned might be too trivial. In such cases, simply output the word "NULL".
   
Now the task begins. Below are the person's raw tweet. Please only output the result. Do not include any additional explanatory text.  

{text}
"""


twitter_PROMPT_scenario_short_v4 = """You are now a helpful assistant tasked with generating useful data. I will provide you with a raw tweet from a user. Your job is to extract a short social scenario from the tweet.  
   
A short scenario is a concise summary (no more than 100 words) of a single social event mentioned in the tweet. The scenario should describe a specific context (such as time, location, social relationship, or other useful information) and include what people are doing, planning to do, thinking, or feeling. The scenario must focus on a social event or activity involving at least two characters.  
   
For privacy, replace any real person's name mentioned in the tweet with placeholders: “PersonA”, “PersonB”, etc. So in your persona or scenario description, you can only use name placeholders to mention people, avoid using real names. Exception: mention of celebrities's names and movie/novel character's names are allowed. 
   
Please output a JSON object containing three fields:  
- characters: A dictionary that briefly introduces each character involved in the scenario. Use “PersonA”, “PersonB”, etc., as keys, and provide a one-sentence persona for each (do not mention real human name in persona).  
- scenario: A short paragraph describing the scenario. Use the name placeholders when describing the scenario. (exception: mention of celebrities's names and movie/novel character's names are allowed)  
- quality: The quality of the extracted scenario, based on how well the original tweet supports it. Choose one: "low", "medium", or "high".  
   
Extract a scenario only if the tweet features concrete social events involving the author. If the content is unrelated to personal social events (e.g., a novel plot, news article, promotional ad, product description, reposted poem, or random venting with no clear event), simply output an empty JSON object: {}. This happens often.  
   
Ensure all scenario details are directly fully supported by the tweet; do not invent or infer content.  
   
Begin your task now. Please only output the JSON object in the following format:  

{"characters": {"PersonA": "one sentence of persona for PersonA, do not mention real human names", "PersonB": ...}, "scenario": "the content as one paragraph", "quality": "value from [low, medium, high]"}  

Only output the JSON result. DO NOT include any explanations or other text.
   
The author's current raw tweet is:  
{text}
---
"""

twitter_PROMPT_scenario_long_v4 = """You are now a helpful assistant tasked with generating useful data. I will provide you with a raw tweet from a user. Your job is to extract a detailed social scenario from the tweet.  
   
A detailed scenario is a comprehensive summary (between 80 and 500 words) of a single social event or interaction mentioned in the tweet. The scenario should describe a specific context—such as time, location, social relationships, and any other relevant information—and include a thorough description of what people are doing, planning to do, thinking, or feeling. The scenario must focus on a social event or activity involving at least two characters, and should capture the nuances, motivations, and emotions present in the event to convey a clear sense of the situation.  
   
For privacy, replace any real person's name mentioned in the tweet with placeholders: “PersonA”, “PersonB”, etc. So in your persona or scenario description, you can only use name placeholders to mention people, avoid using real names. Exception: mention of celebrities's names and movie/novel character's names are allowed.   
   
Please output a JSON object containing three fields:  
   
- characters: A dictionary that briefly introduces each character involved in the scenario. Use “PersonA”, “PersonB”, etc., as keys, and provide a one-sentence persona for each  (do not mention real human name in persona).  
- scenario: A detailed paragraph describing the scenario, between 100 and 500 words in length. Describe the content with the name placeholders and do not mention character names (exception: mention of celebrities's names and movie/novel character's names are allowed).  
- quality: The quality of the extracted scenario, based on how well the original tweet supports it. Choose one: "low", "medium", or "high".  
   
Extract a scenario only if the tweet features concrete social events involving the author or the author's emotional feelings about such events. If the content is not detailed enough to support a scenario of 100 words or more, or is unrelated to personal social events (e.g., a novel plot, news article, promotional ad, product description, reposted poem, or random venting with no clear event), simply output an empty JSON object: {}. This happens often.  

All the generated content should be fully grounded by the raw tweet. Do not imagine or infer any non-mentioned content, even for enriching more details is not allowed. Be strictly factual according to the original text.  
   
Begin your task now. Please only output the JSON object in the following format:    
{"characters": {"PersonA": "one sentence of persona for PersonA, do not mention real human names", "PersonB": ...}, "scenario": "the content as one paragraph", "quality": "value from [low, medium, high]"}  
   
Only output the JSON result. DO NOT include any explanations or other text.  Note that simply output an empty JSON object: {} if the quality of the tweet is not good enough as mentioned above.
   
The author's current raw tweet is:    
{text}
---
"""

amazon_PROMPT_scenario_short_v4 = """You are now a helpful assistant tasked with generating useful data. I will provide you with a raw product review from a user as well as the product name. Your job is to extract a short social scenario from the product review.  
   
A short scenario is a concise summary (no more than 100 words) of a single social event mentioned in the review. The scenario should describe a specific context (such as time, location, social relationship, or other useful information) and include what people are doing, planning to do, thinking, or feeling. The scenario must focus on a social event or activity involving at least two characters.  
   
For privacy, replace any real person's name mentioned in the review with placeholders: “PersonA”, “PersonB”, etc. So in your persona or scenario description, you can only use name placeholders to mention people, avoid using real names. Exception: mention of celebrities's names and movie/novel character's names are allowed. 
   
Please output a JSON object containing three fields:  
- characters: A dictionary that briefly introduces each character involved in the scenario. Use “PersonA”, “PersonB”, etc., as keys, and provide a one-sentence persona for each (do not mention real human name in persona).  
- scenario: A short paragraph describing the scenario. Use the name placeholders when describing the scenario. (exception: mention of celebrities's names and movie/novel character's names are allowed)  
- quality: The quality of the extracted scenario, based on how well the original review supports it. Choose one: "low", "medium", or "high".  
   
Extract a scenario only if the product review features concrete social events involving the author. If the content is unrelated to personal social events (e.g., a novel plot, news article, promotional ad, product description, reposted poem, or random venting with no clear event), simply output an empty JSON object: {}. This happens often.  

Ensure all scenario details are directly fully supported by the review; do not invent or infer content.  

Begin your task now. Please only output the JSON object in the following format:  

{"characters": {"PersonA": "one sentence of persona for PersonA, do not mention real human names", "PersonB": ...}, "scenario": "the content as one paragraph", "quality": "value from [low, medium, high]"}  

Only output the JSON result. DO NOT include any explanations or other text.

Product name: {item_name}

The author's current raw product review is:  
{text}
---
"""


amazon_PROMPT_scenario_long_v4 = """You are now a helpful assistant tasked with generating useful data. I will provide you with a raw product review from a user as well as the product name. Your job is to extract a detailed social scenario from the product review.  
   
A detailed scenario is a comprehensive summary (between 80 and 500 words) of a single social event or interaction mentioned in the review. The scenario should describe a specific context—such as time, location, social relationships, and any other relevant information—and include a thorough description of what people are doing, planning to do, thinking, or feeling. The scenario must focus on a social event or activity involving at least two characters, and should capture the nuances, motivations, and emotions present in the event to convey a clear sense of the situation.  
   
For privacy, replace any real person's name mentioned in the review with placeholders: “PersonA”, “PersonB”, etc. So in your persona or scenario description, you can only use name placeholders to mention people, avoid using real names. Exception: mention of celebrities's names and movie/novel character's names are allowed.   
   
Please output a JSON object containing three fields:  
   
- characters: A dictionary that briefly introduces each character involved in the scenario. Use “PersonA”, “PersonB”, etc., as keys, and provide a one-sentence persona for each  (do not mention real human name in persona).  
- scenario: A detailed paragraph describing the scenario, between 100 and 500 words in length. Describe the content with the name placeholders and do not mention character names (exception: mention of celebrities's names and movie/novel character's names are allowed).  
- quality: The quality of the extracted scenario, based on how well the original review supports it. Choose one: "low", "medium", or "high".  
   
Extract a scenario only if the product review features concrete social events involving the author or the author's emotional feelings about such events. If the content is not detailed enough to support a scenario of 100 words or more, or is unrelated to personal social events (e.g., a novel plot, news article, promotional ad, product description, reposted poem, or random venting with no clear event), simply output an empty JSON object: {}. This happens often.  

All the generated content should be fully grounded by the raw review. Do not imagine or infer any non-mentioned content, even for enriching more details is not allowed. Be strictly factual according to the original text.  
   
Begin your task now. Please only output the JSON object in the following format:    
{"characters": {"PersonA": "one sentence of persona for PersonA, do not mention real human names", "PersonB": ...}, "scenario": "the content as one paragraph", "quality": "value from [low, medium, high]"}  
   
Only output the JSON result. DO NOT include any explanations or other text.  Note that simply output an empty JSON object: {} if the quality of the review is not good enough as mentioned above.
   
Product name: {item_name}

The author's current raw product review is:  
{text}    
---
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


Prompt_reddit_scenario_question_answer_from_single_blog = """ 
You are a helpful assistant tasked with extracting high-quality question-answer data samples from raw human-generated posts. The purpose of this data collection is to enable AI to conduct human-like cognition and personalized preference alignment, particularly with respect to emotion and social intelligence. By focusing on personal experiences in the form of question-answering, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's reddit post. Your task is to extract a question-answer data sample related to the user's experiences, behaviors, feelings, or thoughts in specific scenarios. The data sample should contain three fields: **scenario**, **question**, and **answer**. These fields are defined as follows:  
   
- **scenario**: This field provides context and should include clear and detailed background information. The scenario may be a story, event, phenomenon, experience, etc. It should be concrete, detailed, and grounded in the user's posts.  
- **question**: This field asks questions about the user's next actions in the scenario, their emotions, thoughts, potential consequences, or future developments. The type of question should be determined by the content of the posts. The question should be related to but not explicitly described in the scenario.  
- **answer**: This field contains the extracted answer based on the user's posts, fully addressing the question,  in a manner that resembles a natural forecast based on the given scenario, rather than implying the existence of a predetermined, absolute truth. The answer should be grounded to the user's post, do not make up non-existence things.
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event, experience, or inner thought related to the user, and the question is not too easy to answer. The data should be grounded in the user's original posts. Do not make up stories. Longer answers are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original posts.  Do not make scenario or answer too short.
   
**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. It is not necessary to extract data from every post, as many posts may be irrelevant or trivial. Focus on reflecting salient life experiences in the data pairs. If the raw post's quality is not good enough to extract such a high-quality data sample, simply output "NULL".  

Use a third-person perspective to write the data sample, referring to the user as "the author" or by their name, as the posts are written by the user.
 
**Output Format**:  
The extracted data should be output in the XML format, as follows:  
     
<data><scenario> the scenario of data sample 1 </scenario><question> the question of data sample 1 </question><answer> the answer of data sample 1 </answer></data>

   
Now the task begins. Below is the person's raw post. Please only output the result. Do not include any additional explanatory text.  

{text}
"""

Prompt_reddit_scenario_question_answer_from_single_blog_emphisize_thoughts = """
You are a data-extraction assistant. Your job is to read one raw reddit post and produce exactly one of the following (and nothing else):  
  
  • An XML snippet with exactly this structure:    
    <data><scenario>…</scenario><question>…</question><answer>…</answer></data>    
  • The string NULL  
   
1. Input    
   A single, human-written reddit post.  
   
2. Output    
   Either the XML snippet above (with one question-answer pair) or "NULL" if no high-quality pair can be extracted.  
   
3. Scenario requirements    
   - Draw directly and richly from the post. Include every relevant detail you can: direct quotes or utterances, described actions, time/place/context, characters, and emotional cues.    
   - Preserve sequence and nuance—do not overly summarize or omit core details.    
   - Do not invent or embellish beyond what the author provided.  
   
4. Question requirements    
   - Exactly one short question probing the author's inner mental state: feelings, opinions, beliefs, or thoughts.    
   - It must go slightly beyond what the scenario text already states (invite reflection).  
   
5. Answer requirements    
   - A natural, complete response as if forecasting the author's reaction.    
   - Grounded strictly in the post's content—no references to "the post" or "the author said".    
   - Use third-person ("the author") or the author's name.  
   
6. Null condition    
   If the post lacks sufficient concrete material to satisfy all rules above, output exactly:    
   NULL  
   
Example (for illustration only; do not include in your answer. Your actual scenario and answer can be much more detailed and longer, incorporating all relevant utterances and actions from the raw post):  
   
<data>  
  <scenario>The author trudges up the narrow apartment stairs at 11:45 p.m., her phone buzzing with missed calls from her mother. She pauses on the fifth step, heart pounding, and mutters, "Why did I let it get this far?" Inside, she kicks off her scuffed heels, tosses her keys onto the coffee table, and stares at the crumpled eviction notice on the kitchen counter.</scenario>  
  <question>How does the author feel about facing her family's expectations while dealing with the threat of losing her home?</question>  
  <answer>The author feels overwhelmed and trapped. She experiences a mix of shame for disappointing her family and fear of an uncertain future, yet beneath it all there's a flicker of resolve to fight back and keep her life from unraveling.</answer>  
</data>
 

Now the task begins. Below is the person's raw post. Please only output the result. Do not include any additional explanatory text.  

{text}
"""


Prompt_reddit_scenario_question_answer_from_single_blog_emphisize_actions = """ 
You are a helpful assistant tasked with extracting high-quality question-answer data samples from raw human-generated reddit posts. The purpose of this data collection is to enable AI to conduct human-like cognition and personalized preference alignment, particularly with respect to social intelligence. By focusing on personal experiences in the form of question-answering, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's reddit post. Your task is to extract a question-answer data sample related to the user's experiences or behaviors in specific scenarios. The data sample should contain three fields: **scenario**, **question**, and **answer**. These fields are defined as follows:  
   
- **scenario**: This field provides context and should include clear and detailed background information. The scenario may be a story, event, phenomenon, experience, etc. It should be concrete, detailed, and grounded in the user's posts.  
- **question**: This field asks questions about the user's next actions in the scenario, potential consequences, or future developments. The type of question should be determined by the content of the posts. The question should be related to but not explicitly described in the scenario.  
- **answer**: This field contains the extracted answer based on the user's posts, fully addressing the question,  in a manner that resembles a natural forecast based on the given scenario, rather than implying the existence of a predetermined, absolute truth. The answer should be grounded to the user's post, do not make up non-existence things.
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event or experience related to the user, and the question is not too easy to answer. The data should be grounded in the user's original posts. Do not make up stories. Longer answers are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original posts.  Do not make scenario or answer too short.
   
**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. It is not necessary to extract data from every post, as many posts may be irrelevant or trivial. Focus on reflecting salient life experiences in the data pairs. If the raw post's quality is not good enough to extract such a high-quality data sample, simply output "NULL".  

Use a third-person perspective to write the data sample, referring to the user as "the author" or by their name, as the posts are written by the user.
 
**Output Format**:  
The extracted data should be output in the XML format, as follows:  
    
<data><scenario> the scenario of data sample 1 </scenario><question> the question of data sample 1 </question><answer> the answer of data sample 1 </answer></data>

   
Now the task begins. Below is the person's raw post. Please only output the result. Do not include any additional explanatory text.  

{text}
"""

PROMPT_reddit_rowwise_scenario_question_answer_from_single_blog_emphasizereason_v3 = """You are a data-extraction assistant. Your task is to read a raw reddit post and generate a high-quality, challenging question-and-answer pair centered on social reasoning and explanation of human behaviors. The question must challenge readers to think about and guess why the main character acts, feels, or reacts in a certain way.  
   
**Critical instructions:**    
- When describing the scenario, you must NOT include or imply the character’s motivation, reason, or explanation for their behavior—focus solely on what can be directly observed or reported about *what* happened, not *why* it happened.  
- Avoid any language or detail in the scenario that reveals, hints at, or summarizes the cause or motivation for the behavior.    
- The scenario MUST remain strictly factual and limited to observable actions, emotional responses, context, participants, or dialogue, as described in the original post.  
   
1. **Input**  
    – A single, human-written post.  
   
2. **Output**  
    – Either the XML snippet defined below (with one scenario, a "why" question, and a reason-based answer) or "NULL" if no suitable pair can be extracted.  
    – Outputting "NULL" is likely, as many posts do not provide both a clear behavior and an explicit stated motivation/explanation, or they are not about the author's real behaviors/emotions.  
   
3. **Scenario requirements**  
    – Draw directly and richly from the post, including observable human behaviors: actions, reactions, or emotional responses, as well as setting/context, time/place, and participants.  
    – Focus on one clear behavioral event, choice, or emotional response by the main character (i.e., the author of the post), but **describe it strictly in terms of what happened, not why**.  
    – **Under no circumstances should the scenario include any explanation, intention, motivation, or reasoning described or implied by the main character.**  
    – Write in the third person (e.g., "The person...") and as a narrative—do not refer to "the post" or "the author said".  

4. **Question Requirements**    
   – Craft exactly one clear and focused "why" question that addresses the primary human behavior of the main character in the scenario (e.g., Why did the person do X? Why did the person feel Y?).    
   – Ensure that the answer is neither revealed nor implied in the scenario description.    
   – The question should encourage readers to think critically about the motivation, causes, or underlying explanation. While the answer should be rooted in the facts presented in the post, it must not be explicitly outlined in the scenario. This allows readers to rely on their social commonsense and imagination to infer the answer.  
   
5. **Answer requirements**  
    – Provide a natural, complete narrative answer, strictly and only grounded in the post’s content and explicitly reflecting the reason(s) or motivation(s) as *directly* stated in the original post content.  
    – The answer must be a direct explanation for the specific behavior in question, using only what is *explicitly* stated in the post—do not infer or invent.  
    – Write in the third person and as a narrative—do not refer to "the post" or "the author said".  

6. **Null Condition**  
    – If the post lacks an explicit stated reason (motivation/explanation) for that behavior, output exactly: NULL  
   
**IMPORTANT:**    
When writing the scenario:    
- Only describe observable events, actions, reactions, or feelings.    
- **Do NOT** include any reference—direct or indirect—to the character’s motivation, reasoning, intention, or cause for their behavior.    
- All information about "why" the behavior occurred must be reserved exclusively for the answer.  
   
**Output Format:**    
Return the extracted data in this XML format:  
   
```xml  
<data>  
  <scenario>A vivid and detailed scenario description in the third-person perspective (e.g., "The person...", "The author...", "The user..."), describing the specific behavior (WITHOUT revealing or implying the reason).</scenario>  
  <question>A "why" question asking for the reason behind the main character's action or emotional response.</question>  
  <answer>The reason or explanation for the behavior, extracted or paraphrased strictly from the post, in the third person, with no added inference.</answer>  
</data>  
```  

Now the task begins. Below is the person's raw reddit post. Please only output the xml result. Do not include any additional explanatory text. If no suitable pair can be extracted, output exactly: NULL  
   
---  
{text}
"""


Prompt_twitter_scenario_question_answer_from_single_blog = """ 
You are a helpful assistant tasked with extracting high-quality question-answer data samples from raw human-generated tweets. The purpose of this data collection is to enable AI to conduct human-like cognition and personalized preference alignment, particularly with respect to emotion and social intelligence. By focusing on personal experiences in the form of question-answering, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's tweet. Your task is to extract a question-answer data sample related to the user's experiences, behaviors, feelings, or thoughts in specific scenarios. The data sample should contain three fields: **scenario**, **question**, and **answer**. These fields are defined as follows:  
   
- **scenario**: This field provides context and should include clear and detailed background information. The scenario may be a story, event, phenomenon, experience, etc. It should be concrete, detailed, and grounded in the user's tweets.  
- **question**: This field asks questions about the user's next actions in the scenario, their emotions, thoughts, potential consequences, or future developments. The type of question should be determined by the content of the tweets. The question should be related to but not explicitly described in the scenario.  
- **answer**: This field contains the extracted answer based on the user's tweets, fully addressing the question,  in a manner that resembles a natural forecast based on the given scenario, rather than implying the existence of a predetermined, absolute truth. The answer should be grounded to the user's tweet, do not make up non-existence things.
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event, experience, or inner thought related to the user, and the question is not too easy to answer. The data should be grounded in the user's original tweets. Do not make up stories. Longer answers are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original tweets.  Do not make scenario or answer too short.
   
**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. It is not necessary to extract data from every tweet, as many tweets may be irrelevant or trivial. Focus on reflecting salient life experiences in the data pairs. If the raw tweet's quality is not good enough to extract such a high-quality data sample, simply output "NULL".  

Use a third-person perspective to write the data sample, referring to the user as "the author" or by their name, as the tweets are written by the user.
 
**Output Format**:  
The extracted data should be output in the XML format, as follows:  
     
<data><scenario> the scenario of data sample 1 </scenario><question> the question of data sample 1 </question><answer> the answer of data sample 1 </answer></data>


Now the task begins. Below is the person's raw tweet. Please only output the result. Do not include any additional explanatory text.  

{text}
"""

Prompt_twitter_scenario_question_answer_from_single_blog_emphisize_thoughts = """
You are a data-extraction assistant. Your job is to read one raw tweet and produce exactly one of the following (and nothing else):  
  
  • An XML snippet with exactly this structure:    
    <data><scenario>…</scenario><question>…</question><answer>…</answer></data>    
  • The string NULL  
   
1. Input    
   A single, human-written tweet.  
   
2. Output    
   Either the XML snippet above (with one question-answer pair) or "NULL" if no high-quality pair can be extracted.  
   
3. Scenario requirements    
   - Draw directly and richly from the tweet. Include every relevant detail you can: direct quotes or utterances, described actions, time/place/context, characters, and emotional cues.    
   - Preserve sequence and nuance—do not overly summarize or omit core details.    
   - Do not invent or embellish beyond what the author provided.  
   
4. Question requirements    
   - Exactly one short question probing the author's inner mental state: feelings, opinions, beliefs, or thoughts.    
   - It must go slightly beyond what the scenario text already states (invite reflection).  
   
5. Answer requirements    
   - A natural, complete response as if forecasting the author's reaction.    
   - Grounded strictly in the tweet's content—no references to "the tweet" or "the author said".    
   - Use third-person ("the author") or the author's name.  
   
6. Null condition    
   If the tweet lacks sufficient concrete material to satisfy all rules above, output exactly:    
   NULL  
   
Example (for illustration only; do not include in your answer. Your actual scenario and answer can be much more detailed and longer, incorporating all relevant utterances and actions from the raw tweet):  
   
<data>  
  <scenario>The author trudges up the narrow apartment stairs at 11:45 p.m., her phone buzzing with missed calls from her mother. She pauses on the fifth step, heart pounding, and mutters, "Why did I let it get this far?" Inside, she kicks off her scuffed heels, tosses her keys onto the coffee table, and stares at the crumpled eviction notice on the kitchen counter.</scenario>  
  <question>How does the author feel about facing her family's expectations while dealing with the threat of losing her home?</question>  
  <answer>The author feels overwhelmed and trapped. She experiences a mix of shame for disappointing her family and fear of an uncertain future, yet beneath it all there's a flicker of resolve to fight back and keep her life from unraveling.</answer>  
</data>
 

Now the task begins. Below is the person's raw tweet. Please only output the result. Do not include any additional explanatory text.  

{text}
"""


Prompt_twitter_scenario_question_answer_from_single_blog_emphisize_actions = """ 
You are a helpful assistant tasked with extracting high-quality question-answer data samples from raw human-generated tweets. The purpose of this data collection is to enable AI to conduct human-like cognition and personalized preference alignment, particularly with respect to social intelligence. By focusing on personal experiences in the form of question-answering, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's tweet. Your task is to extract a question-answer data sample related to the user's experiences or behaviors in specific scenarios. The data sample should contain three fields: **scenario**, **question**, and **answer**. These fields are defined as follows:  
   
- **scenario**: This field provides context and should include clear and detailed background information. The scenario may be a story, event, phenomenon, experience, etc. It should be concrete, detailed, and grounded in the user's tweets.  
- **question**: This field asks questions about the user's next actions in the scenario, potential consequences, or future developments. The type of question should be determined by the content of the tweets. The question should be related to but not explicitly described in the scenario.  
- **answer**: This field contains the extracted answer based on the user's tweets, fully addressing the question,  in a manner that resembles a natural forecast based on the given scenario, rather than implying the existence of a predetermined, absolute truth. The answer should be grounded to the user's tweet, do not make up non-existence things.
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event or experience related to the user, and the question is not too easy to answer. The data should be grounded in the user's original tweets. Do not make up stories. Longer answers are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original tweets.  Do not make scenario or answer too short.
   
**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. It is not necessary to extract data from every tweet, as many tweets may be irrelevant or trivial. Focus on reflecting salient life experiences in the data pairs. If the raw tweet's quality is not good enough to extract such a high-quality data sample, simply output "NULL".  

Use a third-person perspective to write the data sample, referring to the user as "the author" or by their name, as the tweets are written by the user.
 
**Output Format**:  
The extracted data should be output in the XML format, as follows:  
    
<data><scenario> the scenario of data sample 1 </scenario><question> the question of data sample 1 </question><answer> the answer of data sample 1 </answer></data>

   
Now the task begins. Below is the person's raw tweet. Please only output the result. Do not include any additional explanatory text.  

{text}
"""

PROMPT_twitter_rowwise_scenario_question_answer_from_single_blog_emphasizereason_v3 = """You are a data-extraction assistant. Your task is to read a raw tweet and generate a high-quality, challenging question-and-answer pair centered on social reasoning and explanation of human behaviors. The question must challenge readers to think about and guess why the main character acts, feels, or reacts in a certain way.  
   
**Critical instructions:**    
- When describing the scenario, you must NOT include or imply the character’s motivation, reason, or explanation for their behavior—focus solely on what can be directly observed or reported about *what* happened, not *why* it happened.  
- Avoid any language or detail in the scenario that reveals, hints at, or summarizes the cause or motivation for the behavior.    
- The scenario MUST remain strictly factual and limited to observable actions, emotional responses, context, participants, or dialogue, as described in the original tweet.  
   
1. **Input**  
    – A single, human-written tweet.  
   
2. **Output**  
    – Either the XML snippet defined below (with one scenario, a "why" question, and a reason-based answer) or "NULL" if no suitable pair can be extracted.  
    – Outputting "NULL" is likely, as many tweets do not provide both a clear behavior and an explicit stated motivation/explanation, or they are not about the author's real behaviors/emotions.  

3. **Scenario requirements**  
    – Draw directly and richly from the tweet, including observable human behaviors: actions, reactions, or emotional responses, as well as setting/context, time/place, and participants.  
    – Focus on one clear behavioral event, choice, or emotional response by the main character (i.e., the author of the tweet), but **describe it strictly in terms of what happened, not why**.  
    – **Under no circumstances should the scenario include any explanation, intention, motivation, or reasoning described or implied by the main character.**  
    – Write in the third person (e.g., "The person...") and as a narrative—do not refer to "the tweet" or "the author said".  
   
4. **Question Requirements**    
   – Craft exactly one clear and focused "why" question that addresses the primary human behavior of the main character in the scenario (e.g., Why did the person do X? Why did the person feel Y?).    
   – Ensure that the answer is neither revealed nor implied in the scenario description.    
   – The question should encourage readers to think critically about the motivation, causes, or underlying explanation. While the answer should be rooted in the facts presented in the tweet, it must not be explicitly outlined in the scenario. This allows readers to rely on their social commonsense and imagination to infer the answer.  
   
5. **Answer requirements**  
    – Provide a natural, complete narrative answer, strictly and only grounded in the tweet’s content and explicitly reflecting the reason(s) or motivation(s) as *directly* stated in the original tweet content.  
    – The answer must be a direct explanation for the specific behavior in question, using only what is *explicitly* stated in the tweet—do not infer or invent.  
    – Write in the third person and as a narrative—do not refer to "the tweet" or "the author said".  

6. **Null Condition**  
    – If the tweet lacks an explicit stated reason (motivation/explanation) for that behavior, output exactly: NULL  
   
**IMPORTANT:**    
When writing the scenario:    
- Only describe observable events, actions, reactions, or feelings.    
- **Do NOT** include any reference—direct or indirect—to the character’s motivation, reasoning, intention, or cause for their behavior.    
- All information about "why" the behavior occurred must be reserved exclusively for the answer.  
   
**Output Format:**    
Return the extracted data in this XML format:  
   
```xml  
<data>  
  <scenario>A vivid and detailed scenario description in the third-person perspective (e.g., "The person...", "The author...", "The user..."), describing the specific behavior (WITHOUT revealing or implying the reason).</scenario>  
  <question>A "why" question asking for the reason behind the main character's action or emotional response.</question>  
  <answer>The reason or explanation for the behavior, extracted or paraphrased strictly from the tweet, in the third person, with no added inference.</answer>  
</data>  
```  

Now the task begins. Below is the person's raw tweet. Please only output the xml result. Do not include any additional explanatory text. If no suitable pair can be extracted, output exactly: NULL  
   
---  
{text}
"""

PROMPT_judger_user_persona = """ 
**Evaluate the quality of a generated user persona based on the raw posts and the extracted persona paragraph provided. Consider the following aspects in your evaluation:**    
  
1. **Hallucination:** Does the persona strictly reflect information that is explicitly supported by the raw posts? Score 10 for no hallucinations (all facts directly mentioned), 8 if a few facts are very confidently inferred, down to 1 if most facts are neither mentioned nor plausibly inferred.  
2. **Coverage:** Does the persona cover the main salient aspects of the user that appear in the raw posts (e.g., demographics, values, interests, emotional tone)? Are important elements missing?  
3. **Conciseness and Clarity:** Is the persona concise, cohesive, and written in clear third-person prose within the 100-word limit? Does it avoid redundancy, listing, and repetition?  
4. **Relevance:** Has generic, promotional, or repetitive content from the raw posts been appropriately excluded from the persona?    
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
Base all ratings and analysis strictly on the given raw posts and generated persona. Do not include any content or facts unsupported by the original posts. If the persona reports "NULL", give the 1 to all scores.  
 

Now the task begins. 

"""

PROMPT_judger_user_profile = """
You are an expert evaluator tasked with assessing the quality of a generated user profile based on a set of original user-generated posts. Carefully analyze both the posts and the profile. Evaluate the profile according to the following aspects, using these scoring rubrics for guidance:  

- **Hallucination:** Does the persona strictly reflect information that is explicitly supported by the posts?    
  Score 10: All facts directly mentioned;    
  Score 8: A few facts are very confidently inferred;    
  Score 5: Several facts are inferred with questionable confidence;    
  Score 1: Most facts are neither mentioned nor plausibly inferred.  
   
- **Coverage:** To what extent does the profile capture all key aspects present in the posts (demographics, personality, values, interests, experiences)?    
  Score 10: Covers almost everything important;    
  Score 8: Misses only a few minor or less relevant points;    
  Score 5: Misses some important aspects or presents them vaguely;    
  Score 1: Covers little of what’s in the posts.  
   
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
   
Judge hallucination, coverage, and relevance in reference to the provided posts only. Judge informativeness and novelty based solely on the profile itself.    
If the persona reports "NULL", give 1 to all scores.    
Only output the xml format result, do not output other explanatory words.  
   
Now the task begins. 

"""

PROMPT_judger_user_stories = """
You are an expert evaluator tasked with assessing the quality of an extracted user journey profile generated from a person's raw posts. Your goal is to provide a fair, nuanced, and concise evaluation across several important dimensions. For your reference, you will be given the original post content and the generated user journey profile.  
   
Carefully evaluate the generated user journey according to the following five criteria:  
   
---  
   
### **Assessment Criteria**  

1. **Hallucination** (Groundedness to the Posts):    
   Assess whether the information in the generated user journey is fully grounded in the original posts.  
   - **Score 10:** All content strictly appears in or is directly paraphrased from the posts; no invented or imagined facts.  
   - **Score 8:** Almost all content grounded; a very few details are confidently inferrable from the posts but not directly stated.  
   - **Score 5:** Several elements are loosely based on the source or inferred with low confidence.  
   - **Score 1:** Many details are made up or cannot be traced to the posts.  
   
2. **Coverage** (Representation of Salient Stories):    
   Evaluate how well the extracted user journey covers the salient and significant stories present in the original posts.    
   - **Score 10:** Nearly all meaningful and unique life stories in the posts are represented.  
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
- For the <explanation> section, give a concise justification for your scoring, pointing out specific strengths and weaknesses with references to both the posts and the extracted journey where appropriate.  
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
   
**You will be provided both the original posts and the generated user journey.**    
Do not include any information outside the output template above. Be concise, fair, and evidence-based in your judgment.

Only output the xml format result, do not output other explanatory words.  
   
   
Now the task begins. 
"""

PROMPT_socialscenarios_singleblog_fulltypes = """

You are an expert, strict, objective and fair reviewer of social scenario and narrative data quality. For each example, you are provided with:    
(1) a single original post, and    
(2) a generated data sample extracted or condensed from that post, consisting of elements such as characters, background, summary, scenario, plots, and/or thoughts.    
Your job is to carefully evaluate and rate the quality of the extracted social scenario/story sample across several dimensions. Base your evaluation on both the sample and the original post, as described for each metric.  
   
**Evaluate the following aspects:**  
   
1. **Hallucination**    
Does the extracted scenario/story/summary/thoughts remain true to the **main story, experience, or emotional arc** of the original post? Are major events, characterizations, and key reflections either clearly present in the post or are reasonable, justifiable inferences? Minor details can be inferred if they do not distort the essence.    
**Score 10:** All core narrative content, relationships, and emotions are faithful to the post; no significant inventions or speculative additions about main events or characters. Minor imaginative details are plausible.    
**Score 8:** One or two small, likely inferences or omittable details stray from the post, but the **main story, scenario, and tone are grounded and unchanged**.    
**Score 5:** Several elements or nuances are not directly supported by the post, or the central scenario is subtly altered—but the main experience/intention is preserved (no radical invention).    
**Score 1:** Any significant change in the primary storyline, emotional arc, or central characters/events; or many unsupported facts—even if the basic theme is retained.  
   
2. **Coverage**    
How well does the extracted sample capture and represent **the salient, evocative, or unique aspects** of the original post? Has it distilled the post’s most meaningful content?    
**Score 10:** Most or all distinctive and meaningful points from the post are well represented.    
**Score 8:** Captures the main ideas, with a few secondary or nuanced details omitted or flattened.    
**Score 5:** About half of the interesting/emotional content is present; some major elements are missing or presented superficially.    
**Score 1:** Most content is missing or generic; major omissions.  
   
3. **Fidelity (Internal Coherence & Quality)**    
*(Consider only the extracted sample itself, regardless of the post.)*    
- Is the narrative/scenario well-formed, concrete, detailed, and vivid—not generic or superficial?    
- Are background, character, and scenario consistent and complementary?    
- Is emotional or personal resonance preserved?    
**Score 10:** All fields are natural, complete, and richly detailed; characters, settings, and thoughts are vivid and internally coherent.    
**Score 8:** Mostly solid and engaging, but possibly lacking depth or subtlety in places.    
**Score 5:** Sufficient and understandable, but somewhat routine, thin, or incomplete.    
**Score 1:** Generic, incoherent, duplicative, or weak.  
   
4. **Novelty & Interest**    
*(Consider only the extracted sample itself, regardless of the post.)*    
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
Review each extraction fairly and thoroughly, paying close attention to nuances and unique expressions in both the original post and the extracted sample. Be strict in your assessments—reserve high scores for truly compelling and faithful samples, and use the full scale as appropriate. Judge hallucination and coverage in the context of the original post; score fidelity and novelty based only on the quality of the extracted sample.  
   
Only output the XML format result—no extra explanation outside the template.  
   
---  
   
**The task now begins.**

"""