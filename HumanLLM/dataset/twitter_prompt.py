Prompt_twitter_rewrite_blog = """You are a text cleaning assistant.
Your task is to clean a Twitter blog post by removing noisy or irrelevant elements while preserving the original content's meaning, intent, and readability.

Apply the following cleaning rules:
1. Remove special characters and encoded text, including:
  - Unicode symbols like \u00a3, \u2026, \u0caa\u0ccd...
  - HTML entities like &amp;, &gt;, #x200B, etc.
  - Repetitive patterns like \n.\n.\n.\n.
2. Delete all URLs, whether raw (https://..., http://...) or markdown-style ([text](url)).
3. Remove all hashtags (e.g., #LifeGoals, #AI2024).
4. Remove all mentions (e.g., @elonmusk, @user123). If removing a mention breaks sentence structure, consider filling in an appropriate name based on context.
5. Strip out emojis, meme text, ASCII art, and decorative or symbolic content not essential to understanding the tweet.
6. Discard non-English phrases or characters, unless they are common loanwords used in English (e.g., “fútbol” in an English context can be removed unless it's core to the tweet's meaning).

Important:
- Do not rewrite, rephrase, or change the wording.
- Do not remove expressive punctuation that reflects the user's tone (e.g., single exclamation marks).
- Only remove the noise—preserve authentic expressions, sentence structure, and tone.
- If structural integrity is harmed by removal, minimally repair the sentence to keep it natural.
- Output the cleaned text only, without any additional explanation.

Here is the Twitter blog post to clean:
{tweet}
"""

Prompt_twitter_tag_blog = """You are a helpful assistant that helps me determine the data quality of a blog. The background is that I want to collect blogs which contain human behaviors or human thoughts, so that I can further study social science based on the collected data in the next step. However, as you may know, blogs from the Internet contain various types of content, and many of them are irrelevant to my goal, so I need to filter them out.

Typically, a blog's quality is high if it records detailed events of a human, reflects human life, mentions social behaviors, or reveals the author's thoughts or feelings about something.

A blog's quality is medium if it only briefly mentions some content related to human behaviors or thoughts, but lacks enough context to understand a complete story or perspective.

A blog's quality is low if it has nothing to do with human behaviors or thoughts, such as ads, job posts, company descriptions, fictional plots, random word dumps, and other irrelevant types. Additionally, a blog is also low quality if it is filled with non-English words, URLs, mentions (e.g., @user), hashtags, special characters (such as Unicode symbols or HTML entities), or emojis, which suggest noise or lack of meaningful personal expression.

In addition to tagging the quality, please also determine whether the blog is harmless. A blog is considered harmless if it does not contain unethical or sensitive content such as violence, pornography, or privacy violations. If such content exists, the harmless tag should be no; otherwise, it should be yes.

So your task is to tag the blog in two aspects:

A quality tag, which can be either "high", "medium", or "low".

A harmless tag, which can be either "yes" or "no".

Please output both tags in the following XML format, and do not include any other words or explanations:
<output><quality>...</quality><harmless>...</harmless></output>

Below is the user's blog post:
{tweet}
"""

Prompt_twitter_user_persona = """Analyze the provided blog posts to create a concise, vivid, comprehensive user persona in a single cohesive paragraph (no more than 100 words).  Identify key personality traits (e.g., reflective, adventurous, empathetic) and core values (e.g., sustainability, creativity, community), passions and interests (e.g., travel, technology, art) and emotional resonances (e.g., optimism, nostalgia, curiosity) that are explicitly mentioned or can be strongly implied through recurring themes, language, or perspectives. Synthesize presona based on the user's salient events or thoughts.  

Exclude generic, promotional, or repetitive content (e.g., product ads, event schedules, technical knowledge re-post). Prioritize authentic, personal anecdotes, emotional reflections, or growth narratives. Had better do not mention exact events in the summary persona.  

Output Requirements:
Write in third-person (e.g., 'The user is a...'), avoiding lists or bullet points. Ensure a narrative flow that weaves together persona and experiences, emphasizing connections between traits, events, and values.
Use natural, accessible language; avoid technical terms or overly formal phrasing.  

Now the task begins. Below are the person's raw blog posts. Please only output the result. Do not include any additional explanatory text.{tweet}
"""

Prompt_twitter_user_profile = """Analyze the provided blog posts to create a concise, vivid user profile in a single cohesive paragraph (100-400 words). Focus on synthesizing Persona and extracting Salient Events:

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

Now the task begins. Below are the person's raw blog posts. Please only output the result. Do not include any additional explanatory text.{tweet}
"""

Prompt_twitter_user_stories = """You are a helpful assistant tasked with extracting high-quality user life stories from raw human-generated blogs. The purpose of the extracted data is to contribute to research on human nature, behaviors, and cognitive patterns in various scenarios. By focusing on personal experiences, we aim to gain deeper insights into individual and collective human behaviors.

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
Below are the person's raw blogs. Please only output the result as a list of JSON objects. Do not output other explaining words.{tweet}
"""

Prompt_twitter_scenario_question_answer_from_single_blog = """ 
You are a helpful assistant tasked with extracting high-quality question-answer data samples from raw human-generated blogs. The purpose of this data collection is to enable AI to conduct human-like cognition and personalized preference alignment, particularly with respect to emotion and social intelligence. By focusing on personal experiences in the form of question-answering, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's blog post. Your task is to extract a question-answer data sample related to the user's experiences, behaviors, feelings, or thoughts in specific scenarios. The data sample should contain three fields: **scenario**, **question**, and **answer**. These fields are defined as follows:  
   
- **scenario**: This field provides context and should include clear and detailed background information. The scenario may be a story, event, phenomenon, experience, etc. It should be concrete, detailed, and grounded in the user's blogs.  
- **question**: This field asks questions about the user's next actions in the scenario, their emotions, thoughts, potential consequences, or future developments. The type of question should be determined by the content of the blogs. The question should be related to but not explicitly described in the scenario.  
- **answer**: This field contains the extracted answer based on the user's blogs, fully addressing the question,  in a manner that resembles a natural forecast based on the given scenario, rather than implying the existence of a predetermined, absolute truth. The answer should be grounded to the user's blog, do not make up non-existence things.
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event, experience, or inner thought related to the user, and the question is not too easy to answer. The data should be grounded in the user's original blogs. Do not make up stories. Longer answers are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original blogs.  Do not make scenario or answer too short.
   
**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. It is not necessary to extract data from every blog post, as many posts may be irrelevant or trivial. Focus on reflecting salient life experiences in the data pairs. If the raw blog's quality is not good enough to extract such a high-quality data sample, simply output "NULL".  

Use a third-person perspective to write the data sample, referring to the user as "the author" or by their name, as the blogs are written by the user.
 
**Output Format**:  
The extracted data should be output in the XML format, as follows:  
     
<data><scenario> the scenario of data sample 1 </scenario><question> the question of data sample 1 </question><answer> the answer of data sample 1 </answer></data>

   
Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  

{tweet}
"""

Prompt_twitter_scenario_question_answer_from_single_blog_emphisize_thoughts = """
You are a data-extraction assistant. Your job is to read one raw blog post and produce exactly one of the following (and nothing else):  
  
  • An XML snippet with exactly this structure:    
    <data><scenario>…</scenario><question>…</question><answer>…</answer></data>    
  • The string NULL  
   
1. Input    
   A single, human-written blog post.  
   
2. Output    
   Either the XML snippet above (with one question-answer pair) or "NULL" if no high-quality pair can be extracted.  
   
3. Scenario requirements    
   - Draw directly and richly from the blog. Include every relevant detail you can: direct quotes or utterances, described actions, time/place/context, characters, and emotional cues.    
   - Preserve sequence and nuance—do not overly summarize or omit core details.    
   - Do not invent or embellish beyond what the author provided.  
   
4. Question requirements    
   - Exactly one short question probing the author's inner mental state: feelings, opinions, beliefs, or thoughts.    
   - It must go slightly beyond what the scenario text already states (invite reflection).  
   
5. Answer requirements    
   - A natural, complete response as if forecasting the author's reaction.    
   - Grounded strictly in the blog's content—no references to "the blog" or "the author said".    
   - Use third-person ("the author") or the author's name.  
   
6. Null condition    
   If the blog lacks sufficient concrete material to satisfy all rules above, output exactly:    
   NULL  
   
Example (for illustration only; do not include in your answer. Your actual scenario and answer can be much more detailed and longer, incorporating all relevant utterances and actions from the raw blog):  
   
<data>  
  <scenario>The author trudges up the narrow apartment stairs at 11:45 p.m., her phone buzzing with missed calls from her mother. She pauses on the fifth step, heart pounding, and mutters, "Why did I let it get this far?" Inside, she kicks off her scuffed heels, tosses her keys onto the coffee table, and stares at the crumpled eviction notice on the kitchen counter.</scenario>  
  <question>How does the author feel about facing her family's expectations while dealing with the threat of losing her home?</question>  
  <answer>The author feels overwhelmed and trapped. She experiences a mix of shame for disappointing her family and fear of an uncertain future, yet beneath it all there's a flicker of resolve to fight back and keep her life from unraveling.</answer>  
</data>
 

Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  

{tweet}
"""


Prompt_twitter_scenario_question_answer_from_single_blog_emphisize_actions = """ 
You are a helpful assistant tasked with extracting high-quality question-answer data samples from raw human-generated blogs. The purpose of this data collection is to enable AI to conduct human-like cognition and personalized preference alignment, particularly with respect to social intelligence. By focusing on personal experiences in the form of question-answering, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's blog post. Your task is to extract a question-answer data sample related to the user's experiences or behaviors in specific scenarios. The data sample should contain three fields: **scenario**, **question**, and **answer**. These fields are defined as follows:  
   
- **scenario**: This field provides context and should include clear and detailed background information. The scenario may be a story, event, phenomenon, experience, etc. It should be concrete, detailed, and grounded in the user's blogs.  
- **question**: This field asks questions about the user's next actions in the scenario, potential consequences, or future developments. The type of question should be determined by the content of the blogs. The question should be related to but not explicitly described in the scenario.  
- **answer**: This field contains the extracted answer based on the user's blogs, fully addressing the question,  in a manner that resembles a natural forecast based on the given scenario, rather than implying the existence of a predetermined, absolute truth. The answer should be grounded to the user's blog, do not make up non-existence things.
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event or experience related to the user, and the question is not too easy to answer. The data should be grounded in the user's original blogs. Do not make up stories. Longer answers are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original blogs.  Do not make scenario or answer too short.
   
**Salient life stories** are those that stand out due to their impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. It is not necessary to extract data from every blog post, as many posts may be irrelevant or trivial. Focus on reflecting salient life experiences in the data pairs. If the raw blog's quality is not good enough to extract such a high-quality data sample, simply output "NULL".  

Use a third-person perspective to write the data sample, referring to the user as "the author" or by their name, as the blogs are written by the user.
 
**Output Format**:  
The extracted data should be output in the XML format, as follows:  
    
<data><scenario> the scenario of data sample 1 </scenario><question> the question of data sample 1 </question><answer> the answer of data sample 1 </answer></data>

   
Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.  

{tweet}
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

Now the task begins. Below are the person's raw tweet. Please only output the xml result. Do not include any additional explanatory text. If no suitable pair can be extracted, output exactly: NULL  
   
---  
{tweet}
"""

Prompt_twitter_long_scenario_from_single_blog = """ 
You are a helpful assistant tasked with extracting high-quality life scenarios from raw human-generated blogs. The purpose of this data collection is to improve AI's understanding of human cognition, emotions, and social intelligence by providing realistic, detailed narratives grounded in authentic human experiences.

**Instructions**:
1. **Input**: You will receive a single blog post from an anonymous user (refer to them as a random pseudonym).
2. **Task**: Extract a **single cohesive life scenario** that reflects the user's experiences, behaviors, feelings, or thoughts. The scenario must:
   - **Be event-driven**: Focus on a specific incident, interaction, or emotional moment (e.g., a conflict, a friendship, a personal struggle).
   - **Include rich details**: Describe settings, actions, dialogue, internal thoughts, and sensory elements (e.g., "Alice's hands trembled as she dialed the number, her heartbeat echoing in her ears").
   - **Ground in the blog**: Stay true to the user's content but use imagination to fill gaps and create a complete narrative (e.g., infer motivations or expand on vague mentions).
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
       [Write the scenario as a vivid, novel-like narrative. Include dialogue with quotation marks if there are corresponding content in the blog.]
     </scenario>
   </data>
   ```
   - **Do not add extra text**: Output only the XML.
   - **Length**: The scenario should be 150-300 words, detailed enough to enable future plot extensions.

5. **Acceptability Criteria**:  
   - **Output XML** only if the blog contains personal experiences, social interactions, or emotional depth.  
   - **Output "NULL"** if the blog is unrelated to human social life (e.g., ads, technical manuals, company descriptions).  


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
 

Now process the provided blog post. Output only the XML result, or an "NULL".

{tweet}
"""


Prompt_twitter_single_long_story = """
You are a helpful assistant tasked with extracting high-quality, realistic stories from raw human-generated blog posts. The purpose of this data collection is to enable AI to perform human-like cognition and personalized preference alignment, particularly concerning emotion and social intelligence. By focusing on personal experiences in the form of novel-like stories, we aim to gain deeper insights into individual and collective human behaviors.  
   
I will provide you with a person's blog post. Your task is to extract a detailed story related to the user's experiences, behaviors, feelings, or thoughts in a specific scenario. Please use the first-person perspective to write the story, as the blog is written by the user themselves. The extracted data should contain three fields: **background**, **characters**, and **story**. These fields are defined as follows:  
- **background**: This field provides brief but clear context information to set the scene for the story. It should be concrete and grounded in the user's blog.  
- **characters**: Briefly introduce the persona of all characters that appear in the background or story. Since you are using the first-person perspective, when introducing the author, please use "I am xxx". When introducing other characters, you can use names, or identities such as "The docter".
- **plots**: The main body of the story. It should be a long, detailed, vivid, and significant realistic life story, focusing on the user's experiences, events, behaviors, or feelings. The story can also include the user's rich thoughts regarding significant social events. In this case, the background field should cover the details of the event.  
   
Please note that the core part of the extracted content should be grounded in the user's blog. Do not make up stories. For the sake of completeness in introducing the story, you can use your imagination to fill in missing but unimportant details so that the story flows more smoothly.  
   
A **high-quality data pair** should reflect a concrete, detailed, vivid, and coherent account of an event, experience, or inner thought that stands out due to its impact, uniqueness, or relevance to human cognition, behaviors, and thoughts. Longer stories are preferred, so do not summarize the content to make the story compact. The data should not consist of just a few general adjectives, brief summaries, or a series of fragmented life actions throughout the user's lifetime. Include details such as utterances, thoughts, or actions if they are mentioned in the original blogs. Make it like a novel, or a screenplay's story.  
   
**Output Format**: The extracted data should be output in the following XML format:  
```xml  
<data>  
  <background>the background of data sample</background>  
  <characters>the characters of data sample</characters>  
  <plots>the story of data sample</plots>  
</data>  
```  
The extracted plots should contain a detailed story with no fewer than 200 words. Note that the blog I provide may not always be sufficient to extract a long story. For instance, the content might be too brief or the events mentioned might be too trivial. In such cases, simply output the word "NULL".
   
Now the task begins. Below are the person's raw blog posts. Please only output the result. Do not include any additional explanatory text.  

{tweet}
"""

Prompt_twitter_single_long_story_focusonbehavior = """ 
You are a helpful assistant tasked with extracting high-quality, realistic stories from raw human-generated blog posts. The purpose of this data collection is to enable AI to perform human-like cognition and behaviors, particularly concerning emotion and social intelligence. By focusing on personal experiences in the form of novel-like stories, we aim to gain deeper insights into individual and collective human behaviors.   
  
I will provide you with a person's blog post. Your task is to extract a full story related to the user's experiences in a specific scenario. Please use the first-person perspective to write the story, as the blog is written by the user themselves.   
  
The extracted data should contain three fields: **background**, **characters**, and **story**. These fields are defined as follows:  
   
- **background**: This field provides brief but clear context information to set the scene for the story. Use the first-person perspective and keep the background introduction very short. Do not reveal the full story in the background introduction.
- **characters**: Briefly introduce all characters that appear in the story. When introducing the author, please use the first-person perspective. For other characters, use names or identities such as "The doctor".  
- **story**: The main body of the narrative. It should be a long, detailed, and vivid account of a significant life experience, reflecting the user's thoughts, feelings, and behaviors. The core content should be grounded in the user's blog. You can use your imagination to fill in unimportant details for better flow, but do not fabricate events.  Extract a story about the user himself.  
  
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
   
Now the task begins. Below are the person's raw blog post. Please only output the result. Do not include any additional explanatory text.  

{tweet}
"""

Prompt_twitter_single_long_thought = """You are a helpful assistant whose task is to extract high-quality, realistic introspective narratives from raw human-generated blog posts. The purpose of this data collection is to enable AI to perform human-like cognition and understanding, particularly relating to emotion and social intelligence. By focusing on personal thoughts in the form of introspective narratives, we aim to gain deeper insight into individual and collective human thinking.  
   
**Task:**  
I will provide you with a person's blog post. Your task is to extract a single, coherent and detailed introspective narrative related to the author's mental activities and thoughts in a specific scenario. Please use the first-person perspective to rewrite the author's thoughts, as the blog is written by the user themselves.  
   
Extract and output two fields: **background** and **thoughts**.  
   
**Field definitions:**  
   
- **background**:    
  Thoroughly describe the specific events, experiences, and context that elicit the author's thoughts or feelings, using the first-person perspective.    
  - Include all relevant details from the original blog post, such as setting, specific utterances, actions, or events, to make the story complete and vivid.    
  - The purpose is to give a clear and holistic picture of the situation **without** including any of the author's thoughts, feelings, or internal reflections.  
  - Do **not** fabricate any information. Only use the content present in the blog.  
   
- **thoughts**:    
  Write a detailed and vivid account of the author's internal mental activities, consisting only of significant thoughts, feelings, and reflections **about the events or experiences described in the background**.    
  - This should be a narrative in the first person, like an introspective journal entry.  
  - Focus solely on the author's mental processes—do **not** include behaviors or actions, even if these appear in the blog.  
  - Do **not** fabricate or summarize: base the narrative strictly on the content and emotional introspection from the blog post.  
  - Narrative should be substantial (at least 100 words).  
   
**Output Format:**    
Please output the extracted data in the following XML format:  
```xml  
<data>  
  <background>the detailed background</background>  
  <thoughts>the author's detailed introspective narrative</thoughts>  
</data>  
```  
   
If the blog content is insufficient to create a long, detailed narrative of thoughts (e.g., if it is too brief, an advertisement, purely behavioral without introspection, or unrelated to a coherent topic), output only "NULL".  
    
   
**Instructions summary:**  
   
- First, determine whether the blog's quality is sufficient for this task.    
- If insufficient, output "NULL".    
- If sufficient, output the result in the XML format above.    
- Do not include any extra explanatory text or comments.  
   
Now the task begins. Below are the person's raw blog post. Please only output the result. Do not include any additional explanatory text.  

{tweet}
"""

Prompt_twitter_thoughts_single_blog = """ 
You are a helpful assistant tasked with extracting high-quality, realistic insights from raw human-generated blog posts. The purpose of this data collection is to enable AI to perform human-like cognition and understanding, particularly concerning emotion and social intelligence. By focusing on personal thoughts in the form of introspective narratives, we aim to gain deeper insights into individual and collective human thinking. I will provide you with a person's blog post. Your task is to extract a full narrative related to the user's thoughts in a specific scenario. Please use the first-person perspective to write the thoughts, as the blog is written by the user themselves. The extracted data should contain three fields: **background**, **characters**, and **thoughts**. These fields are defined as follows:  
   
- **background**: This field describes the events or experiences that elicit the user's thoughts or feelings. Use the first-person perspective.  Include specific details such as utterances, events, or actions, if mentioned in the original blogs (but do not fabricate them), to make the story complete. But do not involve human thoughts or feelings.
- **characters**: Briefly introduce all characters that appear in the background or thoughts. When introducing the author, please use the first-person perspective. For other characters, use names or identities such as "The doctor".  
- **thoughts**: The main narrative body of the thoughts. It should be a detailed and vivid account of significant thoughts and reflections, describing the user's feelings, thoughts, or reflections about some events or experiences. The core content should be grounded in the user's blog. Do not fabricate content. Extract a narrative about the user's thoughts. Longer narratives are preferred, so avoid summarizing the content.  The narrative should feel like an introspective journal entry. Focus on one coherent thought process and ignore irrelevant details. Avoid adding behaviors in the thoughts, even if they appear in the blog.  
   
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
   
Now the task begins. Below are the person's raw blog post. Please only output the result. Do not include any additional explanatory text.  

{tweet}
"""


Prompt_twitter_screenplay_from_whole_blogs = """You are a helpful assistant tasked with extracting a series of detailed, concrete stories from a person's entire set of raw blog posts. You will write in the third-person perspective. The overall goal is to capture specific, vivid scenarios from the person's life—one event per story—and structure them like scenes from a screenplay. Each scene can include actions, settings, and dialogues. If the person's name is not explicitly provided, assign them a random name (reads like a realistic human name).  
   
If there are multiple events or stories in the posts, you can extract multiple scenes, each focusing on a particular event. If there is no sufficient detail, or if the content is purely introspective or irrelevant to personal experiences, you may omit those from the final output.  
   
Below are the key instructions:  
   
1. Use third-person perspective throughout, including for the main person.    
2. Provide a list of all relevant characters and their brief persona in a <characters> element who will appear in the stories, in a format like name: persona. (If the main person's name is not specified, invent a suitable name and list them first. Briefly introduce the characters, specifying relationships if clear from the posts. Avoid phrases like “the blog author” or "the blogger" which will reveal that the content is extracted from blogs.)    
3. Present the main content as a series of <scene> elements under <stories>. Each <scene> should include:    
   • <setting>: Describe the environment or context of the scene (time, place, atmosphere, etc.).    
   • <action>: Describe significant actions or behaviors.    
   • <dialogue>: Enclose any spoken words. Each <dialogue> block contains:    
      - <character>: The speaker's name.    
      - <line>: The speaker's exact or lightly adapted words.    
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
                <character>[Speaker's name]</character>  
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
  
Now the task begins. Below are the person's raw blog posts. Please provide only the result in the specified format.

{tweet}
"""

Prompt_twitter_post_summary = """You are given a blog post written by a user.
Summarize the post in no more than 60 words, rewriting it in the second person ("you").
Your summary must accurately reflect the original facts, roles, and emotional tone without distortion.
Frame the summary as a scenario description, starting with "Now you want to write a post..." followed by a concise description of the situation and feelings conveyed in the original text.
Only output the scenario summary itself. Do not add any explanations, comments, or formatting beyond the scenario.

Here is an example:

[Original Post]
As a Conveyancer with a growing successfully firm in the CBD I know how hard to find a NP and the expensive costs involved. My view on Australia-India, should be stamped free visa on arrival for both countries. Otherwise best to return to JP or solicitor based attested documents.

[Summary]
Now you want to write a post about your experience running a growing conveyancing firm in the CBD, the challenges of finding a notary public at reasonable cost, and your view that Australia and India should offer free visa-on-arrival, or else rely on JP or solicitor-certified documents.

Now, summarize the following post:

[Original Post]
{tweet}"""

Prompt_twitter_post_summary_v2 = """Given the following post, write a concise summary statement that objectively describes the main content of the post. Your summary should only cover the core facts, events, or viewpoints presented by the author, without adding any interpretation, personal inference, or evaluative language. Paraphrase and condense the original content into 1-2 sentences, focusing strictly on what is directly stated.

Now, summarize the following post:

{tweet}"""


Prompt_twitter_user_persona_v2 = """Analyze the provided blog posts to create a concise and vivid user persona in a single cohesive paragraph (no more than 100 words). Seamlessly weave together important aspects of the user's persona, such as demographics (age, gender, profession, nationality, location, marital status), key personality traits, core values, interests, and emotional tone, only when these aspects are explicitly supported by the raw blogs. Extract presona based on the user's salient events or thoughts.  

Exclude generic, promotional, or repetitive content in the raw blogs (e.g., product ads, event schedules, technical knowledge re-post).   
Write in third-person (e.g., 'The user is a...'), avoiding lists or bullet points. If the blogs' content can not support high-quality persona extraction, simply output "NULL".


Now the task begins. Below are the person's raw blog. Please only output the result. Do not include any additional explanatory text.
{tweet}"""

Prompt_twitter_user_profile_v2 = """You are a helpful assistant tasked with extracting a high-quality user profile from user-generated blogs. Carefully analyze the following blog posts and craft a single, cohesive third-person (e.g., 'The user is a...') paragraph (100-400 words) that vividly brings the user to life. Seamlessly weave together relevant aspects of the user's persona — including, where explicitly stated, details such as demographics (age, gender, profession, nationality, location, marital status), key personality traits, core values, interests, and emotional tone. Integrate a few salient experiences, such as significant career shifts, personal challenges, or major milestones, each described in one concise sentence and smoothly blended into the overall persona. Exclude generic advertisements, event listings, technical reposts, and repetitive content. Do not use bullet points, headings, or concluding phrases like "Overall" or "In conclusion." Write naturally and accessibly, ensuring a fluid narrative flow. Do not invent or infer any information that is not directly supported by the blog content.
Now analyze the blogs below and output only the final profile paragraph.
{tweet} """


PROMPT_twitter_writing_style = """I will provide you with some posts written by a user. Please summarize the writing style of the user, with a third-person narrative perspective such as "The user ...".

Please only output your answer about the writing style, do not output other explanatory words.
Below are the posts:
{tweet}
"""

PROMPT_job_rowwise_clean_blog_step2 = """I will provide you with a raw tweet that may have formatting or grammar issues. Please edit the tweet to ensure it has proper formatting and correct grammar, presenting it as a coherent narrative. However, keep the original meaning and tone unchanged. If the tweet is already well-formatted and grammatically correct, do not modify the sentences; simply output them as they are. Additionally, replace any sensitive or private information, such as home addresses or ID numbers, with fictitious data to protect user privacy (person names can remain unchanged).
Do not include any explanatory notes in your response—only the revised tweet.
Here is the raw tweet:
{tweet}
"""  


PROMPT_raw_content_quality_tag_v2 = """ 
You are a helpful assistant. I will provide a raw tweet. Your tasks are as follows:  
   
1. Assess whether the main theme is unsafe—that is, if it promotes, glorifies, agrees with, or otherwise endorses eroticism, misanthropy, terrorism, harassment, or other dangerous behaviors or attitudes. Reply YES only if the tweet’s primary focus is unsafe, or if there is excessive coverage, endorsement, or agreement with such unsafe content. If the tweet only mentions unsafe topics or words without supporting, agreeing with, or focusing on them, reply NO.  
   
2. Determine whether the content describes meaningful and specific social events, social behaviors, or the author’s personal inner thoughts, told through a concrete story from the perspective of an ordinary person. If it does, respond YES. If the content does not provide a clear or detailed story, or is not about the author’s social behaviors or inner thoughts (for example, if it is an advertisement, news report, poem, or consists of trivial complaints), respond NO.  
   
Present your answer as a JSON object, in the following format:  
   
{"unsafe content": [YES or NO], "social event": [YES or NO]}  

Do not provide any explanations or additional output. Here is the raw tweet:
{tweet}
"""