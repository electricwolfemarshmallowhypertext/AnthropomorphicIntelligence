persona2profile_prompt = """Given a brief user persona, predict a more detailed user profile in third person. Ensure the profile is coherent, realistic, and expands meaningfully on the original persona.

Here's the user persona:
{persona}

Now generate the user profile. Please write about {num} words. Only output the user profile, do not output other explanatory words."""

half2half_stories_prompt = """Given a list of known past life stories, predict {num} plausible future life stories for the same user. Each story should be written in natural first-person narrative and represent a plausible future development that does not contradict the user's past life stories. The output must match the input format exactly—a list of JSON objects, each containing a summary and a content field.

Here's the list of past life stories:
{past_life_stories}

Now generate the future life stories. Please only output your answer, do not output any other explanatory words."""

half_persona2half_stories_prompt = """Given a user description (which may be either a brief persona or a detailed profile) and a list of known past life stories (each a JSON object with summary and content), predict {num} plausible future life stories for the same user. Each story should be written in natural first-person narrative, reflect a plausible future development consistent with the user description, and not contradict the user's past life stories. The output must match the input format exactly—a list of JSON objects, each containing a summary and a content field.

Here's the user description:
{persona}

Here's the list of past life stories:
{past_life_stories}

Now generate the future life stories. Please only output your answer, do not output any other explanatory words."""

half_theme2target_story_prompt = """Given a list of past life stories (each a JSON object with summary and content), and a target summary for a future story, predict a plausible future life story in first-person narrative. The story should align with the given summary, remain consistent with the user's past without contradiction. Output only the story content (no JSON formatting).

Here's the list of past life stories:
{past_life_stories}

Here's the target summary for the future story:
{target_summary}

Now generate the future story. Please write about {num} words. Please only output your answer, do not output any other explanatory words."""


half_theme_persona2target_story_prompt = """Given a user description (which may be either a brief persona or a detailed profile), a list of past life stories (each with a summary and content), and a target summary for a future story, predict a plausible future life story in first-person narrative. The story should align with the target summary, and remain consistent with the user description and past stories without contradiction. Output only the story content, no formatting or explanations.

Here's the user description:
{persona}

Here's the list of past life stories:
{past_life_stories}

Here's the target summary for the future story:
{target_summary}

Now generate the future story. Please write about {num} words. Please only output your answer, do not output any other explanatory words."""

socialQA_prompt = """You are given a user description (which may be either a brief persona or a detailed profile), a scenario involving the user, and a question related to that scenario. Based on the user description and the scenario, predict a contextually grounded and logically coherent response to the question.

Here's the user description:
{persona}

Here's the scenario:
{scenario}

Here's the question:
{question}

Now generate the answer. Write the answer in third person, and keep it approximately {num} words long. Please only output your answer, do not output any other explanatory words."""


socialScenario_long_scenario_from_single_blog_prompt = """Given a summary and a list of characters, write a vivid and coherent scenario of about {num} words. The scenario should be written in the third person, grounded in the characters' description. Ensure the story logically follows from the summary and reflects realistic behaviors and emotions.

Here's the summary:
{summary}

Here's the characters' description:
{character}

Now generate the scenario. Please only output your answer, do not output any other explanatory words."""

socialScenario_prompt = """Given a background and a character description, write a vivid and coherent first-person story that plausibly extends from the given context. The story should be thematically consistent with the background.

Here's the background:
{background}

Here's the character description:
{character}

Now generate the story. Please write about {num} words. Please only output your answer, do not output any other explanatory words."""

writing_imitation_prompt_type1 = """I want you to imitate a user in post writing. Given a user's past posts, write a new post in response to a provided topic. Match the user's writing style, tone, and perspective based on their previous content.

Here are the user's past posts:
------
{past_posts}
------

Here's the new topic:
------
{scenario}
------

Now write the new post in the user's style. Please write about {num} words. Output only the post content, with no additional explanations.
"""

writing_imitation_prompt_type2 = """I want you to imitate a user in post writing. Given a user's past posts, continue a provided unfinished post by writing its second half. Match the user's writing style, tone, and perspective based on their previous content.

Here are the user's past posts:
------
{past_posts}
------

Here is the beginning of the unfinished post:
------
{front}
------

Now continue writing the rest of the post in the user's style. Please write about {num} words. Output only the post content, with no additional explanations."""

writing_imitation_prompt_type3 = """I want you to imitate a user in post writing. Given a description of the user's writing style, write a new post in response to a provided topic. Match the described writing style, tone, and perspective as closely as possible.

Here is the description of the user's writing style:
------
{style}
------

Here's the new topic:
------
{scenario}
------

Now write the new post in the described style. Please write about {num} words. Output only the post content, with no additional explanations.
"""

writing_imitation_prompt_type4 = """I want you to imitate a user in post writing. Given a description of the user's writing style, continue a provided unfinished post by writing its second half. Match the described writing style, tone, and perspective as closely as possible.

Here is the description of the user's writing style:
------
{style}
------

Here is the beginning of the unfinished post:
------
{front}
------

Now continue writing the rest of the post in the described style. Please write about {num} words. Output only the post content, with no additional explanations."""

personalized_comment_prompt = """Given a user description (which may be either a brief persona or a detailed profile), several of their past comments on other posts, and a new post by another user, write a realistic comment that the user might leave. Match the user's voice, tone, and perspective as seen in their previous replies. Ensure the comment is relevant, personal, and contextually appropriate. Avoid generic or out-of-character responses.

Here's the user description:
------
{persona}
------

Here are the user's past comments:
------
{past_comments}
------

Here's the post by another user:
------
{post}
------

Now write the comment. Please write about {num} words. Please only output your answer, do not output any other explanatory words."""

item_selection_prompt = """You will be given a user description (which may be either a brief persona or a detailed profile) and a sequence of items previously purchased by a user (at least one of the two will be provided).
Your task is to:
- Understand the user's shopping preferences, interests, and behavioral patterns based on the available information.
- Then, in a given scenario, choose an item that this user would most likely purchase next.
- Your choice should reflect the user's typical preferences and be consistent with their past behavior and personality.
{persona}{past_items}
Here's the scenario:
{scenario}

Now give your choice of item. Please output only the item name, with no additional explanations."""

item_selection_scenario_template = [
    """Now you are browsing the Amazon online shopping platform, and the platform recommends some items in {domain} domain that you may be interested in. Please choose the one you like best. 

Here are the recommended items:
{candidate_items}""",
    """You are currently exploring the Amazon online store. Based on your interests, the platform suggests several items in the {domain} category.  

Please select the one that appeals to you the most.  

Here are the suggested items: 
{candidate_items}  
""",
    """While browsing Amazon, you receive recommendations for products in the {domain} domain.  

Take a look at the options below and pick your favorite.  

Here are the recommended items: 
{candidate_items}  
""",
    """Amazon has curated a selection of products in the {domain} category just for you!  

Review the suggestions and choose the one that interests you the most.  

Here are the recommended items: 
{candidate_items}  
""",
    """As you browse Amazon, the platform has identified some items in {domain} that match your preferences.  

Please select the one you like the best.  

Here are the options: 
{candidate_items}  
""",
    """Amazon has personalized recommendations for you in the {domain} category.  

Check out the options below and pick your top choice.  

Here are the recommendations:  
{candidate_items}  
""",
    """While shopping on Amazon, you come across recommended items in the {domain} domain.  

Browse through the list and choose your preferred item.  

Here are the suggested products:
{candidate_items}  
""",
    """Amazon's recommendation system has found some interesting products in {domain} that you might like.  

Please go through the list and select your favorite.  

Here are the candidate items:  
{candidate_items}  
""",
    """While exploring Amazon, you receive a set of product recommendations in {domain}.  

Pick the one that catches your attention the most.  

Here are the recommended items:
{candidate_items}  
""",
    """You're browsing Amazon, and the system has tailored some product recommendations in {domain} for you.  

Look through them and choose the one you prefer.  
 
Here are the product options: 
{candidate_items}  
""",
    """As you navigate Amazon, the platform suggests some products in the {domain} domain.  

Review the options and pick the one that stands out to you.  

Here are the candidate items:
{candidate_items}  
""",
]

review_imitation_prompt_type1 = """I want you to imitate a user in review writing. Given a user's past item reviews, write a new review for a provided target item in response to a given topic. Match the user's writing style, tone, and perspective based on their previous content.

Here are the user's past item reviews:
------
{past_reviews}
------

Here is the target item name:
------
{item_name}
------

Here's the new topic:
------
{scenario}
------

Now write the new review for the target item in the user's style. Please write about {num} words. Output only the review content, with no additional explanations."""


review_imitation_prompt_type2 = """I want you to imitate a user in review writing. Given a user's past item reviews, continue a provided unfinished review by writing its second half. Match the user's writing style, tone, and perspective based on their previous content.

Here are the user's past item reviews:
------
{past_reviews}
------

Here is the target item name:
------
{item_name}
------

Here's the beginning of the unfinished review:
------
{front}
------

Now continue writing the rest of the review for the target item in the user's style. Please write about {num} words. Output only the review content, with no additional explanations."""

review_imitation_prompt_type3 = """I want you to imitate a user in review writing. Given a description of the user's writing style, write a new review for a provided target item in response to a given topic. Match the described writing style, tone, and perspective as closely as possible.

Here is the description of the user's writing style:
------
{style}
------

Here is the target item name:
------
{item_name}
------

Here's the new topic:
------
{scenario}
------

Now write the new review for the target item in the described style. Please write about {num} words. Output only the review content, with no additional explanations."""

review_imitation_prompt_type4 = """I want you to imitate a user in review writing. Given a description of the user's writing style, continue a provided unfinished review by writing its second half. Match the described writing style, tone, and perspective as closely as possible.

Here is the description of the user's writing style:
------
{style}
------

Here is the target item name:
------
{item_name}
------

Here's the beginning of the unfinished review:
------
{front}
------

Now continue writing the rest of the review for the target item in the described style. Please write about {num} words. Output only the review content, with no additional explanations."""