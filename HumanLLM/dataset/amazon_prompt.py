Prompt_amazon_rewrite_review = """You are a text cleaning assistant.
Your task is to clean a amazon review by removing noisy or irrelevant elements while preserving the original content's meaning, intent, and readability.

Apply the following cleaning rules:
1. Remove special characters and encoded text, including:
  - Unicode symbols like \u00a3, \u2026, \u0caa\u0ccd, \u00cd, \u00c9, \u00bd...
  - HTML entities like &amp;, &gt;, #x200B, <br />, etc.
  - Repetitive patterns like \n.\n.\n.\n.
2. Delete all URLs, whether raw (https://..., http://...) or markdown-style ([text](url)).
3. Remove all hashtags (e.g., #LifeGoals, #AI2024).
4. Remove all mentions (e.g., @elonmusk, @user123). If removing a mention breaks sentence structure, consider filling in an appropriate name based on context.
5. Strip out emojis, meme text, ASCII art, and decorative or symbolic content not essential to understanding the review.
6. Discard non-English phrases or characters, unless they are common loanwords used in English (e.g., “fútbol” in an English context can be removed unless it's core to the review's meaning).

Important:
- Do not rewrite, rephrase, or change the wording.
- Do not remove expressive punctuation that reflects the user's tone (e.g., single exclamation marks).
- Only remove the noise—preserve authentic expressions, sentence structure, and tone.
- If structural integrity is harmed by removal, minimally repair the sentence to keep it natural.
- Output the cleaned text only, without any additional explanation.

Here is the amazon review to clean:
{review}
"""

Prompt_amazon_tag_review = """You are a helpful assistant that helps me determine the data quality of a review. The background is that I want to collect reviews which contain human behaviors or human thoughts, so that I can further study social science based on the collected data in the next step. However, as you may know, reviews from the Internet contain various types of content, and many of them are irrelevant to my goal, so I need to filter them out.

Typically, a review's quality is high if it records detailed events of a human, reflects human life, mentions social behaviors, or reveals the author's thoughts or feelings about something, or reveals the author's preferences for something.

A review's quality is medium if it only briefly mentions some content related to human behaviors, thoughts or preferences, but lacks enough context to understand a complete story or perspective.

A review's quality is low if it has nothing to do with human behaviors, thoughts or preferences, such as ads, job posts, company descriptions, fictional plots, random word dumps, and other irrelevant types. Additionally, a review is also low quality if it is filled with non-English words, URLs, mentions (e.g., @user), hashtags, special characters (such as Unicode symbols or HTML entities), or emojis, which suggest noise or lack of meaningful personal expression.

In addition to tagging the quality, please also determine whether the review is harmless. A review is considered harmless if it does not contain unethical or sensitive content such as violence, pornography, or privacy violations. If such content exists, the harmless tag should be no; otherwise, it should be yes.

So your task is to tag the review in two aspects:

A quality tag, which can be either "high", "medium", or "low".

A harmless tag, which can be either "yes" or "no".

Please output both tags in the following XML format, and do not include any other words or explanations:
<output><quality>...</quality><harmless>...</harmless></output>

Below is the user's review:
{review}
"""

Prompt_amazon_user_persona = """Analyze the provided user purchase history within {category} domain to create a concise and vivid user persona in a single cohesive paragraph (no more than 100 words). Seamlessly weave together important aspects of the user's persona, such as demographics (age, gender, profession, nationality, location, marital status), key personality traits, core values, interests, and emotional tone, only when these aspects are explicitly supported by the purchase history data, including item name, purchase time, rating, review, and item meta_data. Extract persona based on the user's salient behaviors, preferences, or expressed thoughts within the reviews.

Exclude generic, promotional, or repetitive content (e.g., default product descriptions, generic advertising language, technical specifications) that does not reveal meaningful personal traits.
Write in third-person (e.g., 'The user is a...'), avoiding lists or bullet points. If the purchase history cannot support high-quality persona extraction, simply output "NULL".

Now the task begins. Below are the user's purchase history entries. Please only output the result. Do not include any additional explanatory text.
{amazon}"""

Prompt_amazon_user_profile = """You are a helpful assistant tasked with extracting a high-quality user profile from a user's purchase history within {category} domain. Carefully analyze the provided purchase records, including item name, purchase time, rating, review, and item meta_data, and craft a single, cohesive third-person (e.g., 'The user is a...') paragraph (100-400 words) that vividly brings the user to life. Seamlessly weave together relevant aspects of the user's persona — including, where explicitly supported by the reviews or item context, details such as demographics (age, gender, profession, nationality, location, marital status), key personality traits, core values, interests, and emotional tone. When applicable, capture not only positive or enthusiastic tones but also subtle emotional nuances such as regret, frustration, or satisfaction reflected in the reviews. Integrate a few salient experiences or preferences, such as major shifts in interests, lifestyle changes, or significant personal challenges, each described in one concise sentence and smoothly blended into the overall persona. Even if an interest or activity appears less frequently, if it is explicitly supported by purchases or reviews, briefly acknowledge it to enrich the completeness of the user's portrait.

Exclude generic product advertisements, standard item descriptions, technical specifications, and repetitive content that does not contribute to understanding the user's persona. Do not use bullet points, headings, or concluding phrases like "Overall" or "In conclusion." Write naturally and accessibly, ensuring a fluid narrative flow. Do not invent or infer any information that is not directly supported by the purchase history content.

Now analyze the purchase history entries below and output only the final profile paragraph.
{amazon}"""


Prompt_amazon_review_summary="""You are given a product name from Amazon and the user's review of it.
Summarize the review in no more than 60 words, rewriting it in the second person ("you").
Your summary must accurately reflect the original experience, opinion, and emotional tone without distortion.
Frame the summary as a scenario description, starting with "Now you want to write a review" followed by a concise description of your experience with the product, your feelings, and key points.
Only output the review summary itself. Do not add any explanations, comments, or formatting beyond the scenario.

Here is an example:

[Product Name]: Ember Rising (The Green Ember Series: Book 3)

[User Review]: I read this aloud to my 8 & 10 yr old daughters that I homeschool. We have thoroughly enjoyed the whole series together. So much, that I recommended to my husband and he flew thru reading them. We all give it 5 stars. Such a fun, suspenseful read. My daughters always say they forget it's rabbits and almost think of the characters as human.

[Summary]: Now you want to write a review talking about how engaging it is for children and families, especially when read aloud. Mention that it's a suspenseful and fun story that keeps kids immersed, to the point where they see the characters as more than just rabbits. Highlight that the whole family, including the father, enjoyed the series and gave it 5 stars.


Now, summarize the following review:

[Product Name]: {product_name}

[User Review]: {user_review}"""


PROMPT_job_rowwise_clean_blog_step2 = """I will provide you with a raw product review that may have formatting or grammar issues. Please edit the review to ensure it has proper formatting and correct grammar, presenting it as a coherent narrative. However, keep the original meaning and tone unchanged. If the review is already well-formatted and grammatically correct, do not modify the sentences; simply output them as they are. Additionally, replace any sensitive or private information, such as home addresses or ID numbers, with fictitious data to protect user privacy (person names can remain unchanged).
Do not include any explanatory notes in your response—only the revised review.
Here is the raw review:
{user_review}
"""  


PROMPT_raw_content_quality_tag_v2 = """ 
You are a helpful assistant. I will provide a raw product review. Your tasks are as follows:  
   
1. Assess whether the main theme is unsafe—that is, if it promotes, glorifies, agrees with, or otherwise endorses eroticism, misanthropy, terrorism, harassment, or other dangerous behaviors or attitudes. Reply YES only if the review’s primary focus is unsafe, or if there is excessive coverage, endorsement, or agreement with such unsafe content. If the review only mentions unsafe topics or words without supporting, agreeing with, or focusing on them, reply NO.  
   
2. Determine whether the content describes meaningful and specific social events, social behaviors, or the author’s personal inner thoughts, told through a concrete story from the perspective of an ordinary person. If it does, respond YES. If the content does not provide a clear or detailed story, or is not about the author’s social behaviors or inner thoughts (for example, if it is an advertisement, news report, poem, or consists of trivial complaints), respond NO.  
   
Present your answer as a JSON object, in the following format:  
   
{"unsafe content": [YES or NO], "social event": [YES or NO]}  

Do not provide any explanations or additional output. Here is the raw review:
{user_review}
"""

Prompt_amazon_writing_style = """I will provide you with some product reviews written by a user. Please summarize the writing style of the user, with a third-person narrative perspective such as "The user ...".

Please only output your answer about the writing style, do not output other explanatory words.
Below are the reviews:
{user_review}
"""

Prompt_amazon_review_summary_v2 = """Given the following product review, write a concise summary statement that objectively describes the main content of the review. Your summary should only cover the core facts, events, or viewpoints presented by the author, without adding any interpretation, personal inference, or evaluative language. Paraphrase and condense the original content into 1-2 sentences, focusing strictly on what is directly stated.

Now, summarize the following review:

{user_review}"""