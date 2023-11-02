#!/usr/bin/env python
# coding: utf-8

# # Structured Data Extraction using LLMs and Instructor
# 
# In this post, we're going to be extracting structured data from a podcast transcript. We've all seen the ability for generative models to be effective *generating* text.
# 
# **But what about extracting data?**
# 
# Data extraction is a far more common use case (today) than generation, in particular for businesses. Businesses have to process all kinds of documents, exchange them and so on.
# 
# **Emerging Use Case: LLMs for data extraction**
# 
# If you think about it, summarization is effectively data extraction. We've seen LLMs perform well at summarization, but did you know that they can extract structured data quite well?
# 
# ## What you'll learn from this post
# 
# In this post, you'll learn how to extract structured data from LLMs into Pydantic objects using the [Instructor](https://jxnl.github.io/instructor/) library.
# 
# ## Here's us setting up on environment

# In[1]:


get_ipython().system('python --version')


# In[2]:


get_ipython().system('pip freeze | grep instructor')


# In[3]:


import pydantic
print(pydantic.__version__)


# In[4]:


from dotenv import load_dotenv
load_dotenv()


# ## Background on Instructor
# 
# [Instructor](https://jxnl.github.io/instructor/) is a library that helps get structured data out of LLMs. It has narrow dependencies and a simple API, requiring basic nothing sophisticated from the end user.
# 
# In the author's words:
# 
# > `Instructor` helps to ensure you get the exact response type you're looking for when using openai's function call api. Once you've defined the `Pydantic` model for your desired response, `Instructor` handles all the complicated logic in-between - from the parsing/validation of the response to the automatic retries for invalid responses. This means that we can build in validators 'for free' and have a clear separation of concerns between the prompt and the code that calls openai.
# 
# The library is still early, version 0.2.9 has a modest star count (1300) but hits above it's weight since it's basically a single person working on it.
# 
# ![instructor contributors page](https://images.learnbybuilding.ai/instructor-contributions-page.webp)
# 
# Now just because it's one person doesn't mean it's bad. In fact, I found it to be very simple to use. It just means that from a long term maintenance perspective, it might be challenging. Something to note.
# 
# ## Pre-processing the data
# 
# Here, we are fetching and parsing an RSS feed from a podcast. We specifically target an episode by its title and then retrieve its summary from the podcast description. This is similar to a [tutorial on llamaindex](https://learnbybuilding.ai/tutorials/rag-chatbot-on-podcast-llamaindex-faiss-openai) that we recently published.

# In[5]:


import feedparser

podcast_atom_link = "https://api.substack.com/feed/podcast/1084089.rss" # latent space podcastbbbbb
parsed = feedparser.parse(podcast_atom_link)
episode = [ep for ep in parsed.entries if ep['title'] == "Why AI Agents Don't Work (yet) - with Kanjun Qiu of Imbue"][0]
episode_summary = episode['summary']
print(episode_summary[:100])


# Now we're going to extract the shortened episode summary. We'll also shorten the summary just to speed up our extraction. In the future, we can extend the transcript to cover the whole thing.

# In[6]:


from unstructured.partition.html import partition_html

parsed_summary = partition_html(text=''.join(episode_summary)) 
start_of_transcript = [x.text for x in parsed_summary].index("Transcript") + 1
print(f"First line of the transcript: {start_of_transcript}")
text = '\n'.join(t.text for t in parsed_summary[start_of_transcript:])
text = text[:3508] # shortening the transcript for speed & cost


# ## Using Instructor
# 
# Instructor is the "thinnest" of [the libraries that I've used for structured data extraction](https://learnbybuilding.ai/vs/marvin-ai-vs-guardrails-vs-instructor). The interface is super simple.

# In[8]:


from pydantic import BaseModel
from typing import Optional, List
from pydantic import Field

class Person(BaseModel):
    name: str
    school: Optional[str] = Field(..., description="The school this person attended")
    company: Optional[str] = Field(..., description="The company this person works for ")

class People(BaseModel):
    people: List[Person]


# All we do is [Monkey patch](https://en.wikipedia.org/wiki/Monkey_patch#:~:text=Monkey%20patching%20is%20a%20technique,Python%2C%20Groovy%2C%20etc.)) OpenAI's SDK.
# 
# **Note:** This could be hard to maintain, so take note of versions and be sure to not let either one get upgraded without testing.

# In[9]:


import openai
import instructor

instructor.patch()


# Now we simply call OpenAI but ask for a `response_model`. This is the instructor patch at work!

# In[11]:


response = openai.ChatCompletion.create(
    model="gpt-4",
    response_model=People,
    messages=[
        {"role": "user", "content": text},
    ]
)
print(response)


# Overall the result is high quality, we've gotten all the names and companies.
# 
# Now we can take this to the next level by trying to extract multiple objects at the same time.

# In[12]:


class Company(BaseModel):
    name:str

class ResearchPaper(BaseModel):
    paper_name:str = Field(..., description="an academic paper reference discussed")
    
class ExtractedInfo(BaseModel):
    people: List[Person]
    companies: List[Company]
    research_papers: Optional[List[ResearchPaper]]

response = openai.ChatCompletion.create(
    model="gpt-4",
    response_model=ExtractedInfo,
    messages=[
        {"role": "user", "content": text},
    ]
)

print(response)


# You can see how well this works, to grab a whole bunch of structured data for us with basically no work. Right now we're just working on a excerpt of this data, but with basically zero NLP specific work, we're able to get some pretty powerful results.
# 
# 
# ## Wrapping it all up
# 
# The sky is really the limit here. There's so much structured data that's been locked up in unstructured text. I'm bullish on this space and can't wait to see what you build with this tool!
# 
# If you found this interesting, you might want to see similar posts on:
# 1. [Comparing 3 Data Extraction Libraries: Marvin, Instructor, and Guardrails](https://learnbybuilding.ai/vs/marvin-ai-vs-guardrails-vs-instructor)
# 2. [Structured Data Extraction using LLMs and Marvin](https://learnbybuilding.ai/tutorials/structured-data-extraction-with-marvin-ai-and-llms)
# 3. [Structured Data Extraction using LLMs and Guardrails AI](https://learnbybuilding.ai/tutorials/structured-data-extraction-with-guardrails-and-llms)
