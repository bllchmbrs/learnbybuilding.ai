#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('python --version')


# In[10]:


get_ipython().system('pip freeze | grep guardrails-ai')


# In[11]:


import pydantic
print(pydantic.__version__)


# In[12]:


from dotenv import load_dotenv
load_dotenv()


# In[13]:


import feedparser

podcast_atom_link = "https://api.substack.com/feed/podcast/1084089.rss" # latent space podcastbbbbb
parsed = feedparser.parse(podcast_atom_link)
episode = [ep for ep in parsed.entries if ep['title'] == "Why AI Agents Don't Work (yet) - with Kanjun Qiu of Imbue"][0]
episode_summary = episode['summary']
print(episode_summary[:100])


# In[14]:


from unstructured.partition.html import partition_html

parsed_summary = partition_html(text=''.join(episode_summary)) 
start_of_transcript = [x.text for x in parsed_summary].index("Transcript") + 1
print(f"First line of the transcript: {start_of_transcript}")
text = '\n'.join(t.text for t in parsed_summary[start_of_transcript:])
text = text[:3508] # shortening the transcript for speed & cost


# ## Using Guardrails

# In[15]:


import guardrails as gd

from pydantic import BaseModel
from typing import Optional, List
from pydantic import Field

class Person(BaseModel):
    name: str
    school: Optional[str] = Field(..., description="The school this person attended")
    company: Optional[str] = Field(..., description="The company this person works for")

class People(BaseModel):
    people: List[Person]

guard = gd.Guard.from_pydantic(output_class=People, prompt="Get the following objects from the text:\n\n ${text}")


# In[16]:


import openai
import os

raw_llm_output, validated_output = guard(
    openai.ChatCompletion.create,
    prompt_params={"text": text},
)

print(validated_output)


# In[17]:


class Company(BaseModel):
    name:str

class ResearchPaper(BaseModel):
    paper_name:str = Field(..., description="an academic paper reference discussed")
    
class ExtractedInfo(BaseModel):
    people: List[Person]
    companies: List[Company]
    research_papers: Optional[List[ResearchPaper]]

guard = gd.Guard.from_pydantic(output_class=ExtractedInfo, prompt="Get the following objects from the text:\n\n ${text}")
raw_llm_output, validated_output = guard(
    openai.ChatCompletion.create,
    prompt_params={"text": text},
)

print(validated_output)


# **Note:** The result failed to parse, while the other two libraries succeeded just fine. It might be in the implementation or might be a bug in this version, but it didn't work out of the box.
