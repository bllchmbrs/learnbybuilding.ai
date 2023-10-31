#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('python --version')


# In[6]:


import pydantic
print(pydantic.__version__)
import marvin
print(marvin.__version__)


# In[7]:


from dotenv import load_dotenv
load_dotenv()


# ## Pre-processing the data

# In[15]:


import feedparser

podcast_atom_link = "https://api.substack.com/feed/podcast/1084089.rss" # latent space podcastbbbbb
parsed = feedparser.parse(podcast_atom_link)
episode = [ep for ep in parsed.entries if ep['title'] == "Why AI Agents Don't Work (yet) - with Kanjun Qiu of Imbue"][0]
episode_summary = episode['summary']
print(episode_summary[:100])


# In[16]:


from unstructured.partition.html import partition_html

parsed_summary = partition_html(text=''.join(episode_summary)) 
start_of_transcript = [x.text for x in parsed_summary].index("Transcript") + 1
print(f"First line of the transcript: {start_of_transcript}")
text = '\n'.join(t.text for t in parsed_summary[start_of_transcript:])
text = text[:3508] # shortening the transcript for speed & cost


# ## Using Marvin

# In[14]:


from marvin import ai_model
from pydantic import BaseModel
from typing import Optional, List
from pydantic import Field

class Person(BaseModel):
    name: str
    school: Optional[str] = Field(..., description="The school this person attended")
    company: Optional[str] = Field(..., description="The company this person works for")

@ai_model
class People(BaseModel):
    people: List[Person]

People(text)


# In[23]:


class Company(BaseModel):
    name:str

class ResearchPaper(BaseModel):
    paper_name:str = Field(..., description="an academic paper reference discussed")

@ai_model(instructions="Get the following information from the text")
class ExtractedInfo(BaseModel):
    people: List[Person]
    companies: List[Company]
    research_papers: Optional[List[ResearchPaper]]

ExtractedInfo(text)


# **Note**: What's interesting is that if you don't supply the `instructions` you won't get as strong of results. I started off with `instructions` blank and got poor results. When I did the same with instructor, it worked fine.
# 
# Changing the decorator instructions gave me better results. This is also allows you to [configure th LLM](https://www.askmarvin.ai/components/ai_model/#configuring-the-llm) as well as temperature and more.
