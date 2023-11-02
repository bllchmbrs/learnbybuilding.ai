#!/usr/bin/env python
# coding: utf-8

# # Structured Data Extraction using LLMs and Marvin AI 
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
# In this post, you'll learn how to extract structured data from LLMs into Pydantic objects using the [Marvin AI](https://www.askmarvin.ai/welcome/what_is_marvin/) library.
# 
# ## Here's us setting up on environment

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


# ## Background on Marvin
# 
# ![Ask Marvin AI](https://images.learnbybuilding.ai/marvin-ai-screenshot.webp)
# 
# Marvin, as a library, has larger ambitions than just a data extraction library. In the creators' words,
# 
# > Marvin is a lightweight AI engineering framework for building natural language interfaces that are reliable, scalable, and easy to trust.
# > 
# > Sometimes the most challenging part of working with generative AI is remembering that it's not magic; it's software. It's new, it's nondeterministic, and it's incredibly powerful - but still software.
# > 
# > Marvin's goal is to bring the best practices for building dependable, observable software to generative AI. As the team behind [Prefect](https://www.prefect.io/), which does something very similar for data engineers, we've poured years of open-source developer tool experience and lessons into Marvin's design.
# 
# At the time of this writing, [Marvin](https://www.askmarvin.ai/), has a growing discord, is version 1.5.5 (and therefore ready for production). It also has the highest github stars of our contenders as well as the most forks.
# 
# Marvin is commercially backed by [Prefect](https://www.prefect.io/).
# 
# 
# Overall, Marvin is a fairly simple feeling library. You'll see that below.
# 
# Let's get started with the tutorial.
# 
# 
# ## Pre-processing the data
# 
# Here, we are fetching and parsing an RSS feed from a podcast. We specifically target an episode by its title and then retrieve its summary from the podcast description. This is similar to a [tutorial on llamaindex](https://learnbybuilding.ai/tutorials/rag-chatbot-on-podcast-llamaindex-faiss-openai) that we recently published.

# In[8]:


import feedparser

podcast_atom_link = "https://api.substack.com/feed/podcast/1084089.rss" # latent space podcastbbbbb
parsed = feedparser.parse(podcast_atom_link)
episode = [ep for ep in parsed.entries if ep['title'] == "Why AI Agents Don't Work (yet) - with Kanjun Qiu of Imbue"][0]
episode_summary = episode['summary']
print(episode_summary[:100])


# Now we're going to extract the shortened episode summary. We'll also shorten the summary just to speed up our extraction. In the future, we can extend the transcript to cover the whole thing.

# In[9]:


from unstructured.partition.html import partition_html

parsed_summary = partition_html(text=''.join(episode_summary)) 
start_of_transcript = [x.text for x in parsed_summary].index("Transcript") + 1
print(f"First line of the transcript: {start_of_transcript}")
text = '\n'.join(t.text for t in parsed_summary[start_of_transcript:])
text = text[:3508] # shortening the transcript for speed & cost


# ## Using Marvin
# 
# The first step is to define our pydantic models. You'll notice that we [use Python Decorators](https://realpython.com/primer-on-python-decorators/) the actual class that we'll be classing with `@ai_model`. This allows us to call this class directly on our data.
# 
# This is a key aspect of the user experience of Marvin.

# In[10]:


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


# Once we do that, we simply call our function on the input data. Simple!

# In[11]:


People(text)


# Now that we've extracted the people from the podcast, we can make things a bit more complicated.
# 
# Let's try to extract not just people but also the companies mentioned as well as any research papers mentioned.
# 
# This hints at some of the power of large language models, we're taking unstructured text and turning it into JSON objects (well, pydantic objects, but that's no problem) that we can use in down stream pipelines. This is an extremely powerful tool to have in our tool belt.
# 
# Let's see how it does!

# In[13]:


class Company(BaseModel):
    name:str

class ResearchPaper(BaseModel):
    paper_name:str = Field(..., description="an academic paper reference discussed")

@ai_model(instructions="Get the following information from the text")
class ExtractedInfo(BaseModel):
    people: List[Person]
    companies: List[Company]
    research_papers: Optional[List[ResearchPaper]]


# Once we do that, we simply call our function on the input data. Simple!

# In[14]:


ExtractedInfo(text)


# Overall, it did a great job! We got some good results!
# 
# 
# **Note**: What's interesting is that if you don't supply the `instructions` you won't get as strong of results. I started off with `instructions` blank and got poor results. When I did the same with instructor, it worked fine.
# 
# Changing the decorator instructions gave me better results. This is also allows you to [configure th LLM](https://www.askmarvin.ai/components/ai_model/#configuring-the-llm) as well as temperature and more.
# 
# 
# ## Wrapping it all up
# 
# The sky is really the limit here. There's so much structured data that's been locked up in unstructured text. I'm bullish on this space and can't wait to see what you build with this tool!
# 
# If you found this interesting, you might want to see similar posts on:
# 1. [Comparing 3 Data Extraction Libraries: Marvin, Instructor, and Guardrails](https://learnbybuilding.ai/vs/marvin-ai-vs-guardrails-vs-instructor)
# 2. [Structured Data Extraction using LLMs and Instructor](https://learnbybuilding.ai/tutorials/structured-data-extraction-with-instructor-and-llms)
# 3. [Structured Data Extraction using LLMs and Guardrails AI](https://learnbybuilding.ai/tutorials/structured-data-extraction-with-guardrails-and-llms)
