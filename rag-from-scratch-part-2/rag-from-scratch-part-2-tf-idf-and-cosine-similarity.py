#!/usr/bin/env python
# coding: utf-8

# # Rag From Scratch - Part 2 TFIDF and Cosine Similarity to Improve Similarity Search

# In[ ]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install pinecone-client')
get_ipython().system('pip install sentence-transformers')


# In[34]:


from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


# ![Bad Similarity Problems in Retrieval Augmented Generation](images/a-key-challenge-of-retrieval-augmented-generation-systems-semantics.jpg)

# In[35]:


# Define the corpus of documents
corpus_of_documents = [
    "Take a leisurely walk in the park and enjoy the fresh air.",
    "Visit a local museum and discover something new.",
    "Attend a live music concert and feel the rhythm.",
    "Go for a hike and admire the natural scenery.",
    "Have a picnic with friends and share some laughs.",
    "Explore a new cuisine by dining at an ethnic restaurant.",
    "Take a yoga class and stretch your body and mind.",
    "Join a local sports league and enjoy some friendly competition.",
    "Attend a workshop or lecture on a topic you're interested in.",
    "Visit an amusement park and ride the roller coasters."
]


# Generate embeddings for the documents
doc_embeddings = model.encode(corpus_of_documents)


# In[36]:


query = "What's the best outside activity?"


# In[37]:


doc_embeddings


# In[38]:


similarities = cosine_similarity(model.encode([query]), doc_embeddings)


# In[39]:


similarities[0]


# In[40]:


indexed = list(enumerate(similarities[0]))


# In[41]:


indexed


# In[42]:


sorted_index = sorted(indexed, key=lambda x: x[1], reverse=True)


# In[43]:


sorted_index


# In[44]:


recommended_documents = []
for value, score in sorted_index:
    formatted_score = "{:.2f}".format(score)
    print(f"{formatted_score} => {corpus_of_documents[value]}")
    if score > 0.3:
        recommended_documents.append(corpus_of_documents[value])


# ## Adding in our LLM: Llama 2

# In[45]:


import requests
import json


# In[61]:


prompt = """
You are a bot that makes recommendations for activities. You answer in very short sentences and do not include extra information.

These are potential activities:

{recommended_activities}


The user's query is: {user_input}

Provide the user with 2 recommended activities based on their query.
"""

recommended_activities = "\n".join(recommended_documents)


# In[62]:


print(recommended_activities)


# In[63]:


user_input = "I like to hike"


# In[64]:


full_prompt = prompt.format(user_input=user_input, recommended_activities=recommended_activities)


# In[65]:


url = 'http://localhost:11434/api/generate'
data = {
    "model": "llama2",
    "prompt": full_prompt
}

headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)
full_response=[]
try:
    count = 0
    for line in response.iter_lines():
        #filter out keep-alive new lines
        # count += 1
        # if count % 5== 0:
        #     print(decoded_line['response']) # print every fifth token
        if line:
            decoded_line = json.loads(line.decode('utf-8'))
            
            full_response.append(decoded_line['response'])
finally:
    response.close()
print(''.join(full_response))


# ![simplified version of retrieval augmented generation](images/simplified-version-of-retrieval-augmented-generation.jpg)
# 
# The LLM (if you're lucky) will handle the user input that goes against the recommended document. We can see that below.

# In[ ]:





# 1. Pinecone documentation: https://docs.pinecone.io/docs/overview
# 2. Sentence Transformers documentation: https://www.sbert.net/docs/quickstart.html
# 3. scikit-learn TF-IDF documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# 4. scikit-learn cosine similarity documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
# 

# In[ ]:




