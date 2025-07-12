

import requests
from bs4 import BeautifulSoup
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+

google_api_key = "AIzaSyD00vbnGX8UrPy7mTuKk8foJaZ64j8fmkw"
CX = "16397320c346b4714"

# 📦 Install dependencies first:
# pip install langchain langchain-community openai faiss-cpu rank_bm25 beautifulsoup4 requests
# 1. Install required packages:
# pip install langchain langchain-huggingface langchain-community faiss-cpu rank_bm25 beautifulsoup4 requests sentence_transformers
# Install these before running:
# pip install langchain langchain-huggingface langchain-community faiss-cpu rank_bm25 beautifulsoup4 requests sentence_transformers
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
import re

class TextProcessor:
    def __init__(self):
        self.stop_words = self._initialize_nltk()
        self.tokenizer = self._get_tokenizer()

    def _initialize_nltk(self):
        """Initialize NLTK with fallback to basic stopwords"""
        try:
            import nltk
            from nltk.corpus import stopwords
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
                
            return set(stopwords.words('english'))
        except:
            # Basic English stopwords fallback
            return {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}

    def _get_tokenizer(self):
        """Get the best available tokenizer with fallbacks"""
        try:
            from nltk.tokenize import word_tokenize
            # Verify punkt is available
            try:
                import nltk
                nltk.data.find('tokenizers/punkt')
                return word_tokenize
            except:
                try:
                    nltk.download('punkt')
                    return word_tokenize
                except:
                    return self._simple_tokenize
        except:
            return self._simple_tokenize

    def _simple_tokenize(self, text):
        """Basic tokenizer fallback"""
        return re.findall(r'\b\w+\b', text.lower())

    def preprocess_text(self, text):
        """Robust text preprocessing with multiple fallbacks"""
        try:
            tokens = self.tokenizer(text.lower())
            return [token for token in tokens if token.isalpha() and token not in self.stop_words]
        except:
            return self._simple_tokenize(text)

class GoogleSearchProcessor:
    def __init__(self, api_key, cx_id):
        self.api_key = api_key
        self.cx_id = cx_id
        self.text_processor = TextProcessor()
        
    def search_google(self, query, num_results=5):
        """Fetch results from Google Custom Search JSON API"""
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={self.api_key}&cx={self.cx_id}&num={num_results}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            items = response.json().get('items', [])
            
            processed_results = []
            for item in items[:num_results]:
                processed_results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'content': f"{item.get('title', '')}. {item.get('snippet', '')}"
                })
            return processed_results
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return []

    def rank_with_tfidf(self, query, documents):
        """Rank documents using TF-IDF cosine similarity"""
        vectorizer = TfidfVectorizer(tokenizer=self.text_processor.preprocess_text)
        try:
            tfidf_matrix = vectorizer.fit_transform([query] + [doc['content'] for doc in documents])
            cosine_similarities = (tfidf_matrix[0] * tfidf_matrix[1:].T).toarray()[0]
            ranked_indices = np.argsort(cosine_similarities)[::-1]
            return [documents[i] for i in ranked_indices]
        except:
            # Fallback: return original order if TF-IDF fails
            return documents

    def rank_with_bm25(self, query, documents):
        """Rank documents using BM25 algorithm"""
        try:
            tokenized_docs = [self.text_processor.preprocess_text(doc['content']) for doc in documents]
            bm25 = BM25Okapi(tokenized_docs)
            tokenized_query = self.text_processor.preprocess_text(query)
            doc_scores = bm25.get_scores(tokenized_query)
            ranked_indices = np.argsort(doc_scores)[::-1]
            return [documents[i] for i in ranked_indices]
        except:
            # Fallback: return original order if BM25 fails
            return documents

    def create_llm_context(self, query, ranked_docs, max_words=500):
        """Create concise context for LLM input"""
        context = f"Search Query: {query}\n\nTop Results:\n"
        word_count = 0
        
        for i, doc in enumerate(ranked_docs, 1):
            doc_text = f"{i}. {doc['title']}\n{doc['snippet']}\nURL: {doc['url']}\n\n"
            doc_words = len(doc_text.split())
            
            if word_count + doc_words <= max_words:
                context += doc_text
                word_count += doc_words
            else:
                remaining_words = max_words - word_count
                if remaining_words > 20:
                    truncated = ' '.join(doc_text.split()[:remaining_words]) + "... [truncated]"
                    context += truncated
                break
                
        return context.strip()

# Example Usage
if __name__ == "__main__":
    # Initialize with your Google API credentials
    processor = GoogleSearchProcessor(
        api_key="AIzaSyD00vbnGX8UrPy7mTuKk8foJaZ64j8fmkw",
        cx_id="16397320c346b4714"
    )
    
    # Example search
    query = "latest advancements in renewable energy 2024"
    print("Performing search...")
    results = processor.search_google(query)
    
    if not results:
        print("No results found or search failed")
    else:
        print(f"Found {len(results)} results. Processing...")
        ranked_results = processor.rank_with_bm25(query, results)
        llm_context = processor.create_llm_context(query, ranked_results)
        
        print("\n=== Generated Context for LLM ===")
        print(llm_context)