import requests
from bs4 import BeautifulSoup
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
from collections import Counter
import math

class TextProcessor:
    def __init__(self):
        # Basic English stopwords
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'of', 'to', 'in', 'that', 'for', 'on', 'with', 'as', 'at', 'by',
            'it', 'this', 'that', 'be', 'have', 'has', 'had', 'not', 'will'
        }
        
    def preprocess_text(self, text):
        """Tokenize and clean text without NLTK"""
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in self.stop_words and len(word) > 2]

class ContentOptimizer:
    def __init__(self):
        self.text_processor = TextProcessor()
        
    def process_for_llm(self, text, target_length=500):
        """Optimize content for LLM input"""
        if not text:
            return ""
            
        # Step 1: Clean content
        cleaned = self._clean_text(text)
        
        # Step 2: Extract key content if too long
        if len(cleaned) > target_length * 1.5:
            cleaned = self._extract_key_content(cleaned, target_length)
        
        # Step 3: Final truncation
        return cleaned[:target_length]

    def _clean_text(self, text):
        """Comprehensive text cleaning"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove common noise patterns
        patterns = [
            r'advertisement.*?\n',
            r'sponsored content.*?\n',
            r'privacy policy.*?\n',
            r'terms of use.*?\n',
            r'©.*?\n',
            r'all rights reserved.*?\n',
            r'read more.*?\n'
        ]



        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_key_content(self, text, target_length):
        """Extract important sentences using TF-like scoring"""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        word_scores = self._calculate_word_scores(text)
        
        # Score sentences
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue
            score = sum(word_scores.get(word, 0) for word in self.text_processor.preprocess_text(sentence))
            scored_sentences.append((score, i, sentence))
        
        # Sort by score and select top sentences
        scored_sentences.sort(reverse=True)
        selected = []
        total_len = 0
        
        for score, i, sentence in scored_sentences:
            if total_len + len(sentence) > target_length:
                break
            selected.append((i, sentence))
            total_len += len(sentence)
        
        # Reconstruct in original order
        selected.sort()
        return ' '.join(sentence for i, sentence in selected)

    def _calculate_word_scores(self, text):
        """Calculate word importance scores"""
        words = self.text_processor.preprocess_text(text)
        word_counts = Counter(words)
        total_words = len(words)
        
        scores = {}
        for word, count in word_counts.items():
            scores[word] = count / total_words * math.log(1 + total_words / (1 + count))
            
        return scores

class GoogleSearchProcessor:
    def __init__(self, api_key, cx_id):
        self.api_key = api_key
        self.cx_id = cx_id
        self.text_processor = TextProcessor()
        self.content_optimizer = ContentOptimizer()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def search_google(self, query, num_results=5):
        """Fetch results from Google Custom Search JSON API"""
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={self.api_key}&cx={self.cx_id}&num={num_results}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            items = response.json().get('items', [])
            
            return [{
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'content': f"{item.get('title', '')}. {item.get('snippet', '')}"
            } for item in items[:num_results]]
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return []

    def get_page_content(self, url):
        """Fetch and optimize page content"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
                element.decompose()
            
            # Extract main content
            content_areas = []
            for selector in ['article', 'main', 'div.content', 'section', 'div.article-body']:
                content_areas.extend([e.get_text(separator=" ", strip=True) for e in soup.select(selector)])
            
            # Fallback to body if no specific containers found
            if not content_areas:
                content_areas = [soup.body.get_text(separator=" ", strip=True)]
            
            full_content = " ".join(content_areas)
            return self.content_optimizer.process_for_llm(full_content, 500)
        except Exception as e:
            print(f"Content fetch failed for {url}: {str(e)}")
            return None

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
            return documents

    def create_llm_context(self, query, documents, max_tokens=1500):
        """Create optimized context for LLM"""
        context = [f"Search Query: {query}\n\nTop Results:"]
        current_length = len(context[0])
        
        for i, doc in enumerate(documents, 1):
            content = doc.get('optimized_content', doc['snippet'])
            result_text = f"\n\n{i}. {doc['title']}\nURL: {doc['url']}\nContent: {content}"
            
            if current_length + len(result_text) <= max_tokens:
                context.append(result_text)
                current_length += len(result_text)
            else:
                remaining = max_tokens - current_length
                if remaining > 50:
                    context.append(result_text[:remaining] + "... [truncated]")
                break
                
        return "".join(context)

if __name__ == "__main__":
    # Initialize with your Google API credentials
    processor = GoogleSearchProcessor(
        api_key="",
        cx_id=""
    )
    
    # Example search
    query = "latest advancements in renewable energy 2024"
    print("Performing search...")
    results = processor.search_google(query)
    
    if results:
        print(f"Found {len(results)} results. Fetching content...")
        
        # Enhance results with full content
        for result in results:
            result['optimized_content'] = processor.get_page_content(result['url'])
            time.sleep(1)  # Polite delay between requests
        
        # Rank results
        ranked_results = processor.rank_with_bm25(query, results)
        
        # Generate LLM context
        llm_context = processor.create_llm_context(query, ranked_results)
        
        print("\n=== Optimized LLM Context ===")
        print(llm_context[:2000] + ("..." if len(llm_context) > 2000 else ""))
        print(f"\nContext length: {len(llm_context)} characters")
        
        # Save to file
        with open("search_context.txt", "w", encoding="utf-8") as f:
            f.write(llm_context)
        print("Context saved to 'search_context.txt'")
    else:
        print("No results found")
