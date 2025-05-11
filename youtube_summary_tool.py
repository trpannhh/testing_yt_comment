# First, regular imports
import argparse
import os
import shutil
import re
import sys
import time
import json

# Set matplotlib backend before importing anything matplotlib-related
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend

# Now it's safe to import pyplot
import matplotlib.pyplot as plt
from googleapiclient import discovery

# Continue with the rest of your imports
import emoji
import nltk
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Global constants
CHROMA_PATH = "chroma"

# Global variable to track the current video being analyzed
CURRENT_VIDEO_ID = None

# Models and Vector DBs for sentiment analysis
EMBEDDING_MODEL = OllamaEmbeddings(model="mxbai-embed-large")
POSITIVE_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
NEGATIVE_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="llama3")


def close_chroma_connection():
    """Close any open connections to the Chroma database."""
    # Force garbage collection to release file handles
    import gc
    gc.collect()
    time.sleep(1)  # Give a moment for resources to be released

def get_embedding_function():
    """Returns the embedding function for vector database."""
    return OllamaEmbeddings(model="mxbai-embed-large")


def extract_video_id(youtube_url):
    """Extract video ID from a YouTube URL."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)

    # If no patterns match, assume it's already a video ID if it's 11 chars
    if len(youtube_url) == 11:
        return youtube_url

    return None


def get_comments(video_id, api_key):
    """
    Fetch comments and replies from YouTube with improved data structure.

    Args:
        video_id: YouTube video ID
        api_key: YouTube API key

    Returns:
        List of comment dictionaries with author, text, likes, etc.
    """
    # Create a YouTube API client
    youtube = discovery.build('youtube', 'v3', developerKey=api_key)

    # Call the API to get the comments
    comments = []
    next_page_token = None
    total_comments = 0

    print("Fetching comments from YouTube API...")

    try:
        while True:
            # Request comments
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=next_page_token,
                maxResults=100,  # Maximum allowed by API
                textFormat='plainText'
            )
            response = request.execute()

            # Handle potential API errors
            if 'error' in response:
                print(f"API Error: {response['error']['message']}")
                break

            # Extract top-level comments and replies
            items_count = len(response.get('items', []))
            if items_count == 0:
                print("No comments found or all comments processed.")
                break

            print(f"Processing batch of {items_count} comment threads...")

            for item in response.get('items', []):
                # Top-level comment
                top_level_comment = item['snippet']['topLevelComment']['snippet']
                comment = top_level_comment['textDisplay']
                author = top_level_comment['authorDisplayName']
                likes = top_level_comment.get('likeCount', 0)

                # No 'replied_to' for top-level comment
                comments.append({
                    'author': author,
                    'comment': comment,
                    'likes': likes
                })
                total_comments += 1

                # Replies (if any)
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_author = reply['snippet']['authorDisplayName']
                        reply_comment = reply['snippet']['textDisplay']
                        reply_likes = reply['snippet'].get('likeCount', 0)

                        # Include the 'replied_to' field only for replies
                        comments.append({
                            'author': reply_author,
                            'comment': reply_comment,
                            'replied_to': author,
                            'likes': reply_likes
                        })
                        total_comments += 1

            # Print progress
            print(f"Fetched {total_comments} comments so far...")

            # Check for more comments (pagination)
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break  # No more pages, exit the loop

            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.5)

    except Exception as e:
        print(f"Error fetching comments: {str(e)}")

    print(f"Completed fetching {total_comments} comments.")
    return comments


def save_comments_to_chroma(comments, video_id):
    """
        Populate comments into Chroma database, clearing previous data if video ID changed.

        Args:
            comments: List of comment dictionaries
            video_id: YouTube video ID to check if we need to refresh the database

        Returns:
            Number of comments saved to the database
        """
    global CURRENT_VIDEO_ID, CHROMA_PATH

    # Check if we already have a database for this video
    if CURRENT_VIDEO_ID == video_id and os.path.exists(CHROMA_PATH):
        print(f"Using existing Chroma database for video ID: {video_id}")
        return len(comments)

    # If video ID changed or no database exists, rebuild it
    if os.path.exists(CHROMA_PATH):
        print(f"Video ID changed from {CURRENT_VIDEO_ID} to {video_id}. Removing existing Chroma database.")

        # Close any open connections before removing
        close_chroma_connection()

        try:
            shutil.rmtree(CHROMA_PATH)
        except Exception as e:
            print(f"Error removing Chroma directory: {e}")
            # If we can't remove the directory, create a new one with a timestamp
            timestamp = int(time.time())
            new_chroma_path = f"{CHROMA_PATH}_{timestamp}"
            print(f"Creating new database at {new_chroma_path}")
            CHROMA_PATH = new_chroma_path

    # Prepare the Chroma database
    print(f"Creating new Chroma vector database for video ID: {video_id}")
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Create Document objects for each comment
    documents = []
    for idx, comment in enumerate(comments, start=1):
        # Format the comment text to include author and likes
        if comment.get('likes', 0) > 0:
            content = f"{comment['author']} [👍 {comment['likes']}]:\n{comment['comment']}"
        else:
            content = f"{comment['author']}:\n{comment['comment']}"

        # Add metadata
        metadata = {
            "source": f"Comment {idx}",
            "author": comment['author'],
            "likes": comment.get('likes', 0)
        }

        if 'replied_to' in comment:
            # Add 'replied_to' for replies
            metadata['replied_to'] = comment['replied_to']
            # Mark as reply in the content for better context
            content = f"[REPLY to {comment['replied_to']}] {content}"

        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    # Add documents to Chroma in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        db.add_documents(batch)
        print(f"Added batch of {len(batch)} comments to Chroma (total {i + len(batch)})")

    # Update the current video ID
    CURRENT_VIDEO_ID = video_id

    # Save video metadata to help with QA
    with open(os.path.join(CHROMA_PATH, "video_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "comment_count": len(documents)
        }, f, ensure_ascii=False, indent=2)

    print(f"Successfully added all {len(documents)} comments to Chroma database.")
    return len(documents)


def read_comments_from_chroma():
    """Read comments from the Chroma database."""
    # Connect to the existing Chroma database
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Get all documents from the database
    results = db.get()

    # Extract comments from the documents
    comments = []
    for doc in results['documents']:
        # Each document has format "Author [👍 Likes]:\nComment" or "Author:\nComment"
        # Split to get just the comment part
        parts = doc.split('\n', 1)
        if len(parts) > 1:
            comments.append(parts[1])  # Just the comment text, not the author

    return comments


def calculate_optimal_k(total_comments):
    """
    Calculate the optimal k value based on total comment count.
    Optimized based on testing that k=50-70 works best for ~200 comments.

    Args:
        total_comments: Total number of comments in the database

    Returns:
        Recommended k value
    """
    # For very small comment sets (<50), use a higher percentage (60-70%)
    if total_comments < 50:
        return max(10, min(int(total_comments * 0.7), total_comments))

    # For small comment sets (50-200), scale between 40-30% of total
    elif total_comments < 200:
        # Linear scaling from 40% at 50 comments to 30% at 200 comments
        percent = 0.4 - ((total_comments - 50) / 150) * 0.1
        return max(20, min(int(total_comments * percent), total_comments))

    # For medium comment sets (200-1000), scale between 30-20% of total
    elif total_comments < 1000:
        # Linear scaling from 30% at 200 comments to 20% at 1000 comments
        percent = 0.3 - ((total_comments - 200) / 800) * 0.1
        return max(60, min(int(total_comments * percent), total_comments))

    # For large comment sets (1000-5000), scale between 20-10% of total
    elif total_comments < 5000:
        # Linear scaling from 20% at 1000 comments to 10% at 5000 comments
        percent = 0.2 - ((total_comments - 1000) / 4000) * 0.1
        return max(200, min(int(total_comments * percent), 500))

    # For very large comment sets (>5000), use 10% with a cap at 600
    else:
        return min(int(total_comments * 0.1), 600)


def answer_question(question, k=None):
    """
    Answer a question based on the YouTube comments data with improved analysis.

    Args:
        question: The user's question about the video comments
        k: Number of relevant comments to retrieve for context (auto-calculated if None)

    Returns:
        Dictionary with answer and metadata
    """
    # Start timing
    start_time = time.time()

    # Load the Chroma vector store
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Get the total number of documents in the database
    doc_count = len(db.get()['ids'])

    # Calculate optimal k if not specified
    if k is None:
        k = calculate_optimal_k(doc_count)
        print(f"Auto-calculated optimal k value: {k} (based on {doc_count} total comments)")

    # Store the original k value for reporting
    k_used = k

    # Adjust k if it's larger than the number of available documents
    if k > doc_count:
        print(f"Adjusting k from {k} to {doc_count} (total available documents)")
        k = doc_count

    # Improved prompt template with better structure and instructions
    PROMPT_TEMPLATE = """
    You are a YouTube comment analyst answering questions about video comments.

    QUESTION: {question}

    Below are relevant comments from the video:
    {context}

    Answer the question ONLY using information in these comments. Your response should:

    1. Start with a direct answer addressing the question
    2. Group similar opinions together
    3. Include specific quotes from commenters as evidence when relevant
    4. Stay STRICTLY focused on the question

    For comparison or preference questions:
    - Use clear headings
    - Use bullet points for listing multiple points
    - Structure information logically by categories

    For numerical questions (counts, percentages, etc.):
    - Provide a direct numerical answer if possible
    - Explain how you arrived at this number
    - Include specific evidence from comments

    DO NOT invent information not present in the comments.
    DO NOT include follow-up questions or recommendations unless requested.
    FOCUS only on answering exactly what was asked: {question}
    """

    print(f"Retrieving {k} most relevant comments for the question...")

    # Retrieve relevant documents
    results = db.similarity_search_with_score(question, k=k)

    retrieval_time = time.time() - start_time
    print(f"Retrieved {len(results)} comments in {retrieval_time:.2f} seconds")

    # Sort comments by relevance score to prioritize most relevant ones
    sorted_results = sorted(results, key=lambda x: x[1])

    # Take only the most relevant comments to avoid overwhelming the LLM
    top_results = sorted_results[:min(k, len(sorted_results))]

    # Build context string from retrieved documents with comment numbering
    context_parts = []
    for i, (doc, score) in enumerate(top_results):
        context_parts.append(f"[{i + 1}] {doc.page_content}")

    context_text = "\n\n".join(context_parts)

    # Format prompt with context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=question, context=context_text)

    # Use OllamaLLM model to generate the answer
    print("Generating answer with language model...")
    model = OllamaLLM(model="llama3")

    generation_start = time.time()
    response_text = model.invoke(prompt)
    generation_time = time.time() - generation_start

    total_time = time.time() - start_time
    print(f"Answer generated in {generation_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")

    # Return both the answer and metadata
    return {
        'answer': response_text,
        'k_used': k_used,
        'comments_total': doc_count,
        'processing_time': f"{total_time:.2f} seconds"
    }


def generate_comment_summary():
    """Generate a general summary of all comments with improved diversity."""
    # Load the Chroma vector store
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Get the total number of documents in the database
    doc_count = len(db.get()['ids'])

    # Calculate appropriate k value based on document count
    # For summaries, we want a larger sample than for QA but not too large
    base_k = calculate_optimal_k(doc_count)
    k = min(base_k * 2, doc_count)  # Double the QA k, but don't exceed doc count

    print(f"Using k={k} for summary (based on {doc_count} total comments)")

    # Use a more balanced prompt that emphasizes diversity
    PROMPT_TEMPLATE = """
    You are a YouTube comment summarizer. Below is a collection of user comments extracted from a video.

    {context}

    ---

    Please write a summary highlighting the key points and general sentiment expressed in these comments.
    Focus on providing a well-rounded overview in less than 5 paragraphs.
    
    IMPORTANT: Make sure to cover diverse topics from the comments. Do not focus too much on any single 
    topic or theme, even if many comments discuss it. Instead, try to capture the overall breadth of 
    topics and opinions present across ALL comments.
    """

    # Get a mix of targeted and random comments for better diversity
    similarity_k = k // 2
    random_k = k - similarity_k

    print(f"Retrieving {similarity_k} targeted comments and {random_k} random comments for summary...")

    # Get targeted comments using similarity search
    results1 = db.similarity_search_with_score("summarize youtube comments", k=similarity_k)

    # Get random comments for diversity
    import random
    all_docs = db.get()
    random_indices = random.sample(range(doc_count), min(random_k, doc_count))
    random_docs = [(Document(page_content=all_docs['documents'][i]), 1.0) for i in random_indices]

    # Combine both sets
    combined_results = results1 + random_docs

    # Build context string from retrieved documents
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in combined_results])

    # Format prompt with context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text)

    # Use OllamaLLM model
    print("Generating summary with language model...")
    model = OllamaLLM(model="llama3")
    response_text = model.invoke(prompt)

    # Save the output to a file
    with open("overall_summary.txt", "w", encoding="utf-8") as f:
        f.write(response_text)

    print("Overall summary saved to overall_summary.txt")
    return response_text


# ---------------------------------- PREPROCESSING ----------------------------------#


def find_emoticons(string):
    """Convert emoticons to text descriptions."""
    happy_faces = [' :D', ' :)', ' (:', ' =D', ' =)',
                   ' (=', ' ;D', ' ;)', ' :-)', ' ;-)', ' ;-D', ' :-D']
    sad_faces = [' D:', ' :(', ' ):', ' =(', ' D=', ' )=', ' ;(',
                 ' D;', ' )-:', ' )-;', ' D-;', ' D-:', ' :/', ' :-/', ' =/']
    neutral_faces = [' :P', ' :*', '=P', ' =S',
                     ' =*', ' ;*', ' :-|', ' :-*', ' =-P', ' =-S']
    for face in happy_faces:
        if face in string:
            string = string.replace(face, ' happy_face ')
    for face in sad_faces:
        if face in string:
            string = string.replace(face, ' sad_face ')
    for face in neutral_faces:
        if face in string:
            string = string.replace(face, ' neutral_face ')
    return string


def preprocess_for_sentiment(comment_list):
    """Preprocess comments for sentiment analysis."""
    processed_comments = []
    for comment in comment_list:
        # Deconvert emojis
        comment = emoji.demojize(comment)
        # Replace emoticons like :) or :( with label
        comment = find_emoticons(comment)
        # Remove URLs
        comment = re.sub(r'http\S+|www\S+|https\S+|t\.co\S+', '', comment)
        comment = unidecode(comment)
        # Clean excessive whitespace
        comment = re.sub(r'\s+', ' ', comment).strip()
        # Keep sentence as-is for VADER
        processed_comments.append(comment)
    return processed_comments


def preprocess_comment_for_wordcloud(comment_list):
    """Preprocess comments: remove stopwords and apply lemmatization."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_comments = []
    for comment in comment_list:
        # Tokenize
        words = word_tokenize(comment)
        # Remove stopwords and lemmatize
        cleaned = [
            lemmatizer.lemmatize(word)
            for word in words
            if word.lower() not in stop_words and word.isalpha()
        ]
        # Join tokens back
        processed_comments.append(' '.join(cleaned))
    return processed_comments


# ---------------------------------- SENTIMENT ANALYSIS ----------------------------------#

def analyze_sentiment(comments):
    """Analyze sentiment of comments using VADER."""
    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    comments_positive = []
    comments_negative = []
    comments_neutral = []

    # Count the number of neutral, positive, and negative comments
    num_neutral = 0
    num_positive = 0
    num_negative = 0

    for comment in comments:
        sentiment_scores = analyzer.polarity_scores(comment)
        if sentiment_scores['compound'] > 0.0:
            num_positive += 1
            comments_positive.append(comment)
        elif sentiment_scores['compound'] < 0.0:
            num_negative += 1
            comments_negative.append(comment)
        else:
            comments_neutral.append(comment)
            num_neutral += 1

    results = [num_neutral, num_positive, num_negative]
    # Return the results and categorized comments
    return results, comments_positive, comments_negative, comments_neutral


matplotlib.use('Agg')
plt.ioff()  # Turn off interactive mode


# Then modify your plot_sentiment_pie_chart function to not show the plot:
def plot_sentiment_pie_chart(results):
    """Create a pie chart of sentiment distribution."""
    # Get the counts for each sentiment category
    num_neutral = results[0]
    num_positive = results[1]
    num_negative = results[2]

    labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
    sizes = [num_positive, num_negative, num_neutral]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    explode = (0.1, 0, 0)  # explode 1st slice (Positive)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels,
           colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    plt.close(fig)  # Close the figure to prevent display
    return fig


# ---------------------------------- WORD CLOUD ----------------------------------#


# Similarly, modify generate_wordcloud function:
def generate_wordcloud(all_comments):
    """Generate a word cloud from comments."""
    # Preprocess the entire list for word cloud
    processed_comments = preprocess_comment_for_wordcloud(all_comments)

    # Combine into a single string
    text_all = ' '.join(processed_comments)

    # Generate the word cloud
    wc_all = WordCloud(width=1000, height=500,
                       background_color='white').generate(text_all)

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc_all, interpolation='bilinear')
    # ax.set_title("☁️ Word Cloud of Video's Comments", fontsize=18)
    ax.axis('off')
    plt.tight_layout()
    plt.close(fig)  # Close the figure to prevent display
    return fig


# ------------------ Summarize positive and negative comment -----------------------#

def chunk_documents(raw_documents):
    """Split documents into chunks for better vector search."""
    document_objects = [Document(page_content=doc) for doc in raw_documents]
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        add_start_index=True
    )
    return text_processor.split_documents(document_objects)


def index_positive_documents(documents):
    """Add positive comments to vector database."""
    POSITIVE_VECTOR_DB.add_documents(chunk_documents(documents))


def index_negative_documents(documents):
    """Add negative comments to vector database."""
    NEGATIVE_VECTOR_DB.add_documents(chunk_documents(documents))


def find_related_positive(query, k=200):
    """Find related positive comments for a query."""
    return POSITIVE_VECTOR_DB.similarity_search(query, k=k)


def find_related_negative(query, k=200):
    """Find related negative comments for a query."""
    return NEGATIVE_VECTOR_DB.similarity_search(query, k=k)


def generate_positive_summary_from_vector(query="Summarize main point of these comments."):
    PROMPT_TEMPLATE_POSITIVE = """
    You are a YouTube sentiment analysis assistant. Your task is to summarize YouTube video comments.
    Below are the positive comments:
    ---
    {positive_comments}
    ---
    Please summarize the main points expressed in these positive comments.
    Return only a bullet-point list of the main takeaways with the layout of 1 line break for each point.
    Start the summary with bullet points right away, and do not include any other text.
    Note that just include 2-3 main points.
    """
    docs = find_related_positive(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_POSITIVE)
    chain = prompt | LANGUAGE_MODEL
    return chain.invoke({
        "user_query": query,
        "positive_comments": context
    })


def generate_negative_summary_from_vector(query="Summarize main point of the negative comments."):
    PROMPT_TEMPLATE_NEGATIVE = """
    You are a YouTube sentiment analysis assistant. Your task is to analyze and summarize YouTube video comments.
    Below are the negative comments:
    ---
    {negative_comments}
    ---
    Please summarize the main points expressed in these negative comments
    Return only a bullet-point list of the main takeaways with the layout of 1 line break for each point.
    Start the summary with bullet points right away, and do not include any other text.
    Note that just include 2-3 main points.
    """
    docs = find_related_negative(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_NEGATIVE)
    chain = prompt | LANGUAGE_MODEL
    return chain.invoke({
        "user_query": query,
        "negative_comments": context
    })

def reset_sentiment_databases():
    """Reset the in-memory vector stores used for sentiment analysis."""
    global POSITIVE_VECTOR_DB, NEGATIVE_VECTOR_DB, EMBEDDING_MODEL

    print("Resetting sentiment analysis databases...")
    # Recreate the vector stores to clear them
    POSITIVE_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
    NEGATIVE_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

def summarize_both_sentiments(positive_comments, negative_comments, output_file="sentiment_summary.txt"):
    # Reset the vector stores before adding new documents
    reset_sentiment_databases()

    # Index each into their own vector store
    index_positive_documents(positive_comments)
    index_negative_documents(negative_comments)

    # Summarize each
    pos_summary = generate_positive_summary_from_vector()
    neg_summary = generate_negative_summary_from_vector()

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")
        f.write(pos_summary if isinstance(
            pos_summary, str) else str(pos_summary))
        f.write("\n\n")
        f.write(neg_summary if isinstance(
            neg_summary, str) else str(neg_summary))

    return pos_summary, neg_summary


# ---------------------------------- MAIN FUNCTION ----------------------------------#

def analyze_youtube_comments(youtube_url, api_key="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0"):
    """
    Main function to analyze YouTube comments from a URL.

    Args:
        youtube_url: URL or video ID of the YouTube video
        api_key: YouTube API key (uses default if not provided)

    Returns:
        Dictionary with summaries and analysis results
    """
    print(f"Analyzing comments for: {youtube_url}")

    # Extract video ID if full URL is provided
    video_id = extract_video_id(youtube_url)
    print(f"Extracted video ID: {video_id}")

    # Step 1: Get comments from YouTube API
    print("Fetching comments from YouTube...")
    comments = get_comments(video_id, api_key)

    # Step 2: Save comments to Chroma vector database (reuse if same video)
    print("Saving comments to vector database...")
    comment_count = save_comments_to_chroma(comments, video_id)

    # Step 3: Read comments from Chroma
    raw_comments = read_comments_from_chroma()

    # Step 4: Generate overall comment summary
    print("Generating overall comment summary...")
    overall_summary = generate_comment_summary()

    # Step 5: Preprocess comments for sentiment analysis
    print("Analyzing sentiment...")
    processed_comments = preprocess_for_sentiment(raw_comments)

    # Step 6: Perform sentiment analysis
    sentiment_results, positive_comments, negative_comments, neutral_comments = analyze_sentiment(
        processed_comments)

    # Step 7: Create sentiment visualization
    print("Creating visualizations...")
    sentiment_chart = plot_sentiment_pie_chart(sentiment_results)
    sentiment_chart.savefig("sentiment_pie_chart.png")

    # Step 8: Generate word cloud
    wordcloud = generate_wordcloud(processed_comments)
    wordcloud.savefig("comment_wordcloud.png")

    # Step 9: Summarize positive and negative comments
    print("Generating sentiment-specific summaries...")
    pos_summary, neg_summary = summarize_both_sentiments(
        positive_comments, negative_comments)

    # Return results
    results = {
        "video_id": video_id,
        "comment_count": comment_count,
        "overall_summary": overall_summary,
        "sentiment_counts": {
            "positive": sentiment_results[1],
            "negative": sentiment_results[2],
            "neutral": sentiment_results[0]
        },
        "positive_summary": pos_summary,
        "negative_summary": neg_summary,
        "output_files": {
            "sentiment_chart": "sentiment_pie_chart.png",
            "wordcloud": "comment_wordcloud.png",
            "overall_summary": "overall_summary.txt",
            "sentiment_summary": "sentiment_summary.txt"
        }
    }

    print("\nAnalysis complete! Results saved to output files.")
    return results


# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube Comment Analysis Tool")
    parser.add_argument(
        "youtube_url", help="YouTube URL or video ID to analyze")
    parser.add_argument("--api-key", default="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0",
                        help="YouTube API key (optional)")
    parser.add_argument("--question", help="Ask a specific question about the comments")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of comments to retrieve for context (default: auto-calculated)")
    parser.add_argument("--reuse-db", action="store_true",
                        help="Force reuse of existing database without confirmation")

    args = parser.parse_args()

    # Extract video ID
    video_id = extract_video_id(args.youtube_url)

    # Check if we need to run the analysis
    run_analysis = True
    if CURRENT_VIDEO_ID == video_id and os.path.exists(CHROMA_PATH) and args.reuse_db:
        print(f"Using existing analysis for video ID: {video_id}")
        run_analysis = False

    # Check if a question was asked
    if args.question:
        # First make sure we've analyzed the comments
        if run_analysis:
            print(f"Analyzing comments for: {args.youtube_url}")
            analyze_youtube_comments(args.youtube_url, args.api_key)

        # Then answer the question
        print(f"\nQuestion: {args.question}")
        print("\nSearching for answer...")
        answer = answer_question(args.question, k=args.k)
        print("\nAnswer:")
        print(answer)
    else:
        # Regular analysis
        if run_analysis:
            results = analyze_youtube_comments(args.youtube_url, args.api_key)

            # Print a summary of results
            print("\n===== ANALYSIS RESULTS =====")
            print(f"Video ID: {results['video_id']}")
            print(f"Total comments analyzed: {results['comment_count']}")
            print(f"Sentiment distribution: {results['sentiment_counts']['positive']} positive, "
                  f"{results['sentiment_counts']['negative']} negative, "
                  f"{results['sentiment_counts']['neutral']} neutral")
            print("Output files:")
            for name, path in results['output_files'].items():
                print(f"- {name}: {path}")
        else:
            print("To perform a new analysis, run without the --reuse-db flag.")
            print("To ask a question about the existing analysis, use the --question parameter.")
