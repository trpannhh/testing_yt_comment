import os
import shutil
import time
import json
from pathlib import Path

# These functions should be added/modified in your youtube_summary_tool.py file

def get_video_directory(video_id):
    """
    Get the directory path for a specific video's database.
    This approach stores each video in its own subdirectory.
    """
    base_dir = "chroma_db"
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    
    return os.path.join(base_dir, f"video_{video_id}")

def close_chroma_connection():
    """Close any open connections to the Chroma database."""
    # Force garbage collection to release file handles
    import gc
    gc.collect()
    time.sleep(1)  # Give a moment for resources to be released

def save_comments_to_chroma(comments, video_id):
    """
    Populate comments into Chroma database, using a separate directory for each video.
    
    Args:
        comments: List of comment dictionaries
        video_id: YouTube video ID to create or reuse a database
        
    Returns:
        Number of comments saved to the database
    """
    global CURRENT_VIDEO_ID
    
    # Get the directory for this specific video
    video_dir = get_video_directory(video_id)
    
    # Check if we already have a database for this video
    if os.path.exists(video_dir):
        print(f"Using existing Chroma database for video ID: {video_id}")
        CURRENT_VIDEO_ID = video_id
        
        # Verify the database by checking metadata
        metadata_path = os.path.join(video_dir, "video_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if metadata.get('comment_count', 0) > 0:
                        # If valid metadata exists, return the comment count
                        return metadata.get('comment_count', len(comments))
            except Exception as e:
                print(f"Error reading metadata: {e}")
                # If metadata is corrupt, we'll recreate the database
                print("Metadata appears corrupt, recreating database.")
    
    # We need to create a new database
    
    # First, close any existing connections
    close_chroma_connection()
    
    # Create the directory if needed
    os.makedirs(video_dir, exist_ok=True)
    
    # Special handling for macOS - make sure directory is empty
    if os.path.exists(video_dir) and os.path.isdir(video_dir):
        try:
            # List all files in the directory
            for item in os.listdir(video_dir):
                item_path = os.path.join(video_dir, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)  # Remove files
                elif os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)  # Remove directories
                    except Exception as e:
                        print(f"Warning: Could not remove directory {item_path}: {e}")
        except Exception as e:
            print(f"Warning: Error while cleaning directory: {e}")
            
            # If we can't clean the directory, create a new one with timestamp
            timestamp = int(time.time())
            video_dir = f"{video_dir}_{timestamp}"
            os.makedirs(video_dir, exist_ok=True)
    
    # Prepare the Chroma database
    print(f"Creating new Chroma vector database for video ID: {video_id}")
    from langchain_chroma import Chroma
    
    # Create the database with the specific video directory
    db = Chroma(persist_directory=video_dir,
                embedding_function=get_embedding_function())

    # Create Document objects for each comment
    from langchain.schema.document import Document
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
    with open(os.path.join(video_dir, "video_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "comment_count": len(documents),
            "created_at": int(time.time())
        }, f, ensure_ascii=False, indent=2)

    print(f"Successfully added all {len(documents)} comments to Chroma database.")
    return len(documents)

def analyze_youtube_comments(youtube_url, api_key="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0"):
    """
    Main function to analyze YouTube comments from a URL.
    Modified to handle database connections better.
    """
    print(f"Analyzing comments for: {youtube_url}")

    # Extract video ID if full URL is provided
    video_id = extract_video_id(youtube_url)
    print(f"Extracted video ID: {video_id}")
    
    if not video_id:
        raise ValueError("Invalid YouTube URL or video ID")

    # Step 1: Get comments from YouTube API
    print("Fetching comments from YouTube...")
    comments = get_comments(video_id, api_key)

    # Step 2: Save comments to Chroma vector database (reuse if same video)
    print("Saving comments to vector database...")
    video_dir = get_video_directory(video_id)
    
    # Save comments to the video-specific database
    comment_count = save_comments_to_chroma(comments, video_id)

    # Step 3: Read comments from Chroma - updated to use the video directory
    print("Reading comments from vector database...")
    raw_comments = read_comments_from_chroma(video_dir)

    # Step 4: Generate overall comment summary
    print("Generating overall comment summary...")
    overall_summary = generate_comment_summary(video_dir)

    # Step 5: Preprocess comments for sentiment analysis
    print("Analyzing sentiment...")
    processed_comments = preprocess_for_sentiment(raw_comments)

    # Step 6: Perform sentiment analysis
    sentiment_results, positive_comments, negative_comments, neutral_comments = analyze_sentiment(
        processed_comments)

    # Step 7: Create sentiment visualization
    print("Creating visualizations...")
    sentiment_chart = plot_sentiment_pie_chart(sentiment_results)
    chart_path = f"sentiment_pie_chart_{video_id}.png"
    sentiment_chart.savefig(chart_path)

    # Step 8: Generate word cloud
    wordcloud = generate_wordcloud(processed_comments)
    wordcloud_path = f"comment_wordcloud_{video_id}.png"
    wordcloud.savefig(wordcloud_path)

    # Step 9: Summarize positive and negative comments
    print("Generating sentiment-specific summaries...")
    summary_path = f"sentiment_summary_{video_id}.txt"
    pos_summary, neg_summary = summarize_both_sentiments(
        positive_comments, negative_comments, output_file=summary_path)
    
    # Save overall summary to file with video ID in filename
    overall_summary_path = f"overall_summary_{video_id}.txt"
    with open(overall_summary_path, "w", encoding="utf-8") as f:
        f.write(overall_summary)

    # Return results with updated file paths
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
            "sentiment_chart": chart_path,
            "wordcloud": wordcloud_path,
            "overall_summary": overall_summary_path,
            "sentiment_summary": summary_path
        }
    }

    print("\nAnalysis complete! Results saved to output files.")
    return results

def read_comments_from_chroma(video_dir):
    """
    Read comments from the Chroma database in the specified video directory.
    """
    # Connect to the existing Chroma database
    from langchain_chroma import Chroma
    
    db = Chroma(persist_directory=video_dir,
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

def generate_comment_summary(video_dir):
    """Generate a general summary of all comments with improved diversity."""
    # Load the Chroma vector store for the specific video
    from langchain_chroma import Chroma
    from langchain.schema.document import Document
    
    db = Chroma(persist_directory=video_dir,
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
    from langchain.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text)

    # Use OllamaLLM model
    print("Generating summary with language model...")
    model = OllamaLLM(model="llama3")
    response_text = model.invoke(prompt)

    return response_text

def answer_question(question, k=None, video_id=None):
    """
    Answer a question based on the YouTube comments data.
    
    Args:
        question: The user's question about the video comments
        k: Number of relevant comments to retrieve for context (auto-calculated if None)
        video_id: Specific video ID to query (uses CURRENT_VIDEO_ID if None)
        
    Returns:
        Dictionary with answer and metadata
    """
    # Start timing
    start_time = time.time()
    
    global CURRENT_VIDEO_ID
    
    # Use specified video ID or current one
    target_video_id = video_id or CURRENT_VIDEO_ID
    
    if not target_video_id:
        return {
            'error': 'No video has been analyzed yet. Please analyze a YouTube video first.'
        }
    
    # Get directory for this video
    video_dir = get_video_directory(target_video_id)
    
    if not os.path.exists(video_dir):
        return {
            'error': f'No database found for video ID: {target_video_id}. Please analyze this video first.'
        }

    # Load the Chroma vector store
    from langchain_chroma import Chroma
    
    db = Chroma(persist_directory=video_dir,
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
    from langchain.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM
    
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
        'processing_time': f"{total_time:.2f} seconds",
        'video_id': target_video_id
    }
