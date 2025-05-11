def save_comments_to_chroma(comments, video_id):
    """
    Populate comments into Chroma database, using a separate database for each video ID.
    
    Args:
        comments: List of comment dictionaries
        video_id: YouTube video ID
    
    Returns:
        Number of comments saved to the database
    """
    global CURRENT_VIDEO_ID, CHROMA_PATH
    
    # Use a video-specific path to avoid conflicts
    video_db_path = f"chroma_{video_id}"
    
    # Check if we already have a database for this video
    if CURRENT_VIDEO_ID == video_id and os.path.exists(video_db_path):
        print(f"Using existing Chroma database for video ID: {video_id}")
        CHROMA_PATH = video_db_path  # Ensure global path is set correctly
        return len(comments)
    
    # Close any open connections before working with a new database
    close_chroma_connection()
    
    # Update the global path to the video-specific path
    CHROMA_PATH = video_db_path
    
    # Prepare the Chroma database
    print(f"Creating new Chroma vector database for video ID: {video_id}")
    
    # Create the directory if it doesn't exist
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
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
        batch = documents[i:i+batch_size]
        db.add_documents(batch)
        print(f"Added batch of {len(batch)} comments to Chroma (total {i+len(batch)})")
    
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

def close_chroma_connection():
    """Close any open connections to the Chroma database."""
    # Force garbage collection to release file handles
    import gc
    gc.collect()
    time.sleep(1)  # Give a moment for resources to be released
