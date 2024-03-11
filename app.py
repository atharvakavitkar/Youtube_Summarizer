import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def extract_transcript(youtube_video_url: str):
    """
    Extracts the transcript text from a YouTube video URL.

    Args:
        youtube_video_url (str): The URL of the YouTube video.

    Returns:
        str: The extracted transcript text.
    """
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_raw = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ""
        for segment in transcript_raw:
            transcript_text += segment["text"] + " "
        
        return transcript_text
    except Exception as e:
        raise e

def generate_summary(transcript: str):
    """
    Generates a summary of the given transcript text.

    Args:
        transcript (str): The input transcript text.

    Returns:
        str: The generated summary.
    """
    prompt = f'summarise: {transcript}'

    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=False)
    summary_ids = model.generate(input_ids,
                                 num_beams=4,
                                 no_repeat_ngram_size=3,
                                 min_length=100,
                                 max_length=400,
                                 early_stopping=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    st.title("YouTube Video Summarizer")
    youtube_link = st.text_input("Enter YouTube Video Link:")
    
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
        transcript = extract_transcript(youtube_link)
        summary = generate_summary(transcript)
        st.markdown("## Detailed Notes:")
        st.write(summary)