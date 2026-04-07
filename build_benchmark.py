import os
import json
import argparse
import whisperx
from google import genai
import torch
from moviepy import VideoFileClip

# --- CONFIGURATION ---
# Replace with your key
GEMINI_API_KEY = ""

# Configure Gemini
client = genai.Client(api_key=GEMINI_API_KEY)
# We use Gemini 3 Pro for its advanced reasoning capabilities
MODEL_ID = 'gemini-3-pro-preview' 

def extract_audio(video_path, audio_path):
    """Step 1: Extract audio from video file."""
    print(f"🔊 Extracting audio from {video_path}...")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    print("Audio extracted.")

def get_aligned_transcript(audio_path, language="en"):
    """Step 2: Transcribe and Align timestamps using WhisperX."""
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        compute_type = "float32" # MPS often requires float32 for some ops
    elif torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    batch_size = 16 
    
    print(f"Transcribing with WhisperX on {device} ({compute_type})...")
    print(f"   Using language: {language}")
    # 1. Transcribe with forced language
    model_ws = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model_ws.transcribe(audio, batch_size=batch_size, language=language)
    
    # Use the forced language for alignment
    detected_lang = result.get("language", language)
    print(f"   Detected language: {detected_lang}, using: {language}")
    
    # 2. Align (This fixes the timestamp lag)
    print("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Clean up GPU memory
    del model_ws, model_a
    
    return result["segments"]

def segment_events_with_llm(transcript_segments):
    """Step 3: Use Gemini to process transcript into Clinical Events."""
    print("Sending to Gemini for segmentation...")
    
    # Create a simplified string for the LLM to read
    transcript_text = ""
    for seg in transcript_segments:
        start = f"{seg['start']:.2f}"
        end = f"{seg['end']:.2f}"
        text = seg['text'].strip()
        transcript_text += f"[{start}-{end}] {text}\n"

    # The Prompt for Clinical Segmentation
    prompt = f"""
        You are an expert Ultrasound Instructor creating a **verbatim-level clinical script** for training AI models.

        Your goal is NOT to summarize. Your goal is to **preserve every clinical detail** from the transcript, converting conversational speech into precise, step-by-step imperative instructions.

        ### 1. Segmentation Strategy (The "Micro-Event" Rule)
        - **Do NOT group long sequences.** If the instructor performs multiple distinct actions (e.g., "Adjust depth" THEN "Sweep down"), split them into separate events or keep the event highly detailed.
        - **Capture the "Flow":** Retain the transitional steps (e.g., "Scanning through bowel gas," "Re-sweeping to confirm").

        ### 2. Field Generation Rules (Maximize Detail)

        **field: 'action' (The Explicit "How-To")**
        Convert the instructor's narration into direct commands. **You must retain:**
        - **Exact Probe Maneuvers:** "Sweep," "Rock," "Fan," "Slide," "Apply Pressure."
        - **Knobology Details:** for example "Decrease depth" "Move focus to posterior wall," "Adjust TGC for uniform gain."
        - **Patient Instructions:** "Big breath in," "Hold," "Breathe normally," "Relax."

        **field: 'interpretation' (The Visual Commentary)**
        Describe exactly what the instructor claims to see. **Do not hallucinate features not mentioned.**
        - **Visual Adjectives:** Use the exact descriptors from the text (e.g., "strong tubular structure," "hyperechoic," "tuning fork appearance").
        - **Reasoning:** Explain *why* they are looking there (e.g., "Scanning until it tapers to find the bifurcation").
        - **Negative Findings:** (e.g., "IVC is NOT pulsatile," "Avoid reverb artifacts").

        ### 3. Strict Constraints
        - **NO SUMMARIZATION:** Do not condense "Sweep down and adjust gain and check for gas" into "Scan aorta." Keep all three steps.
        - **Tone:** Professional, Imperative, Instructive.
        - **Timestamps:** Start exactly when the specific instruction begins.

        ### Input Transcript:
        {transcript_text}

        ### Output Format (JSON Only):
        [
        {{
            "start": 0.0,
            "end": 8.5,
            "action": "Place probe in transverse. Immediately decrease depth to isolate the superior aorta. Instruct patient: 'Big breath in'.",
            "interpretation": "Visualize the superior aorta. Verify that depth is shallow enough to maximize resolution but deep enough to see the posterior border."
        }},

        ...
        ]
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    
    # Extract JSON from response (handling potential markdown fences)
    try:
        json_str = response.text.replace("```json", "").replace("```", "").strip()
        events = json.loads(json_str)
        return events
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print("Raw response:", response.text)
        return []

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build benchmark from ultrasound video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--audio", type=str, default="temp_audio.wav", help="Path for temporary audio file")
    parser.add_argument("--language", type=str, default="en", help="Force language code (default: en for English)")
    args = parser.parse_args()
    
    VIDEO_PATH = args.video
    AUDIO_PATH = args.audio
    OUTPUT_GT_PATH = args.output
    LANGUAGE = args.language
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_GT_PATH), exist_ok=True)
    
    # 1. Extract Audio
    if not os.path.exists(AUDIO_PATH):
        extract_audio(VIDEO_PATH, AUDIO_PATH)
    
    # 2. Get Raw Aligned Transcript
    raw_segments = get_aligned_transcript(AUDIO_PATH, language=LANGUAGE)
    
    # 3. Process into Ground Truth with Gemini
    clinical_events = segment_events_with_llm(raw_segments)
    
    # 4. Save Final Benchmark File
    with open(OUTPUT_GT_PATH, "w") as f:
        json.dump(clinical_events, f, indent=2)
        
    print(f"Success! Ground truth saved to {OUTPUT_GT_PATH}")
    print(f"Generated {len(clinical_events)} clinical events.")