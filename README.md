<h1>Words Counter</h1>
Real-time **word occurrence counting** in a video using **OpenAI Whisper**, **MoviePy**, and **OpenCV**.

This program automatically:
- Transcribes the audio of a French video
- Finds and timestamps occurrences of specific target words
- Overlays live counters on the video showing how many times each word has been spoken so far

<h2>Demo</h2>

<table align="center">
<tr>
  <th>Title</th>
  <th>Description</th>
</tr>
<tr>
  <td>Word Counter Overlay</td>
  <td>  

https://github.com/user-attachments/assets/0015a3c2-e854-45e8-9d64-c04436bb3d9c
 
  </td>
</tr>
</table>

---

## What it does

- Uses **Whisper** to transcribe a French video into text segments
- Normalizes text (removes accents, converts to lowercase)
- Finds target words (e.g., `"genre"`, `"littéralement"`) and computes precise timestamps for each occurrence
- Plays through the video with **MoviePy**
- At the right moments, increments on-screen counters drawn with **OpenCV**
- Saves the annotated video with audio preserved

---

## How it works (pipeline)

1. **Transcription**  
   - Loads a Whisper ASR model (`large`)  
   - Transcribes the entire video into segments with `start` / `end` times

2. **Text normalization & tokenization**  
   - Removes accents (e.g., `"é"` → `"e"`)
   - Lowercases everything
   - Splits into word tokens (supports apostrophes like `l'amour`)

3. **Word timestamp calculation**  
   - For each target word, finds all matches in each segment  
   - Computes an exact timestamp for each match by proportionally locating it in the segment duration

4. **Overlay counters**  
   - Loads the video with MoviePy
   - For each frame, checks if the current time passes a stored timestamp for any word
   - Increments the corresponding counter and draws `"WORD said: N times"` in colored text

5. **Output**  
   - Writes the final video with counters overlaid and original audio intact

---

## Repository structure

