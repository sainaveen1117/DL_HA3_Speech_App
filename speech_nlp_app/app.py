from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import io
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
from flask import Flask, request, jsonify
import soundfile as sf
import torch

app = Flask(__name__)

#speech recognition model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load NLP models
translator_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
translator_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            # Read the file using pydub
            audio = AudioSegment.from_file(file.stream)
            # Convert to mono and resample to 16kHz
            audio = audio.set_frame_rate(16000).set_channels(1)
            # Get raw audio data as bytes
            audio_bytes = io.BytesIO()
            audio.export(audio_bytes, format='wav')
            audio_bytes.seek(0)
            audio_np, sample_rate = sf.read(audio_bytes)
            
            # Convert to required format for the model
            input_values = processor(audio_np, return_tensors="pt", sampling_rate=sample_rate).input_values
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
            return jsonify({"transcription": transcription})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    text = request.json.get('text', '')
    if not text:
        return jsonify({"error": "No text provided for translation"}), 400

    try:
        translator_tokenizer.src_lang = "en_XX"
        encoded_text = translator_tokenizer(text, return_tensors="pt")
        translated_tokens = translator_model.generate(**encoded_text, forced_bos_token_id=translator_tokenizer.lang_code_to_id["ja_XX"])
        translated_text = translator_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/summarize', methods=['POST'])
def summarize_text():
    text = request.json.get('text', '')
    try:
        result = summarization_pipeline(text, max_length=100, min_length=30, do_sample=False)
        if 'summary_text' in result[0]:
            summary = result[0]['summary_text']
        else:
            summary = 'Failed to find summary text in the response.'
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/question', methods=['POST'])
def answer_question():
    question = request.json.get('question', '')
    context = request.json.get('context', '')
    answer = qa_pipeline(question=question, context=context)['answer']
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
