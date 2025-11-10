from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import tempfile
from datetime import datetime
import base64
import threading
import time

# Import your existing deception detection classes
# (Assuming they're in the same file or imported)
from finalc2 import (
    CameraRecorder,
    EnhancedDeceptionDetector,
    TRANSFORMERS_AVAILABLE,
    LIBROSA_AVAILABLE,
    MEDIAPIPE_AVAILABLE,
    PYAUDIO_AVAILABLE,
    SPEECH_RECOGNITION_AVAILABLE,
    WHISPER_DIRECT_AVAILABLE
)

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global instances
detector = EnhancedDeceptionDetector()
camera_recorder = CameraRecorder()
recording_state = {
    'is_recording': False,
    'video_path': None,
    'audio_path': None
}


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status and available features"""
    return jsonify({
        'status': 'online',
        'features': {
            'facial_analysis': MEDIAPIPE_AVAILABLE,
            'audio_analysis': LIBROSA_AVAILABLE,
            'ai_models': TRANSFORMERS_AVAILABLE,
            'audio_recording': PYAUDIO_AVAILABLE,
            'speech_to_text': SPEECH_RECOGNITION_AVAILABLE,
            'whisper_enhanced': WHISPER_DIRECT_AVAILABLE
        },
        'recording': recording_state['is_recording']
    })


@app.route('/api/start-recording', methods=['POST'])
def start_recording():
    """Start video recording"""
    global camera_recorder, recording_state
    
    if recording_state['is_recording']:
        return jsonify({'error': 'Already recording'}), 400
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"recording_{timestamp}.mp4")
        
        if camera_recorder.start_recording(output_path):
            recording_state['is_recording'] = True
            recording_state['video_path'] = output_path
            return jsonify({
                'success': True,
                'message': 'Recording started'
            })
        else:
            return jsonify({'error': 'Failed to start recording'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop-recording', methods=['POST'])
def stop_recording():
    """Stop video recording"""
    global camera_recorder, recording_state
    
    if not recording_state['is_recording']:
        return jsonify({'error': 'Not recording'}), 400
    
    try:
        video_path, audio_path = camera_recorder.stop_recording()
        recording_state['is_recording'] = False
        recording_state['video_path'] = video_path
        recording_state['audio_path'] = audio_path
        
        return jsonify({
            'success': True,
            'video_path': os.path.basename(video_path) if video_path else None,
            'audio_path': os.path.basename(audio_path) if audio_path else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Upload video file"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        recording_state['video_path'] = filepath
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': filepath
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    """Upload audio file"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        recording_state['audio_path'] = filepath
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': filepath
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio to text with disfluencies"""
    audio_path = recording_state.get('audio_path')
    
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({'error': 'No audio file available'}), 400
    
    try:
        text, disfluency_features = detector.speech_converter.convert_audio_to_text_with_fillers(audio_path)
        
        return jsonify({
            'success': True,
            'text': text,
            'disfluency_features': disfluency_features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_deception():
    """Run deception detection analysis"""
    data = request.get_json()
    text = data.get('text', '')
    
    video_path = recording_state.get('video_path')
    audio_path = recording_state.get('audio_path')
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'No video file available'}), 400
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = detector.detect_deception(video_path, audio_path, text)
        
        # Convert numpy types to native Python types for JSON serialization
        result_clean = json.loads(json.dumps(result, default=str))
        
        return jsonify({
            'success': True,
            'result': result_clean
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-report', methods=['POST'])
def download_report():
    """Generate and download analysis report"""
    data = request.get_json()
    result = data.get('result')
    
    if not result:
        return jsonify({'error': 'No analysis result provided'}), 400
    
    try:
        # Generate report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], f"report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("ENHANCED DECEPTION DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Deception Probability: {result['deception_probability']:.1%}\n")
            f.write(f"Deception Detected: {result['deception_detected']}\n")
            f.write(f"Confidence: {result['confidence']:.1%}\n\n")
            
            f.write("MODALITY SCORES:\n")
            for modality, score in result['modality_scores'].items():
                f.write(f"  {modality.capitalize()}: {score:.1%}\n")
            
            if result.get('key_indicators'):
                f.write("\nDETECTED INDICATORS:\n")
                for indicator in result['key_indicators']:
                    f.write(f"  • {indicator}\n")
            
            f.write(f"\nTRANSCRIBED TEXT:\n{result.get('transcribed_text', '')}\n")
        
        return send_file(report_path, as_attachment=True, download_name=f"report_{timestamp}.txt")
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("ENHANCED DECEPTION DETECTION SERVER")
    print("=" * 70)
    print("Server starting on http://localhost:5000")
    print("Open your browser and navigate to http://localhost:5000")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
