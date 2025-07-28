from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image
import os
import time

app = FastAPI(title="성형외과용 미세표정 분석 시스템 - Cloud Edition")

# 클라우드 환경에서는 MediaPipe 사용 (dlib보다 가볍고 안정적)
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    USE_MEDIAPIPE = True
    print("✅ MediaPipe 로드 성공 - 클라우드 최적화 모드")
except ImportError:
    USE_MEDIAPIPE = False
    print("❌ MediaPipe 없음 - 기본 모드로 실행")

# 전역 변수
captured_images = {}
last_frame_data = {"frame": None, "landmarks": None}

def analyze_face_asymmetry_mediapipe(landmarks, image_width, image_height):
    """MediaPipe 랜드마크를 사용한 비대칭 분석"""
    if not landmarks or len(landmarks) < 468:
        return None
    
    # MediaPipe 주요 랜드마크 인덱스
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    NOSE_TIP = 1
    CHIN = 18
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    
    # 픽셀 좌표로 변환
    def to_pixel(landmark):
        return [int(landmark.x * image_width), int(landmark.y * image_height)]
    
    left_eye = to_pixel(landmarks[LEFT_EYE_OUTER])
    right_eye = to_pixel(landmarks[RIGHT_EYE_OUTER])
    nose_tip = to_pixel(landmarks[NOSE_TIP])
    chin = to_pixel(landmarks[CHIN])
    left_mouth = to_pixel(landmarks[LEFT_MOUTH])
    right_mouth = to_pixel(landmarks[RIGHT_MOUTH])
    
    # 비대칭 계산
    eye_diff = left_eye[1] - right_eye[1]
    mouth_diff = left_mouth[1] - right_mouth[1]
    
    # 얼굴 중심 계산
    face_center_x = (left_eye[0] + right_eye[0]) / 2
    nose_center_offset = nose_tip[0] - face_center_x
    
    # 얼굴 크기 및 위치 계산
    face_width = abs(right_eye[0] - left_eye[0])
    screen_center_x = image_width / 2
    face_center_y = (left_eye[1] + right_eye[1]) / 2
    
    position_offset_x = abs(face_center_x - screen_center_x)
    
    # 거리 피드백
    if face_width < 120:
        distance_feedback = "얼굴을 카메라에 더 가까이 가져오세요"
    elif face_width > 200:
        distance_feedback = "얼굴을 카메라에서 조금 멀리 하세요"
    else:
        distance_feedback = "적정 거리입니다"
    
    # 캡처 준비 상태
    center_alignment_score = abs(nose_center_offset)
    face_stability = abs(eye_diff) + abs(mouth_diff)
    
    capture_ready = bool(
        120 <= face_width <= 200 and
        center_alignment_score < 15 and
        position_offset_x < 60 and
        face_stability < 25
    )
    
    # 비대칭 값 계산
    eye_asymmetry_val = abs(eye_diff)
    mouth_asymmetry_val = abs(mouth_diff)
    nose_asymmetry_val = abs(nose_center_offset)
    
    total_asym_value = eye_asymmetry_val + mouth_asymmetry_val + nose_asymmetry_val
    total_score = min(100, total_asym_value * 3)
    
    # 방향 계산 (영상 반전 고려)
    eye_direction = "오른쪽 높음" if eye_diff < 0 else ("왼쪽 높음" if eye_diff > 0 else "대칭")
    mouth_direction = "오른쪽 높음" if mouth_diff < 0 else ("왼쪽 높음" if mouth_diff > 0 else "대칭")
    nose_direction = "왼쪽 치우침" if nose_center_offset > 0 else ("오른쪽 치우침" if nose_center_offset < 0 else "대칭")
    
    # 주요 랜드마크 좌표 (화면 표시용)
    key_landmarks = [
        left_eye, right_eye, nose_tip, chin, left_mouth, right_mouth
    ]
    
    return {
        "eye_diff": round(float(-eye_diff), 1),
        "mouth_diff": round(float(-mouth_diff), 1),
        "nose_offset": round(float(-nose_center_offset), 1),
        "eye_asymmetry_val": round(float(eye_asymmetry_val), 1),
        "mouth_asymmetry_val": round(float(mouth_asymmetry_val), 1),
        "nose_asymmetry_val": round(float(nose_asymmetry_val), 1),
        "eye_direction": eye_direction,
        "mouth_direction": mouth_direction,
        "nose_direction": nose_direction,
        "total_score": round(float(total_score), 1),
        "assessment": get_asymmetry_assessment(total_score),
        "landmarks_coords": key_landmarks,
        "capture_ready": capture_ready,
        "center_alignment_score": round(float(center_alignment_score), 1),
        "distance_feedback": distance_feedback,
        "face_width": float(face_width),
        "face_stability": round(float(face_stability), 1),
        "accurate_center_x": float(face_center_x)
    }

def get_asymmetry_assessment(score):
    if score < 5:
        return "매우 우수 (Very Symmetrical)"
    elif score < 10:
        return "우수 (Good Symmetry)"
    elif score < 20:
        return "보통 (Slight Asymmetry)"
    elif score < 30:
        return "주의 (Moderate Asymmetry)"
    else:
        return "개선 필요 (Significant Asymmetry)"

def create_symmetry_images_simple(frame, face_center_x):
    """간단한 대칭 이미지 생성 (클라우드 최적화)"""
    h, w, c = frame.shape
    center_x = int(face_center_x)
    
    # 원본 이미지 (좌우반전 해제)
    original = cv2.flip(frame, 1)
    
    # 왼쪽 대칭 이미지
    left_symmetric = frame.copy()
    for x in range(center_x, min(w, center_x + 100)):
        mirror_x = 2 * center_x - x
        if 0 <= mirror_x < center_x:
            left_symmetric[:, x] = left_symmetric[:, mirror_x]
    left_symmetric = cv2.flip(left_symmetric, 1)
    
    # 오른쪽 대칭 이미지
    right_symmetric = frame.copy()
    for x in range(max(0, center_x - 100), center_x):
        mirror_x = 2 * center_x - x
        if center_x <= mirror_x < w:
            right_symmetric[:, x] = right_symmetric[:, mirror_x]
    right_symmetric = cv2.flip(right_symmetric, 1)
    
    # 이미지를 base64로 인코딩
    def encode_image(img):
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buffer).decode('utf-8')
    
    return {
        "original_image": encode_image(original),
        "left_symmetric_image": encode_image(left_symmetric),
        "right_symmetric_image": encode_image(right_symmetric)
    }

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>성형외과용 미세표정 분석 시스템 - Cloud Edition</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: #fff; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 10px;
                min-height: 100vh;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                text-align: center; 
                background: rgba(255,255,255,0.1); 
                border-radius: 20px; 
                padding: 20px; 
                box-shadow: 0 15px 40px rgba(0,0,0,0.4);
            }
            h1 { 
                font-size: 2.2em; 
                margin-bottom: 10px; 
                color: #FFD700; 
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }
            .cloud-badge {
                background: linear-gradient(45deg, #00ff00, #00cc00);
                color: #000;
                padding: 8px 20px;
                border-radius: 25px;
                font-size: 0.9em;
                font-weight: bold;
                margin: 10px 0;
                display: inline-block;
                box-shadow: 0 4px 15px rgba(0,255,0,0.3);
            }
            .content-wrapper { 
                display: flex; 
                flex-wrap: wrap; 
                justify-content: center; 
                gap: 20px; 
                align-items: flex-start;
                margin-top: 20px;
            }
            .video-section { 
                flex: 1; 
                min-width: 300px; 
                max-width: 500px; 
                position: relative; 
                border-radius: 15px; 
                overflow: hidden; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.3); 
                background: #000;
            }
            #video { 
                width: 100%; 
                height: auto; 
                display: block; 
                transform: scaleX(-1);
            }
            #overlayCanvas { 
                position: absolute; 
                top: 0; 
                left: 0; 
                width: 100%; 
                height: 100%; 
                pointer-events: none; 
                transform: scaleX(-1);
            }
            .analysis-section { 
                flex: 1; 
                min-width: 300px; 
                max-width: 400px; 
                background: rgba(255,255,255,0.1); 
                border-radius: 15px; 
                padding: 20px; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }
            .score-display { margin-bottom: 20px; }
            .total-score-val { 
                font-size: 3em; 
                font-weight: bold; 
                color: #FFD700; 
                text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
            }
            .assessment-text { 
                font-size: 1.2em; 
                margin-top: 5px; 
                opacity: 0.9;
            }
            .chart-container { 
                width: 100%; 
                height: 250px; 
                margin-top: 15px;
            }
            .controls { 
                margin: 20px 0; 
                display: flex; 
                justify-content: center; 
                gap: 15px;
                flex-wrap: wrap;
            }
            button { 
                background: linear-gradient(45deg, #4CAF50, #45a049); 
                color: white; 
                border: none; 
                padding: 12px 25px; 
                font-size: 16px; 
                border-radius: 25px; 
                cursor: pointer; 
                transition: all 0.3s; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                font-weight: bold;
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            }
            button:disabled { 
                background: #ccc; 
                cursor: not-allowed; 
                transform: none; 
                box-shadow: none;
            }
            .capture-btn { 
                background: linear-gradient(45deg, #FF6B6B, #ff5252); 
            }
            .status-message { 
                padding: 12px; 
                margin-top: 15px; 
                border-radius: 10px; 
                font-weight: bold; 
                color: #333; 
                background: rgba(255,255,255,0.9); 
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .error-message { 
                color: #ff4d4d; 
                font-weight: bold; 
                margin-top: 10px;
                background: rgba(255,0,0,0.1);
                padding: 10px;
                border-radius: 8px;
            }
            .symmetry-section { 
                margin-top: 30px; 
                background: rgba(255,255,255,0.1); 
                border-radius: 15px; 
                padding: 20px; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }
            .symmetry-title { 
                font-size: 1.8em; 
                color: #FFD700; 
                margin-bottom: 15px; 
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
            .symmetry-images { 
                display: flex; 
                justify-content: center; 
                gap: 15px; 
                flex-wrap: wrap;
            }
            .symmetry-item { 
                text-align: center; 
                flex: 1; 
                min-width: 150px; 
                max-width: 200px;
            }
            .symmetry-item img { 
                width: 100%; 
                height: auto; 
                border-radius: 10px; 
                border: 3px solid rgba(255,255,255,0.3); 
                box-shadow: 0 4px 15px rgba(0,0,0,0.3); 
                background: rgba(0,0,0,0.2);
            }
            .symmetry-label { 
                margin-top: 8px; 
                font-size: 1em; 
                font-weight: bold; 
                color: #fff;
            }
            @media (max-width: 768px) {
                .content-wrapper { flex-direction: column; }
                h1 { font-size: 1.8em; }
                .controls { flex-direction: column; align-items: center; }
                button { width: 200px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 성형외과용 미세표정 분석 시스템</h1>
            <div class="cloud-badge">☁️ Cloud Edition - 전세계 접속 가능</div>
            <div style="font-size: 1.1em; margin-bottom: 20px; opacity: 0.9;">
                실시간 얼굴 비대칭 분석 및 평가 - MediaPipe 클라우드 최적화 버전
            </div>
            
            <div class="content-wrapper">
                <div class="video-section">
                    <video id="video" width="640" height="480" autoplay muted></video>
                    <canvas id="overlayCanvas" width="640" height="480"></canvas>
                </div>
                <div class="analysis-section">
                    <div class="score-display">
                        <div class="total-score-val" id="totalScore">--</div>
                        <div class="assessment-text" id="assessment">분석 시작 대기 중</div>
                    </div>
                    <div class="chart-container">
                        <canvas id="asymmetryChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <button id="startBtn" onclick="startAnalysis()">분석 시작</button>
                <button id="stopBtn" onclick="stopAnalysis()" disabled>분석 중지</button>
                <button id="captureBtn" onclick="manualCapture()" class="capture-btn" disabled>대칭 분석 캡처</button>
            </div>
            
            <div id="status" class="status-message">클라우드 서버 연결 대기 중...</div>
            <div id="errorMsg" class="error-message" style="display: none;"></div>
            
            <div class="symmetry-section" id="symmetrySection" style="display: none;">
                <div class="symmetry-title">🔍 얼굴 대칭 비교 분석</div>
                <div class="symmetry-images">
                    <div class="symmetry-item">
                        <img id="originalImage" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjNjY3ZWVhIiBvcGFjaXR5PSIwLjMiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZiIgZm9udC1zaXplPSIxNCI+캡처 대기 중</dGV4dD48L3N2Zz4=" alt="원본">
                        <div class="symmetry-label">원본</div>
                    </div>
                    <div class="symmetry-item">
                        <img id="leftSymmetryImage" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjNjY3ZWVhIiBvcGFjaXR5PSIwLjMiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZiIgZm9udC1zaXplPSIxNCI+캡처 대기 중</dGV4dD48L3N2Zz4=" alt="왼쪽 대칭">
                        <div class="symmetry-label">왼쪽 기준 대칭</div>
                    </div>
                    <div class="symmetry-item">
                        <img id="rightSymmetryImage" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjNjY3ZWVhIiBvcGFjaXR5PSIwLjMiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZiIgZm9udC1zaXplPSIxNCI+캡처 대기 중</dGV4dD48L3N2Zz4=" alt="오른쪽 대칭">
                        <div class="symmetry-label">오른쪽 기준 대칭</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let video = document.getElementById('video');
            let overlayCanvas = document.getElementById('overlayCanvas');
            let overlayCtx = overlayCanvas.getContext('2d');
            let ws = null;
            let isAnalyzing = false;
            let asymmetryChart;

            function initializeChart() {
                let chartCtx = document.getElementById('asymmetryChart').getContext('2d');
                if (asymmetryChart) { asymmetryChart.destroy(); }
                asymmetryChart = new Chart(chartCtx, {
                    type: 'bar',
                    data: {
                        labels: ['입꼬리', '눈', '코 중심'],
                        datasets: [{
                            label: '비대칭 점수 (px)',
                            data: [0, 0, 0],
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.8)',
                                'rgba(54, 162, 235, 0.8)',
                                'rgba(75, 192, 192, 0.8)'
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(75, 192, 192, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                min: -30,
                                max: 30,
                                ticks: { color: 'white' },
                                grid: {
                                    color: function(context) {
                                        return context.tick.value === 0 ? 'white' : 'rgba(255,255,255,0.15)';
                                    },
                                    lineWidth: function(context) { return context.tick.value === 0 ? 3 : 1; }
                                }
                            },
                            y: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.15)' } }
                        },
                        plugins: {
                            legend: { display: false },
                            title: { display: true, text: '부위별 좌우 비대칭 분석 (px)', color:'white', font:{size:16} }
                        }
                    }
                });
            }

            async function startAnalysis() {
                document.getElementById('errorMsg').style.display = 'none';
                document.getElementById('status').textContent = '웹캠 연결 중...';
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 640, height: 480 } 
                    });
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {
                        overlayCanvas.width = video.videoWidth;
                        overlayCanvas.height = video.videoHeight;
                    };
                    initializeChart();
                    
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                    
                    ws.onopen = function() {
                        document.getElementById('status').textContent = '✅ 클라우드 서버 연결 성공! 분석 시작...';
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                        document.getElementById('captureBtn').disabled = false;
                        document.getElementById('symmetrySection').style.display = 'block';
                        isAnalyzing = true;
                        sendFrames();
                    };
                    ws.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            updateResults(data);
                        } catch (e) {
                            console.error('메시지 파싱 오류:', e);
                        }
                    };
                    ws.onclose = function() {
                        document.getElementById('status').textContent = '❌ 연결 끊김. 다시 시작하려면 "분석 시작" 클릭';
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        document.getElementById('captureBtn').disabled = true;
                        isAnalyzing = false;
                        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                    };
                    ws.onerror = function(error) {
                        console.error('WebSocket 오류:', error);
                        document.getElementById('errorMsg').textContent = '클라우드 서버 연결 오류 발생.';
                        document.getElementById('errorMsg').style.display = 'block';
                        document.getElementById('status').textContent = '❌ 연결 오류!';
                    };
                } catch (err) {
                    document.getElementById('status').textContent = '❌ 웹캠 접근 오류. 브라우저에서 카메라 권한을 허용해주세요.';
                    document.getElementById('errorMsg').textContent = '웹캠 접근 오류: ' + err.message;
                    document.getElementById('errorMsg').style.display = 'block';
                    console.error('웹캠 접근 오류:', err);
                }
            }

            function stopAnalysis() {
                isAnalyzing = false;
                if (ws) { ws.close(); }
                if (video.srcObject) { 
                    video.srcObject.getTracks().forEach(track => track.stop()); 
                }
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('captureBtn').disabled = true;
                document.getElementById('status').textContent = '분석 중지됨.';
                document.getElementById('symmetrySection').style.display = 'none';
                asymmetryChart.data.datasets[0].data = [0, 0, 0];
                asymmetryChart.update();
                document.getElementById('totalScore').textContent = '--';
                document.getElementById('assessment').textContent = '분석 시작 대기 중';
                overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            }

            function manualCapture() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send("manual_capture_request");
                    document.getElementById('status').textContent = '📸 캡처 중... 잠시만 기다려주세요';
                    document.getElementById('captureBtn').disabled = true;
                    setTimeout(() => {
                        document.getElementById('captureBtn').disabled = false;
                        document.getElementById('status').textContent = '✅ 캡처 완료!';
                    }, 2000);
                }
            }

            function sendFrames() {
                if (!isAnalyzing || !ws || ws.readyState !== WebSocket.OPEN) return;
                
                let tempCanvas = document.createElement('canvas');
                let tempCtx = tempCanvas.getContext('2d');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                tempCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
                
                tempCanvas.toBlob(function(blob) {
                    if (blob && isAnalyzing && ws && ws.readyState === WebSocket.OPEN) {
                        const reader = new FileReader();
                        reader.onloadend = function() {
                            try {
                                ws.send(reader.result.split(',')[1]);
                            } catch (e) {
                                console.error('프레임 전송 오류:', e);
                            }
                        };
                        reader.readAsDataURL(blob);
                    }
                }, 'image/jpeg', 0.6);
                
                setTimeout(sendFrames, 200);
            }

            function updateResults(data) {
                if (data.error) {
                    document.getElementById('errorMsg').textContent = `❌ 분석 오류: ${data.error}`;
                    document.getElementById('errorMsg').style.display = 'block';
                    document.getElementById('status').textContent = '얼굴 감지 안됨';
                    document.getElementById('totalScore').textContent = '--';
                    document.getElementById('assessment').textContent = '얼굴 감지 안됨';
                    asymmetryChart.data.datasets[0].data = [0, 0, 0];
                    asymmetryChart.update();
                    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                    return;
                }
                
                document.getElementById('errorMsg').style.display = 'none';
                document.getElementById('status').textContent = '✅ 클라우드에서 얼굴 분석 중...';
                
                document.getElementById('totalScore').textContent = data.total_score;
                document.getElementById('assessment').textContent = data.assessment;
                asymmetryChart.data.datasets[0].data = [
                    data.mouth_diff || 0,
                    data.eye_diff || 0,
                    data.nose_offset || 0
                ];
                asymmetryChart.update();
                
                drawGuideLines(data.landmarks_coords);
                
                if (data.original_image) {
                    document.getElementById('originalImage').src = `data:image/jpeg;base64,${data.original_image}`;
                    document.getElementById('leftSymmetryImage').src = `data:image/jpeg;base64,${data.left_symmetric_image}`;
                    document.getElementById('rightSymmetryImage').src = `data:image/jpeg;base64,${data.right_symmetric_image}`;
                }
            }

            function drawGuideLines(landmarks_coords) {
                overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                if (!landmarks_coords || landmarks_coords.length < 6) return;

                overlayCtx.lineWidth = 3;
                overlayCtx.setLineDash([]);
                
                overlayCtx.shadowColor = "rgba(0, 0, 0, 0.8)";
                overlayCtx.shadowBlur = 3;
                
                // 눈 가로선 (파란색)
                overlayCtx.strokeStyle = "#00BFFF";
                overlayCtx.beginPath();
                overlayCtx.moveTo(landmarks_coords[0][0], landmarks_coords[0][1]);
                overlayCtx.lineTo(landmarks_coords[1][0], landmarks_coords[1][1]);
                overlayCtx.stroke();

                // 입 가로선 (빨간색)
                overlayCtx.strokeStyle = "#FF6B6B";
                overlayCtx.beginPath();
                overlayCtx.moveTo(landmarks_coords[4][0], landmarks_coords[4][1]);
                overlayCtx.lineTo(landmarks_coords[5][0], landmarks_coords[5][1]);
                overlayCtx.stroke();

                // 세로 중심선 (노란색)
                overlayCtx.strokeStyle = "#FFD700";
                overlayCtx.lineWidth = 4;
                overlayCtx.beginPath();
                overlayCtx.moveTo(landmarks_coords[2][0], landmarks_coords[2][1]);
                overlayCtx.lineTo(landmarks_coords[3][0], landmarks_coords[3][1]);
                overlayCtx.stroke();
                
                overlayCtx.shadowColor = "transparent";
                overlayCtx.shadowBlur = 0;
            }

            document.addEventListener('DOMContentLoaded', (event) => {
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('captureBtn').disabled = true;
                initializeChart();
            });
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global captured_images, last_frame_data
    await websocket.accept()
    print("클라우드 클라이언트 연결됨")
    
    try:
        while True:
            message_type = await websocket.receive_text()
            
            if message_type == "manual_capture_request":
                if last_frame_data["frame"] is not None and last_frame_data["landmarks"] is not None:
                    print("클라우드 캡처 요청 수신")
                    symmetry_images = create_symmetry_images_simple(
                        last_frame_data["frame"], 
                        last_frame_data["landmarks"]["accurate_center_x"]
                    )
                    captured_images.update(symmetry_images)
                    
                    result_to_send = last_frame_data["landmarks"].copy()
                    result_to_send.update(captured_images)
                    await websocket.send_text(json.dumps(result_to_send, ensure_ascii=False))
                else:
                    await websocket.send_text(json.dumps({"error": "캡처할 얼굴 데이터가 없습니다."}, ensure_ascii=False))
                continue
            
            try:
                image_data = base64.b64decode(message_type)
                image = Image.open(BytesIO(image_data))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                if USE_MEDIAPIPE:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        result = analyze_face_asymmetry_mediapipe(landmarks, frame.shape[1], frame.shape[0])
                        
                        if result:
                            last_frame_data["frame"] = frame.copy()
                            last_frame_data["landmarks"] = result
                            
                            if captured_images:
                                result.update(captured_images)
                            
                            await websocket.send_text(json.dumps(result, ensure_ascii=False))
                        else:
                            await websocket.send_text(json.dumps({"error": "얼굴 분석 실패"}, ensure_ascii=False))
                    else:
                        await websocket.send_text(json.dumps({"error": "얼굴을 찾을 수 없습니다"}, ensure_ascii=False))
                else:
                    await websocket.send_text(json.dumps({"error": "MediaPipe 로드 실패"}, ensure_ascii=False))
                    
            except Exception as e:
                print(f"프레임 처리 오류: {e}")
                await websocket.send_text(json.dumps({"error": f"처리 오류: {str(e)}"}, ensure_ascii=False))
                
    except Exception as e:
        print(f"WebSocket 오류: {e}")
    finally:
        print("클라우드 클라이언트 연결 해제됨")
        captured_images.clear()
        last_frame_data["frame"] = None
        last_frame_data["landmarks"] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "성형외과용 미세표정 분석 시스템이 정상 작동 중입니다."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
