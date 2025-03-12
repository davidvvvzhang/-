from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import io
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import webbrowser
import threading

app = Flask(__name__)

# 加载您的YOLO模型
model = YOLO('runs/detect/train5/weights/best.pt')

fruit_classes = ['apple', 'orange', 'banana']  # 定义水果类别

def read_image_from_base64(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return img

@app.route('/')
def home():
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>What fruit is this?</title>
            <style>
                body {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                    font-family: Arial, sans-serif;
                    background-color: #fff;
                    position: relative;
                    overflow: hidden;
                }
                #uploadForm {
                    margin-bottom: 20px;
                    text-align: center;
                    z-index: 10;
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 10px;
                    border-radius: 10px;
                }
                #imageInput {
                    display: none;
                }
                #uploadButton {
                    padding: 10px 20px;
                    font-size: 16px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                #uploadButton:hover {
                    background-color: #45a049;
                }
                #fruitImage, #resultImage {
                    max-width: 400px;
                    max-height: 400px;
                    width: auto;
                    height: auto;
                    border: 1px solid #ccc;
                    display: none;
                    z-index: 20;
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 10px;
                    border-radius: 10px;
                }
                #waitMessage {
                    display: none;
                    font-size: 1.2em;
                    margin-top: 20px;
                    z-index: 20;
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 10px;
                    border-radius: 10px;
                }
                .background-word {
                    position: absolute;
                    font-size: 2em;
                    font-weight: bold;
                    opacity: 0.1;
                    z-index: 1;
                }
            </style>
        </head>
        <body>
            <h1 style="z-index: 10;">What fruit is this?</h1>
            <form id="uploadForm">
                <input type="file" id="imageInput" accept="image/*" required>
                <label for="imageInput" id="uploadButton">Upload Image</label>
            </form>
            <img id="fruitImage" src="" alt="Uploaded Fruit Image">
            <img id="resultImage" src="" alt="Detected Fruit Image">
            <div id="waitMessage">Wait a moment...</div>

            <script>
                document.getElementById('imageInput').addEventListener('change', function(event) {
                    const imageInput = document.getElementById('imageInput').files[0];
                    if (imageInput) {
                        document.getElementById('fruitImage').style.display = 'none';
                        document.getElementById('resultImage').style.display = 'none';
                        document.getElementById('waitMessage').style.display = 'none';
                        document.getElementById('waitMessage').textContent = 'Wait a moment...';

                        const reader = new FileReader();
                        reader.onload = function(e) {
                            document.getElementById('fruitImage').src = e.target.result;
                            document.getElementById('fruitImage').style.display = 'block';
                            document.getElementById('waitMessage').style.display = 'block';
                            detectFruit(e.target.result);
                        };
                        reader.readAsDataURL(imageInput);
                    }
                });

                function detectFruit(imageData) {
                    fetch('/detect', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ image: imageData })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.resultImage) {
                            document.getElementById('fruitImage').style.display = 'none';
                            document.getElementById('resultImage').src = data.resultImage;
                            document.getElementById('resultImage').style.display = 'block';
                            document.getElementById('waitMessage').textContent = data.fruitName;
                        } else {
                            document.getElementById('fruitImage').style.display = 'none';
                            document.getElementById('resultImage').style.display = 'none';
                            document.getElementById('waitMessage').textContent = 'OOPS, NO FRUIT...';
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error detecting fruit.');
                        document.getElementById('waitMessage').style.display = 'none';
                    });
                }

                function createWordDiv(word, color) {
                    var wordDiv = document.createElement("div");
                    wordDiv.className = 'background-word';
                    wordDiv.style.color = color;
                    wordDiv.textContent = word;
                    wordDiv.style.transform = "rotate(" + (Math.random() * 30 - 15) + "deg)";
                    return wordDiv;
                }

                function isOverlap(wordDiv, existingDivs) {
                    var rect1 = wordDiv.getBoundingClientRect();
                    for (var i = 0; i < existingDivs.length; i++) {
                        var rect2 = existingDivs[i].getBoundingClientRect();
                        var overlap = !(rect1.right < rect2.left || 
                                        rect1.left > rect2.right || 
                                        rect1.bottom < rect2.top || 
                                        rect1.top > rect2.bottom);
                        if (overlap) {
                            return true;
                        }
                    }
                    return false;
                }

                function generateBackgroundWords() {
                    var words = [
                        { word: "apple", color: "red" },
                        { word: "banana", color: "yellow" },
                        { word: "orange", color: "orange" }
                    ];
                    var body = document.body;
                    var width = window.innerWidth;
                    var height = window.innerHeight;
                    var existingDivs = [];

                    for (var i = 0; i < 50; i++) {
                        var wordObj = words[Math.floor(Math.random() * words.length)];
                        var wordDiv = createWordDiv(wordObj.word, wordObj.color);
                        var attempts = 0;

                        do {
                            wordDiv.style.left = Math.random() * width + "px";
                            wordDiv.style.top = Math.random() * height + "px";
                            attempts++;
                        } while (isOverlap(wordDiv, existingDivs) && attempts < 100);

                        if (attempts < 100) {
                            body.appendChild(wordDiv);
                            existingDivs.push(wordDiv);
                        }
                    }
                }

                document.addEventListener("DOMContentLoaded", generateBackgroundWords);
            </script>
        </body>
        </html>
    """)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    img0 = read_image_from_base64(image_data)
    print('Image received and decoded')

    results = model(img0)
    print('Prediction completed')

    detected_fruits = {}
    for result in results:
        for box in result.boxes:
            cls = box.cls.item()
            class_name = model.names[int(cls)]
            if class_name in fruit_classes:
                if class_name in detected_fruits:
                    detected_fruits[class_name] += 1
                else:
                    detected_fruits[class_name] = 1

    if detected_fruits:
        img = results[0].plot()
        _, buffer = cv2.imencode('.png', img)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        print('Result image encoded to base64')

        fruit_name = ""
        for fruit, count in detected_fruits.items():
            if count == 1:
                fruit_name += f"{fruit.upper()}!!! "
            else:
                fruit_name += f"{fruit.upper()}S!!! "
        return jsonify({'resultImage': 'data:image/png;base64,' + result_image_base64, 'fruitName': fruit_name.strip()})
    else:
        return jsonify({'resultImage': None, 'fruitName': None})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
